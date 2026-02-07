from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import ChannelMLP
from .complex import CGELU, apply_complex, ctanh, ComplexValued
from .normalization_layers import AdaIN, InstanceNorm
from .skip_connections import skip_connection
from .spectral_convolution import SpectralConv, EqSpectralConv
from ..utils import validate_scaling_factor

Number = Union[int, float]
                
class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)

class ComplexGELU(nn.Module):
    """Complex-valued GELU activation"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        if x.is_complex():
            return F.gelu(x.real).type(x.dtype) + 1j * F.gelu(x.imag).type(x.dtype)
        else:
            return F.gelu(x)
        
class FNOBlocks(nn.Module):
    """FNOBlocks implements a sequence of Fourier layers, the operations of which 
    are first described in [1]_. The exact implementation details of the Fourier 
    layer architecture are discussed in [2]_.

    Parameters
    ----------
    in_channels : int
        input channels to Fourier layers
    out_channels : int
        output channels after Fourier layers
    n_modes : int, List[int]
        number of modes to keep along each dimension 
        in frequency space. Can either be specified as
        an int (for all dimensions) or an iterable with one
        number per dimension
    resolution_scaling_factor : Optional[Union[Number, List[Number]]], optional
        factor by which to scale outputs for super-resolution, by default None
    n_layers : int, optional
        number of Fourier layers to apply in sequence, by default 1
    max_n_modes : int, List[int], optional
        maximum number of modes to keep along each dimension, by default None
    fno_block_precision : str, optional
        floating point precision to use for computations, by default "full"
    channel_mlp_dropout : int, optional
        dropout parameter for self.channel_mlp, by default 0
    channel_mlp_expansion : float, optional
        expansion parameter for self.channel_mlp, by default 0.5
    non_linearity : torch.nn.F module, optional
        nonlinear activation function to use between layers, by default F.gelu
    stabilizer : Literal["tanh"], optional
        stabilizing module to use between certain layers, by default None
        if "tanh", use tanh
    norm : Literal["ada_in", "group_norm", "instance_norm"], optional
        Normalization layer to use, by default None
    ada_in_features : int, optional
        number of features for adaptive instance norm above, by default None
    preactivation : bool, optional
        whether to call forward pass with pre-activation, by default False
        if True, call nonlinear activation and norm before Fourier convolution
        if False, call activation and norms after Fourier convolutions
    fno_skip : str, optional
        module to use for FNO skip connections, by default "linear"
        see layers.skip_connections for more details
    channel_mlp_skip : str, optional
        module to use for ChannelMLP skip connections, by default "soft-gating"
        see layers.skip_connections for more details

    Other Parameters
    -------------------
    complex_data : bool, optional
        whether the FNO's data takes on complex values in space, by default False
    separable : bool, optional
        separable parameter for SpectralConv, by default False
    factorization : str, optional
        factorization parameter for SpectralConv, by default None
    rank : float, optional
        rank parameter for SpectralConv, by default 1.0
    conv_module : BaseConv, optional
        module to use for convolutions in FNO block, by default SpectralConv
    joint_factorization : bool, optional
        whether to factorize all spectralConv weights as one tensor, by default False
    fixed_rank_modes : bool, optional
        fixed_rank_modes parameter for SpectralConv, by default False
    implementation : str, optional
        implementation parameter for SpectralConv, by default "factorized"
    decomposition_kwargs : _type_, optional
        kwargs for tensor decomposition in SpectralConv, by default dict()
    
    References
    -----------
    .. [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential 
           Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.
    .. [2] Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
           Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024). 
           TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        resolution_scaling_factor=None,
        n_layers=1,
        max_n_modes=None,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        complex_data=False,
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=SpectralConv,
        fixed_rank_modes=False, 
        implementation="factorized", 
        decomposition_kwargs=dict(),
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.n_dim, n_layers)

        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.channel_mlp_skip = channel_mlp_skip
        self.complex_data = complex_data

        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout

        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features

        # apply real nonlin if data is real, otherwise CGELU
        if self.complex_data:
            self.non_linearity = CGELU
        else:
            self.non_linearity = non_linearity
        
        self.convs = nn.ModuleList([
                conv_module(
                self.in_channels,
                self.out_channels,
                self.n_modes,
                resolution_scaling_factor=None if resolution_scaling_factor is None else self.resolution_scaling_factor[i],
                max_n_modes=max_n_modes,
                rank=rank,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                fno_block_precision=fno_block_precision,
                decomposition_kwargs=decomposition_kwargs,
                complex_data=complex_data
            ) 
            for i in range(n_layers)])

        self.fno_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=fno_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        if self.complex_data:
            self.fno_skips = nn.ModuleList(
                [ComplexValued(x) for x in self.fno_skips]
                )

        self.channel_mlp = nn.ModuleList(
            [
                ChannelMLP(
                    in_channels=self.out_channels,
                    hidden_channels=round(self.out_channels * channel_mlp_expansion),
                    dropout=channel_mlp_dropout,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        if self.complex_data:
            self.channel_mlp = nn.ModuleList(
                [ComplexValued(x) for x in self.channel_mlp]
            )
        self.channel_mlp_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=channel_mlp_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        if self.complex_data:
            self.channel_mlp_skips = nn.ModuleList(
                [ComplexValued(x) for x in self.channel_mlp_skips]
            )

        # Each block will have 2 norms if we also use a ChannelMLP
        self.n_norms = 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                    [
                        InstanceNorm()
                        for _ in range(n_layers * self.n_norms)
                    ]
                )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        
        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, ada_in]"
            )

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips[index](x)
        x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)
        #self.convs(x, index, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = x_fno + x_skip_fno

        if (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        x = self.channel_mlp[index](x) + x_skip_channel_mlp

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        return x

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips[index](x)
        x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)

        x = x_fno + x_skip_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        x = self.channel_mlp[index](x) + x_skip_channel_mlp

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)


class EqFNOBlocks(nn.Module):
    """
    Rotation-Equivariant FNO Blocks via Isotropic Weight Re-parameterization.
    Prevents weight divergence by sharing parameters across the same radius.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        resolution_scaling_factor=None,
        n_layers=1,
        max_n_modes=None,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        complex_data=True, 
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=EqSpectralConv,
        fixed_rank_modes=False, 
        implementation="reconstructed", 
        decomposition_kwargs=dict(),
        num_groups=1,
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.resolution_scaling_factor = validate_scaling_factor(resolution_scaling_factor, self.n_dim, n_layers)

        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.channel_mlp_skip = channel_mlp_skip
        self.complex_data = complex_data

        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout
        
        # [Important] Force 'reconstructed' to handle dense tensors manually
        self.implementation = "reconstructed" 
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features

        # Channel groups for group convolutions
        self.num_groups = num_groups
        # Use ComplexGELU for complex data
        if self.complex_data:
            self.non_linearity = ComplexGELU()
        else:
            self.non_linearity = non_linearity
        
        self.convs = nn.ModuleList([
                conv_module(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                n_modes=self.n_modes,
                resolution_scaling_factor=None if resolution_scaling_factor is None else self.resolution_scaling_factor[i],
                max_n_modes=max_n_modes,
                rank=rank,
                fixed_rank_modes=fixed_rank_modes,
                # Force dense implementation without factorization
                implementation='reconstructed', 
                factorization=None,             
                separable=False,                
                num_groups=self.num_groups,
                fno_block_precision=fno_block_precision,
                decomposition_kwargs=decomposition_kwargs,
                complex_data=complex_data
            ) 
            for i in range(n_layers)])

        # Re-parameterize weights to enforce strict isotropy
        for layer in self.convs:
            self._reparameterize_layer_as_isotropic(layer)

        self.fno_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=fno_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        
        self.channel_mlp = nn.ModuleList(
            [
                ChannelMLP(
                    in_channels=self.out_channels,
                    hidden_channels=round(self.out_channels * channel_mlp_expansion),
                    dropout=channel_mlp_dropout,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        
        self.channel_mlp_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=channel_mlp_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )

        # Apply ComplexValued wrapper to handle complex inputs
        if self.complex_data:
            self.fno_skips = nn.ModuleList([ComplexValued(x) for x in self.fno_skips])
            self.channel_mlp = nn.ModuleList([ComplexValued(x) for x in self.channel_mlp])
            self.channel_mlp_skips = nn.ModuleList([ComplexValued(x) for x in self.channel_mlp_skips])

        self.n_norms = 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList([InstanceNorm() for _ in range(n_layers * self.n_norms)])
        elif norm == "group_norm":
            self.norm = nn.ModuleList([nn.GroupNorm(num_groups=1, num_channels=self.out_channels) for _ in range(n_layers * self.n_norms)])
        elif norm == "ada_in":
            self.norm = nn.ModuleList([AdaIN(ada_in_features, out_channels) for _ in range(n_layers * self.n_norms)])
        else:
            raise ValueError(f"Got norm={norm} but expected None or one of [instance_norm, group_norm, ada_in]")

    def get_k_grid(self, n_modes, device):
        """
        [Symmetric Slicing]
        Generates K values in the EXACT order that EqSpectralConv slices them.
        Order: [0, 1, ..., k_pos-1] followed by [-k_neg, ..., -1]
        This ensures 'layer.weight' is constructed with correct radii.
        """
        k1_size, k2_size, k3_size = n_modes
        
        def symmetric_k_values(n_mode):
            # Must match EqSpectralConv logic
            k = n_mode // 2
            k_pos = k + (n_mode % 2)
            k_neg = k
            
            # [0, 1, ..., k_pos-1]
            pos = torch.arange(k_pos, device=device)
            # [-k_neg, ..., -1]
            neg = torch.arange(-k_neg, 0, device=device)
            return torch.cat([pos, neg])

        k1 = symmetric_k_values(k1_size)
        k2 = symmetric_k_values(k2_size)
        k3 = symmetric_k_values(k3_size)
        
        K1, K2, K3 = torch.meshgrid(k1, k2, k3, indexing='ij')
        return K1, K2, K3
    
    def _reparameterize_layer_as_isotropic(self, layer):
        if hasattr(layer.weight, 'tensor'):
            w_tensor = layer.weight.tensor
        else:
            w_tensor = layer.weight
            
        device = w_tensor.device
        spatial_shape = w_tensor.shape[-3:] 
        
        # 1. Get K-grid
        K1, K2, K3 = self.get_k_grid(spatial_shape, device)
        R2 = (K1**2 + K2**2 + K3**2).float()
        
        unique_radii, inverse_indices = torch.unique(R2, return_inverse=True)
        n_unique = len(unique_radii)
        
        # Register buffer so it can be reloaded during retraining
        layer.register_buffer('radius_indices', inverse_indices)
        
        leading_shape = w_tensor.shape[:-3] 
        
        with torch.no_grad():
            # Create new radial weights: (*Leading_Dims, n_unique)
            new_weight_data = torch.zeros((*leading_shape, n_unique), 
                                          dtype=w_tensor.dtype, device=device)
            
            if w_tensor.is_complex():
                 new_weight_data = new_weight_data.to(w_tensor.dtype)
            
            # Flatten spatial dims for efficient masking
            # w_tensor: (*Leading, X, Y, Z) -> (*Leading, X*Y*Z)
            w_flat = w_tensor.reshape(*leading_shape, -1)
            
            # Flatten indices: (X, Y, Z) -> (X*Y*Z)
            indices_flat = inverse_indices.flatten()
            
            for idx in range(n_unique):
                mask = (indices_flat == idx) # (X*Y*Z,)
                if mask.sum() > 0:
                    # Average only weights for this radius
                    # selected: (*Leading, Count)
                    selected = w_flat[..., mask]
                    # Mean over last dim (Count) -> (*Leading)
                    new_weight_data[..., idx] = selected.mean(dim=-1)

        # Remove original weight and register radial_weight
        if 'weight' in layer._parameters: del layer._parameters['weight']
        if 'weight' in layer._modules: del layer._modules['weight']
        if hasattr(layer, 'weight'): del layer.weight
        
        layer.radial_weight = nn.Parameter(new_weight_data)

    def _construct_full_weight(self, layer):
        """Reconstructs the full grid weight from radial parameters."""
        # Broadcasting: (In, Out, n_unique) -> (In, Out, k1, k2, k3)
        return layer.radial_weight[..., layer.radius_indices]

    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        layer = self.convs[index]

        # Assign the reconstructed weight to 'layer.weight' so SpectralConv can use it.
        layer.weight = self._construct_full_weight(layer)

        # 1. Ensure Complex Input (Forces Full FFT logic in SpectralConv)
        if self.complex_data and not x.is_complex():
            x_input = x.to(torch.cfloat)
        else:
            x_input = x

        # 2. Skip Connections
        x_skip_fno = self.fno_skips[index](x_input)
        x_skip_fno = layer.transform(x_skip_fno, output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips[index](x_input)
        x_skip_channel_mlp = layer.transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_data:
                x_input = ctanh(x_input)
            else:
                x_input = torch.tanh(x_input)

        # 3. Main FNO Path
        # SpectralConv will use 'layer.weight' which is perfectly isotropic.
        x_fno = layer(x_input, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = x_fno + x_skip_fno

        if (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        x = self.channel_mlp[index](x) + x_skip_channel_mlp

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        # Return real if needed
        return x.real if self.complex_data else x 

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        layer = self.convs[index]
        layer.weight = self._construct_full_weight(layer)
        
        if self.complex_data and not x.is_complex():
            x = x.to(torch.cfloat)

        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = layer.transform(x_skip_fno, output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips[index](x)
        x_skip_channel_mlp = layer.transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
             if self.complex_data:
                x = ctanh(x)
             else:
                x = torch.tanh(x)

        x_fno = layer(x, output_shape=output_shape)

        x = x_fno + x_skip_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        x = self.channel_mlp[index](x) + x_skip_channel_mlp

        return x.real if self.complex_data else x
    
    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        if self.n_layers == 1:
            raise ValueError("A single layer is parametrized, directly use the main class.")
        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)
    
    def set_ada_in_embeddings(self, *embeddings):
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)


