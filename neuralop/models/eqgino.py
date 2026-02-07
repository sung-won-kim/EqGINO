import torch
import torch.nn.functional as F
from .base_model import BaseModel

from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import SinusoidalEmbedding
from ..layers.gno_block import EqGNOBlock
from ..layers.fno_block import EqFNOBlocks
from ..layers.spectral_convolution import EqSpectralConv

class EqGINO(BaseModel):
    def __init__(
        self,
        in_channels,
        out_channels,
        latent_feature_channels=None,
        projection_channels=256,
        gno_coord_dim=3,
        gno_radius=0.033,
        in_gno_transform_type='linear',
        out_gno_transform_type='linear',
        gno_pos_embed_type='transformer',
        fno_in_channels=3,
        fno_n_modes=(16, 16, 16), 
        fno_hidden_channels=64,
        fno_lifting_channel_ratio=2,
        fno_n_layers=4,
        # Other GNO Params
        gno_embed_channels=32,
        gno_embed_max_positions=10000,
        in_gno_channel_mlp_hidden_layers=[80, 80, 80],
        out_gno_channel_mlp_hidden_layers=[512, 256],
        gno_channel_mlp_non_linearity=F.gelu, 
        gno_use_open3d=True,
        gno_use_torch_scatter=True,
        out_gno_tanh=None,
        # Other FNO Params
        fno_resolution_scaling_factor=None,
        fno_incremental_n_modes=None,
        fno_block_precision='full',
        fno_use_channel_mlp=True, 
        fno_channel_mlp_dropout=0,
        fno_channel_mlp_expansion=0.5,
        fno_non_linearity=F.gelu,
        fno_stabilizer=None, 
        fno_norm=None,
        fno_ada_in_features=4,
        fno_ada_in_dim=1,
        fno_preactivation=False,
        fno_skip='linear',
        fno_channel_mlp_skip='soft-gating',
        fno_separable=False,
        fno_factorization=None,
        fno_rank=1.0,
        fno_joint_factorization=False, 
        fno_fixed_rank_modes=False,
        fno_implementation='factorized',
        fno_decomposition_kwargs=dict(),
        fno_conv_module=EqSpectralConv,
        num_groups=1,
        **kwargs
        ):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_feature_channels = latent_feature_channels
        self.gno_coord_dim = gno_coord_dim
        self.fno_hidden_channels = fno_hidden_channels
        self.num_groups = num_groups

        self.lifting_channels = fno_lifting_channel_ratio * fno_hidden_channels

        # TODO: make sure this makes sense in all contexts
        if in_gno_transform_type in ["nonlinear", "nonlinear_kernelonly"]:
            in_gno_out_channels = self.in_channels
        else:
            in_gno_out_channels = fno_in_channels

        self.fno_in_channels = in_gno_out_channels

        if latent_feature_channels is not None:
            self.fno_in_channels += latent_feature_channels

        if self.gno_coord_dim != 3 and gno_use_open3d:
            print(f'Warning: GNO expects {self.gno_coord_dim}-d data but Open3d expects 3-d data')
            gno_use_open3d = False

        self.in_coord_dim = len(fno_n_modes)
        self.gno_out_coord_dim = len(fno_n_modes) # gno output and fno will use same dimensions
        if self.in_coord_dim != self.gno_coord_dim:
            print(f'Warning: FNO expects {self.in_coord_dim}-d data while input GNO expects {self.gno_coord_dim}-d data')

        self.in_coord_dim_forward_order = list(range(self.in_coord_dim))
        # channels starting at 2 to permute everything after channel and batch dims
        self.in_coord_dim_reverse_order = [j + 2 for j in self.in_coord_dim_forward_order]

        self.fno_norm = fno_norm
        if self.fno_norm == "ada_in":
            if fno_ada_in_features is not None and gno_pos_embed_type is not None:
                self.adain_pos_embed = SinusoidalEmbedding(in_channels=fno_ada_in_dim,
                                                        num_frequencies=fno_ada_in_features, 
                                                        max_positions=10000,
                                                        embedding_type=gno_pos_embed_type)                    
                self.ada_in_dim = self.adain_pos_embed.out_channels
            else:
                self.ada_in_dim = fno_ada_in_dim
                self.adain_pos_embed = None
        else:
            self.adain_pos_embed = None
            self.ada_in_dim = None
        
        self.gno_radius = gno_radius
        self.out_gno_tanh = out_gno_tanh

        ### input GNO
        self.gno_in = EqGNOBlock(
            in_channels=in_channels,
            out_channels=in_gno_out_channels,
            coord_dim=self.gno_coord_dim,
            pos_embedding_type=gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            radius=gno_radius,
            channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=in_gno_transform_type,
            use_open3d_neighbor_search=gno_use_open3d,
            use_torch_scatter_reduce=gno_use_torch_scatter,
        )

        ### Lifting layer before FNOBlocks
        self.lifting = ChannelMLP(in_channels=self.fno_in_channels,
                                  hidden_channels=self.lifting_channels,
                                  out_channels=fno_hidden_channels,
                                  n_layers=3)
        
        ### FNOBlocks in latent space
        self.fno_blocks = EqFNOBlocks(
                n_modes=fno_n_modes,
                hidden_channels=fno_hidden_channels,
                in_channels=fno_hidden_channels,
                out_channels=fno_hidden_channels,
                positional_embedding=None,
                n_layers=fno_n_layers,
                resolution_scaling_factor=fno_resolution_scaling_factor,
                incremental_n_modes=fno_incremental_n_modes,
                fno_block_precision=fno_block_precision,
                use_channel_mlp=fno_use_channel_mlp,
                channel_mlp_expansion=fno_channel_mlp_expansion,
                channel_mlp_dropout=fno_channel_mlp_dropout,
                non_linearity=fno_non_linearity,
                stabilizer=fno_stabilizer, 
                norm=fno_norm,
                ada_in_features=self.ada_in_dim,
                preactivation=fno_preactivation,
                fno_skip=fno_skip,
                channel_mlp_skip=fno_channel_mlp_skip,
                separable=fno_separable,
                factorization=fno_factorization,
                rank=fno_rank,
                joint_factorization=fno_joint_factorization, 
                fixed_rank_modes=fno_fixed_rank_modes,
                implementation=fno_implementation,
                decomposition_kwargs=fno_decomposition_kwargs,
                domain_padding=None,
                domain_padding_mode=None,
                conv_module=fno_conv_module,
                num_groups=num_groups,
                **kwargs
        )

        ### output GNO
        self.gno_out = EqGNOBlock(
            in_channels=fno_hidden_channels, # number of channels in f_y
            out_channels=fno_hidden_channels,
            coord_dim=self.gno_coord_dim,
            radius=self.gno_radius,
            pos_embedding_type=gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=out_gno_transform_type,
            use_open3d_neighbor_search=gno_use_open3d,
            use_torch_scatter_reduce=gno_use_torch_scatter
        )

        self.projection = ChannelMLP(in_channels=fno_hidden_channels, 
                              out_channels=self.out_channels, 
                              hidden_channels=projection_channels, 
                              n_layers=2, 
                              n_dim=1, 
                              non_linearity=fno_non_linearity) 

    def latent_embedding(self, in_p, ada_in=None):
        in_p = in_p.permute(0, len(in_p.shape)-1, *list(range(1,len(in_p.shape)-1)))
        #Update Ada IN embedding    
        if ada_in is not None:
            if ada_in.ndim == 2:
                ada_in = ada_in.squeeze(0)
            if self.adain_pos_embed is not None:
                ada_in_embed = self.adain_pos_embed(ada_in.unsqueeze(0)).squeeze(0)
            else:
                ada_in_embed = ada_in
            if self.fno_norm == "ada_in":
                self.fno_blocks.set_ada_in_embeddings(ada_in_embed)

        #Apply FNO blocks
        in_p = self.lifting(in_p)

        for idx in range(self.fno_blocks.n_layers):
            in_p = self.fno_blocks(in_p, idx)

        return in_p 

    def forward(self, input_geom, latent_queries, output_queries, x=None, latent_features=None, ada_in=None, **kwargs):
        if x is None: batch_size = 1
        else: batch_size = x.shape[0]
        
        if latent_features is not None:
            if latent_features.shape[0] != batch_size:
                if latent_features.shape[0] == 1:
                    latent_features = latent_features.repeat(batch_size, *[1]*(latent_features.ndim-1))

        input_geom = input_geom.squeeze(0) 
        latent_queries = latent_queries.squeeze(0)
        grid_shape = latent_queries.shape[:-1]

        in_p = self.gno_in(y=input_geom, x=latent_queries.view((-1, 3)), f_y=x)
        in_p = in_p.view((batch_size, *grid_shape, -1))
        if latent_features is not None:
            in_p = torch.cat((in_p, latent_features), dim=-1)
        
        latent_embed = self.latent_embedding(in_p=in_p, ada_in=ada_in)
        
        latent_embed_flat = latent_embed.permute(0, *self.in_coord_dim_reverse_order, 1).reshape(batch_size, -1, self.fno_hidden_channels)
        if self.out_gno_tanh in ['latent_embed', 'both']: latent_embed_flat = torch.tanh(latent_embed_flat)
        
        out = self.gno_out(y=latent_queries.reshape((-1, 3)), x=output_queries, f_y=latent_embed_flat)
        out = out.permute(0, 2, 1)
        out = self.projection(out).permute(0, 2, 1)

        return out
