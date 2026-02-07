import torch
import pickle
import trimesh
import lightning.pytorch as pl
from neuralop.models import EqGINO
from sklearn.metrics import r2_score
from torch.utils.data import Dataset
from torch.nn import Linear, Sequential
from utils import apply_radial_basis, get_rotation_matrix, compute_metrics

# __________________
# Data Preprocessing
def preprocess(data, args):

    # Rotation Augmentation
    rotation_matrix = get_rotation_matrix(args.aug_type, device=data.x.device)
    data.x = data.x - data.x.mean(dim=0, keepdim=True)
    data.x = torch.matmul(data.x, rotation_matrix.T)
    data.inlet_vel_direction = torch.matmul(data.inlet_vel_direction, rotation_matrix.T)
    
    # AhmedBody
    if '3d_ab_wss' in args.tgt_y:
        data.y_wallShearStress = torch.matmul(data.y_wallShearStress, rotation_matrix.T)
        data.y = data.y_wallShearStress
    elif '3d_ab_p' in args.tgt_y:
        data.y = data.y_p
    elif '3d_ab_k' in args.tgt_y:
        data.y = data.y_k
    elif '3d_ab_omega' in args.tgt_y:
        data.y = torch.log(data.y_omega + 1e-8)
    elif '3d_ab_nut' in args.tgt_y:
        data.y = torch.log(data.y_nut + 1e-8)
    # ShapeNetCar
    elif '3d_snc_press' in args.tgt_y:
        data.conds_feat = torch.ones((data.x.shape[0],1)).float()
        data.y = data.y.unsqueeze(-1).float()

    data.conds_feat = data.conds_feat.expand(data.x.shape[0], -1)
    data.inlet_vel_direction = data.inlet_vel_direction.repeat(data.x.shape[0], 1)

    # Use invariant node features
    data.node_attr = data.conds_feat

    return data

class LargeDataset(Dataset):
    def __init__(self, dataset_files, basepath, args):
        self.dataset_files = dataset_files
        self.basepath = basepath
        self.args = args

    def __len__(self):
        return len(self.dataset_files)

    def __getitem__(self, idx):
        file_path = self.dataset_files[idx]
        with open(f"{self.basepath}/{file_path}", "rb") as f:
            data = pickle.load(f)

        data = preprocess(data, self.args)
        data.shape_id = file_path[:-4]

        return data

class EQGINO(pl.LightningModule):
    def __init__(self, raw_sample_data, args):
        super(EQGINO, self).__init__()
        self.args = args

        raw_sample_data = preprocess(raw_sample_data, args)

        coord_dim_node = raw_sample_data.x.shape[1]
        input_dim_node = raw_sample_data.node_attr.shape[1]
        cond_dim = raw_sample_data.conds_feat.shape[1]
        output_dim = raw_sample_data.y.shape[1]

        # GINO parameters
        self.in_channels = input_dim_node  # Input node feature dimension
        self.out_channels = self.args.hidden_dim  # Output feature dimension
        self.fno_n_mode = self.args.fno_n_mode
        self.fno_n_modes = (self.fno_n_mode, self.fno_n_mode, self.fno_n_mode)  # FNO modes per dimension
        self.fno_hidden_channels = self.args.hidden_dim  # FNO hidden channels
        self.gno_coord_dim = coord_dim_node  # GINO coordinate dimension
        self.gno_radius = self.args.gno_radius  # GINO radius
        self.fno_n_layers = self.args.fno_n_layers  # Number of FNO layers
        self.fno_norm = None  # FNO normalization

        self.operator = EqGINO(
            in_channels=self.in_channels,
            out_channels=output_dim,
            gno_coord_dim=self.gno_coord_dim,
            gno_radius=self.gno_radius,
            fno_n_modes=self.fno_n_modes,
            fno_hidden_channels=self.fno_hidden_channels,
            fno_n_layers=self.fno_n_layers,
            fno_norm=self.fno_norm,
            fno_in_channels=self.in_channels,
            latent_feature_channels=cond_dim, 
            num_groups=self.args.num_groups,
            )

        self.decoder = Sequential(Linear(args.hidden_dim , args.hidden_dim),
                            Linear(args.hidden_dim, output_dim),
                            )

    def generate_bounding_latent_queries(self, grid_size):

        min_val = -1
        max_val = 1

        x = torch.linspace(min_val, max_val, grid_size[0])
        y = torch.linspace(min_val, max_val, grid_size[1])
        z = torch.linspace(min_val, max_val, grid_size[2])

        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")  # (nx, ny, nz)

        latent_queries = torch.stack((X, Y, Z), dim=-1)  # (nx, ny, nz, 3)

        return latent_queries.unsqueeze(0)
    
    def loss(self, pred, inputs):
        labels = inputs.y

        mae=torch.mean(torch.abs(labels-pred))
        error=torch.sum((labels-pred)**2,axis=1)
        loss=torch.sqrt(torch.mean(error)) ## RMSE
        r2 = r2_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())
        return loss, mae, r2

    def forward(self, data):
 
        coord = data.x
        coord = coord - coord.mean(dim=0, keepdim=True)  # Center coordinates at the origin

        # Unit sphere normalization
        max_dist = torch.max(torch.norm(coord, dim=1)) + 1e-8
        coord = coord / max_dist

        if self.args.mesh_subsample_rate != 1 and self.trainer.training:
            sampling_rate = 1/self.args.mesh_subsample_rate
            n_nodes = coord.size(0)
            n_samples = int(n_nodes * sampling_rate)
            # Fixed random subsampling for reproducibility
            torch.manual_seed(self.args.seed)
            subsampled_node_idx = torch.randperm(n_nodes, device=coord.device)[:n_samples]

            coord = coord[subsampled_node_idx,:]
            data.x = data.x[subsampled_node_idx,:]
            data.y = data.y[subsampled_node_idx,:]
            data.node_attr = data.node_attr[subsampled_node_idx,:]
            data.inlet_vel_direction = data.inlet_vel_direction[subsampled_node_idx,:]
            data.batch = data.batch[subsampled_node_idx] if hasattr(data, 'batch') else None
            data.conds_feat = data.conds_feat[subsampled_node_idx,:]

        if self.args.mesh_subsample_rate_valid != 1 and (self.trainer.validating or self.trainer.testing):
            sampling_rate = 1/self.args.mesh_subsample_rate_valid
            n_nodes = coord.size(0)
            n_samples = int(n_nodes * sampling_rate)
            # Fixed random subsampling for reproducibility
            torch.manual_seed(self.args.seed)
            subsampled_node_idx = torch.randperm(n_nodes, device=coord.device)[:n_samples]

            coord = coord[subsampled_node_idx,:]
            data.x = data.x[subsampled_node_idx,:]
            data.y = data.y[subsampled_node_idx,:]
            data.node_attr = data.node_attr[subsampled_node_idx,:]
            data.inlet_vel_direction = data.inlet_vel_direction[subsampled_node_idx,:]
            data.batch = data.batch[subsampled_node_idx] if hasattr(data, 'batch') else None
            data.conds_feat = data.conds_feat[subsampled_node_idx,:]

        latent_queries = self.generate_bounding_latent_queries((self.fno_n_mode, self.fno_n_mode, self.fno_n_mode)).to(coord.device)

        cond_feat = data.conds_feat[0,:].unsqueeze(0)
        latent_feature_dim = cond_feat.shape[1]  
        latent_features = cond_feat.view(self.args.batch_size, 1, 1, 1, latent_feature_dim)
        latent_features = latent_features.expand(self.args.batch_size, *latent_queries.shape[1:-1], latent_feature_dim)

        pred = self.operator(input_geom=coord.unsqueeze(0), latent_queries=latent_queries, latent_features=latent_features, output_queries=coord, x=data.node_attr.unsqueeze(0)).squeeze(0)

        if pred.shape[1] == 3:
            pred = apply_radial_basis(pred, data.x, data.inlet_vel_direction)

        return pred, []
        
        
    def training_step(self, batch, batch_idx):
        pred, utils = self(batch)

        rmse, mae, r2 = self.loss(pred, batch)

        self.log("Train RMSE", rmse, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Train MAE", mae, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Train R2", r2, prog_bar=True, batch_size=self.args.batch_size)

        loss = mae
            
        return loss

    def validation_step(self, batch, batch_idx):
        pred, _ = self(batch)

        batch_index = batch.batch if hasattr(batch, 'batch') else None
        metrics = compute_metrics(pred, batch.y, batch_index)
        
        for k, v in metrics.items():
            self.log(f"Valid {k}", v, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        pred, _ = self(batch)

        batch_index = batch.batch if hasattr(batch, 'batch') else None
        metrics = compute_metrics(pred, batch.y, batch_index)
        
        for k, v in metrics.items():
            self.log(f"Test {k}", v, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)

        return {
            "optimizer": optimizer,
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        sd = checkpoint.get("state_dict", checkpoint)
        if "_metadata" in sd:
            sd.pop("_metadata")