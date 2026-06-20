import torch
import pickle
import trimesh
import lightning.pytorch as pl
from neuralop.models import EqGINO
from sklearn.metrics import r2_score
from torch.utils.data import Dataset
from torch.nn import Linear, Sequential
from utils import apply_radial_basis, get_rotation_matrix, compute_metrics, torch_r2

DEEPJEB_LOADS = ['hor', 'ver', 'dia', 'tor']

# __________________
# Data Preprocessing
def preprocess(data, args):

    is_deepjeb = args.tgt_y.startswith('3d_dj')

    # Rotation Augmentation
    rotation_matrix = get_rotation_matrix(args.aug_type, device=data.x.device)

    if is_deepjeb:
        # DeepJeb: use surface mesh
        data.x = data.x - data.x.mean(dim=0, keepdim=True)
        data.x = torch.matmul(data.x, rotation_matrix.T)

        # Rotate force directions
        for load in DEEPJEB_LOADS:
            force_dir = getattr(data, f'{load}_force_direction')
            setattr(data, f'{load}_force_direction', torch.matmul(force_dir, rotation_matrix.T))

        # Rotate displacement vectors (for deflection target)
        for load in DEEPJEB_LOADS:
            x_disp = getattr(data, f'y_{load}_x_disp')
            y_disp = getattr(data, f'y_{load}_y_disp')
            z_disp = getattr(data, f'y_{load}_z_disp')
            disp = torch.cat([x_disp, y_disp, z_disp], 1)
            disp = torch.matmul(disp, rotation_matrix.T)
            setattr(data, f'y_{load}_x_disp', disp[:, 0:1])
            setattr(data, f'y_{load}_y_disp', disp[:, 1:2])
            setattr(data, f'y_{load}_z_disp', disp[:, 2:3])

        data.conds_feat = torch.ones((data.x.shape[0], 1)).float()

        # node_attr is already [mindist2fixed, dist2load] (2dim) in publish data
        data.node_attr_raw = data.node_attr[:, -2:].clone().float()

        # Init node_attr with hor load for dimension matching at model init
        data.node_attr = torch.cat([
            data.node_attr_raw,
            data.hor_force_dot,
            data.hor_force_mag
        ], 1).float()

        # Init y with hor load for dimension matching at model init
        if '3d_dj_deflection' in args.tgt_y:
            data.y = torch.cat([
                data.y_hor_x_disp,
                data.y_hor_y_disp,
                data.y_hor_z_disp
            ], 1).float()
        elif '3d_dj_stress' in args.tgt_y:
            data.y = data.y_hor_stress.float()

    else:
        # AhmedBody / ShapeNetCar
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
            data.conds_feat = torch.ones((data.x.shape[0], 1)).float()
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
        self.is_deepjeb = args.tgt_y.startswith('3d_dj')

        raw_sample_data = preprocess(raw_sample_data, args)

        coord_dim_node = raw_sample_data.x.shape[1]
        input_dim_node = raw_sample_data.node_attr.shape[1]
        cond_dim = raw_sample_data.conds_feat.shape[1]
        output_dim = raw_sample_data.y.shape[1]

        # GINO parameters
        self.in_channels = input_dim_node
        self.out_channels = self.args.hidden_dim
        self.fno_n_mode = self.args.fno_n_mode
        self.fno_n_modes = (self.fno_n_mode, self.fno_n_mode, self.fno_n_mode)
        self.fno_hidden_channels = self.args.hidden_dim
        self.gno_coord_dim = coord_dim_node
        self.gno_radius = self.args.gno_radius
        self.fno_n_layers = self.args.fno_n_layers
        self.fno_norm = None

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

        self.decoder = Sequential(Linear(args.hidden_dim, args.hidden_dim),
                            Linear(args.hidden_dim, output_dim),
                            )

        # For deepjeb test-time max/min tracking
        if self.is_deepjeb:
            self.test_preds_dict = {}
            self.test_targets_dict = {}

    def generate_bounding_latent_queries(self, grid_size, device=None):

        # The latent grid is fixed across forward passes (range + resolution are
        # constant), so build it once on the target device and cache it to avoid
        # a CPU meshgrid + host->device copy on every step.
        key = (tuple(grid_size), str(device))
        cache = getattr(self, "_latent_query_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]

        grid_range = self.args.latent_grid_range
        min_val = grid_range[0]
        max_val = grid_range[1]

        x = torch.linspace(min_val, max_val, grid_size[0], device=device)
        y = torch.linspace(min_val, max_val, grid_size[1], device=device)
        z = torch.linspace(min_val, max_val, grid_size[2], device=device)

        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

        latent_queries = torch.stack((X, Y, Z), dim=-1).unsqueeze(0)

        self._latent_query_cache = (key, latent_queries)
        return latent_queries

    def loss(self, pred, labels):
        mae = torch.mean(torch.abs(labels - pred))
        error = torch.sum((labels - pred) ** 2, axis=1)
        loss = torch.sqrt(torch.mean(error))  # RMSE
        r2 = torch_r2(labels, pred)  # on-device R2, avoids per-step GPU->CPU sync
        return loss, mae, r2

    def _subsample(self, coord, data, force_dir=None):
        """Subsample nodes during training or validation/test."""
        is_training = self.trainer.training
        is_eval = self.trainer.validating or self.trainer.testing

        if is_training and self.args.mesh_subsample_rate != 1:
            rate = 1 / self.args.mesh_subsample_rate
        elif is_eval and self.args.mesh_subsample_rate_valid != 1:
            rate = 1 / self.args.mesh_subsample_rate_valid
        else:
            return coord, data, force_dir

        n_nodes = coord.size(0)
        n_samples = int(n_nodes * rate)
        idx = torch.randperm(n_nodes, device=coord.device)[:n_samples]

        coord = coord[idx]
        data.x = data.x[idx]
        data.y = data.y[idx]
        data.node_attr = data.node_attr[idx]
        data.conds_feat = data.conds_feat[idx]
        data.batch = data.batch[idx] if getattr(data, 'batch', None) is not None else None

        if not self.is_deepjeb:
            data.inlet_vel_direction = data.inlet_vel_direction[idx]
        if force_dir is not None:
            force_dir = force_dir[idx]

        return coord, data, force_dir

    def _single_forward(self, coord, data):
        """Single operator forward pass (shared logic)."""
        latent_queries = self.generate_bounding_latent_queries(
            (self.fno_n_mode, self.fno_n_mode, self.fno_n_mode), device=coord.device
        )

        cond_feat = data.conds_feat[0, :].unsqueeze(0)
        latent_feature_dim = cond_feat.shape[1]
        latent_features = cond_feat.view(self.args.batch_size, 1, 1, 1, latent_feature_dim)
        latent_features = latent_features.expand(
            self.args.batch_size, *latent_queries.shape[1:-1], latent_feature_dim
        )

        pred = self.operator(
            input_geom=coord.unsqueeze(0),
            latent_queries=latent_queries,
            latent_features=latent_features,
            output_queries=coord,
            x=data.node_attr.unsqueeze(0)
        ).squeeze(0)

        return pred

    def forward(self, data):

        coord = data.x
        coord = coord - coord.mean(dim=0, keepdim=True)

        # Unit sphere normalization (AhmedBody, ShapeNetCar only)
        if self.args.unit_sphere_normalize:
            max_dist = torch.max(torch.norm(coord, dim=1)) + 1e-8
            coord = coord / max_dist

        if not self.is_deepjeb:
            # ---- AhmedBody / ShapeNetCar: single forward ----
            coord, data, _ = self._subsample(coord, data)

            pred = self._single_forward(coord, data)

            if pred.shape[1] == 3:
                pred = apply_radial_basis(pred, data.x, data.inlet_vel_direction)

            return pred, []

        else:
            # ---- DeepJeb: multi-load forward ----
            coord_orig = coord.clone()
            is_deflection = '3d_dj_deflection' in self.args.tgt_y

            # Build per-load ground truths, node_attrs, force_directions
            gts = []
            node_attrs = []
            force_dirs = []
            for load in DEEPJEB_LOADS:
                if is_deflection:
                    gt = torch.cat([
                        getattr(data, f'y_{load}_x_disp'),
                        getattr(data, f'y_{load}_y_disp'),
                        getattr(data, f'y_{load}_z_disp'),
                    ], 1).float()
                else:
                    gt = getattr(data, f'y_{load}_stress').float()
                gts.append(gt)

                node_attrs.append(torch.cat([
                    data.node_attr_raw,
                    getattr(data, f'{load}_force_dot'),
                    getattr(data, f'{load}_force_mag'),
                ], 1).float())

                force_dirs.append(getattr(data, f'{load}_force_direction'))

            conds_feat_orig = data.conds_feat.clone()
            batch_orig = data.batch.clone() if getattr(data, 'batch', None) is not None else None

            preds = []
            for i, load in enumerate(DEEPJEB_LOADS):
                coord_i = coord_orig.clone()
                data.x = coord_orig.clone()
                data.y = gts[i]
                data.node_attr = node_attrs[i]
                data.conds_feat = conds_feat_orig.clone()
                data.batch = batch_orig.clone() if batch_orig is not None else None

                force_dir_i = force_dirs[i]
                coord_i, data, force_dir_i = self._subsample(coord_i, data, force_dir_i)
                gts[i] = data.y  # update gt after subsampling

                pred = self._single_forward(coord_i, data)

                if is_deflection:
                    pred = apply_radial_basis(pred, data.x, force_dir_i)

                preds.append(pred)

            return preds, [gts, DEEPJEB_LOADS]

    def training_step(self, batch, batch_idx):
        pred, utils = self(batch)

        if self.is_deepjeb:
            output_dim = pred[0].shape[1]
            preds = torch.cat(pred, dim=0)
            ys = torch.cat(utils[0], dim=0)
        else:
            preds = pred
            ys = batch.y

        rmse, mae, r2 = self.loss(preds, ys)

        self.log("Train RMSE", rmse, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Train MAE", mae, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Train R2", r2, prog_bar=True, batch_size=self.args.batch_size)

        return mae

    def validation_step(self, batch, batch_idx):
        pred, utils = self(batch)
        batch_index = batch.batch if hasattr(batch, 'batch') else None

        if self.is_deepjeb:
            self._log_deepjeb_metrics(pred, utils, batch_index, prefix="Valid")
        else:
            metrics = compute_metrics(pred, batch.y, batch_index)
            for k, v in metrics.items():
                self.log(f"Valid {k}", v, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        pred, utils = self(batch)
        batch_index = batch.batch if hasattr(batch, 'batch') else None

        if self.is_deepjeb:
            self._log_deepjeb_metrics(pred, utils, batch_index, prefix="Test", track_minmax=True)
        else:
            metrics = compute_metrics(pred, batch.y, batch_index)
            for k, v in metrics.items():
                self.log(f"Test {k}", v, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)

    def _log_deepjeb_metrics(self, preds, utils, batch_index, prefix, track_minmax=False):
        """Compute and log per-load metrics for deepjeb, then log averaged total."""
        gts, loads = utils
        is_deflection = '3d_dj_deflection' in self.args.tgt_y
        metric_keys = ["MSE", "MAE", "RMSE", "Max_AE", "Rel_L2", "Rel_L1", "R2"]
        total_metrics = {k: 0.0 for k in metric_keys}
        total_num = 0

        for i, load in enumerate(loads):
            if is_deflection:
                # Per-axis metrics for deflection
                for j, axis in enumerate(['x', 'y', 'z']):
                    pred_slice = preds[i][:, j].unsqueeze(1)
                    target_slice = gts[i][:, j].unsqueeze(1)

                    metrics = compute_metrics(pred_slice, target_slice, batch_index)
                    for k, v in metrics.items():
                        self.log(f"{prefix} {k} ({load}_{axis})", v, prog_bar=False,
                                 batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)
                        total_metrics[k] += v if isinstance(v, float) else v.item()
                    total_num += 1

                    if track_minmax:
                        self._track_minmax(f'{load}_{axis}', pred_slice, target_slice)
            else:
                # Scalar metrics for stress
                metrics = compute_metrics(preds[i], gts[i], batch_index)
                for k, v in metrics.items():
                    self.log(f"{prefix} {k} ({load})", v, prog_bar=False,
                             batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)
                    total_metrics[k] += v if isinstance(v, float) else v.item()
                total_num += 1

                if track_minmax:
                    self._track_minmax(load, preds[i], gts[i])

        for k in metric_keys:
            self.log(f"{prefix} {k}", total_metrics[k] / total_num, prog_bar=True,
                     batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)

    def _track_minmax(self, key, pred, target):
        """Track max/min predictions for test-time reporting."""
        for suffix, fn in [('max', torch.max), ('min', torch.min)]:
            dict_key = f'{key}_{suffix}'
            if dict_key not in self.test_preds_dict:
                self.test_preds_dict[dict_key] = []
                self.test_targets_dict[dict_key] = []
            self.test_preds_dict[dict_key].append(fn(pred).item())
            self.test_targets_dict[dict_key].append(fn(target).item())

    def on_test_epoch_end(self):
        if self.is_deepjeb:
            for key in self.test_preds_dict:
                preds = torch.tensor(self.test_preds_dict[key]).unsqueeze(1)
                targets = torch.tensor(self.test_targets_dict[key]).unsqueeze(1)
                metrics = compute_metrics(preds, targets, batch_index=None)
                for k, v in metrics.items():
                    self.log(f"Test {k} ({key})", v, prog_bar=True,
                             batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)
            self.test_preds_dict.clear()
            self.test_targets_dict.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return {"optimizer": optimizer}

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        sd = checkpoint.get("state_dict", checkpoint)
        if "_metadata" in sd:
            sd.pop("_metadata")
