import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch_geometric.nn import global_add_pool

def apply_radial_basis(predictions, vertices, inlet_vectors, epsilon=1e-8):
 
    force_norm = inlet_vectors.norm(p=2, dim=1, keepdim=True)
    e1 = inlet_vectors / (force_norm + epsilon)

    centroid = torch.mean(vertices, dim=0, keepdim=True)
    radial_vector = vertices - centroid
    r_unit = F.normalize(radial_vector, p=2, dim=1)

    e2_raw = torch.cross(e1, r_unit, dim=1)
    e2_norm = e2_raw.norm(p=2, dim=1, keepdim=True)
    e2 = e2_raw / (e2_norm + epsilon)

    e3 = torch.cross(e1, e2, dim=1)

    global_deflection = (
        (predictions[:, 0:1] * e1)
        + (predictions[:, 1:2] * e2)
        + (predictions[:, 2:3] * e3)
    )

    return global_deflection

def get_rotation_matrix(mode, device='cpu'):

    # 1. Canonical (No Rotation)
    if mode == 'canonical':
        return torch.eye(3, device=device, dtype=torch.float32)

    # 2. Setup Axis and Theta
    axis = np.random.choice(["x", "y", "z"])
    
    if mode == 'discrete':
        # 0, 90, 180, 270 degrees converted to radians
        theta = np.radians(np.random.choice([0, 90, 180, 270]))
    elif mode == 'arbitrary':
        # Continuous 0 ~ 360 degrees (0 ~ 2pi radians)
        theta = np.random.uniform(0, 2 * np.pi)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'discrete', 'arbitrary', or 'canonical'.")

    # 3. Construct Matrix
    c = np.cos(theta)
    s = np.sin(theta)

    if axis == "x":
        matrix = [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ]
    elif axis == "y":
        matrix = [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ]
    else: # axis == "z"
        matrix = [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ]

    return torch.tensor(matrix, device=device, dtype=torch.float32)

    
def torch_r2(targets, pred):
    """GPU R2 score matching sklearn's `multioutput='uniform_average'`.

    Avoids the per-step GPU->CPU sync + numpy round-trip incurred by
    sklearn.metrics.r2_score, which otherwise stalls the CUDA pipeline on
    every training/validation step.
    """
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
        pred = pred.unsqueeze(1)
    ss_res = ((targets - pred) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    nonzero = ss_tot != 0
    r2 = torch.ones_like(ss_tot)
    r2 = torch.where(nonzero, 1 - ss_res / ss_tot.clamp_min(1e-12), r2)
    # constant-target outputs: perfect fit -> 1.0, otherwise 0.0 (sklearn rule)
    r2 = torch.where((~nonzero) & (ss_res != 0), torch.zeros_like(r2), r2)
    return r2.mean()


def compute_metrics(pred, targets, batch_index=None):
    """
    Calculates MSE, MAE, RMSE, Max AE, Rel L2, Rel L1, R2.
    Handles both 3D node-wise predictions (requires batch_index) and global scalar predictions.
    """
    # 1. Basic Element-wise Errors
    diff = pred - targets
    mse = torch.mean(diff ** 2)
    mae = torch.mean(torch.abs(diff))
    rmse = torch.sqrt(mse)
    
    # 2. Max AE (Maximum Absolute Error)
    max_ae = torch.max(torch.abs(diff))

    # 3. Relative Metrics (Rel L2, Rel L1)
    
    # Case A: 3D Node-wise predictions (batch_index is required to split samples)
    if batch_index is not None and len(pred.shape) > 1:
        # L2 Norm per sample
        # Numerator: || y_hat - y ||_2
        err_sq_sum = global_add_pool(diff ** 2, batch_index)
        err_norm = torch.sqrt(err_sq_sum)
        
        # Denominator: || y ||_2
        target_sq_sum = global_add_pool(targets ** 2, batch_index)
        target_norm = torch.sqrt(target_sq_sum)
        
        rel_l2 = torch.mean(err_norm / (target_norm + 1e-8))

        # L1 Norm per sample
        # Numerator: || y_hat - y ||_1
        err_abs_sum = global_add_pool(torch.abs(diff), batch_index)
        
        # Denominator: || y ||_1
        target_abs_sum = global_add_pool(torch.abs(targets), batch_index)
        
        rel_l1 = torch.mean(err_abs_sum / (target_abs_sum + 1e-8))

    # Case B: Scalar/Global predictions (Shape: [Batch, Output_Dim])
    else:
        # Calculate norms along dim=1 (per sample)
        err_norm = torch.norm(diff, p=2, dim=1)
        target_norm = torch.norm(targets, p=2, dim=1)
        rel_l2 = torch.mean(err_norm / (target_norm + 1e-8))

        err_norm_l1 = torch.norm(diff, p=1, dim=1)
        target_norm_l1 = torch.norm(targets, p=1, dim=1)
        rel_l1 = torch.mean(err_norm_l1 / (target_norm_l1 + 1e-8))

    # 4. R2 Score (computed on-device to avoid a GPU->CPU sync per step)
    try:
        r2 = torch_r2(targets, pred)
    except:
        r2 = 0.0 # Handle edge cases (e.g. batch size 1 or constant target)

    return {
        "MSE": mse, "MAE": mae, "RMSE": rmse, 
        "Max_AE": max_ae, "Rel_L2": rel_l2, "Rel_L1": rel_l1, "R2": r2
    }