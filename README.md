# EqGINO: Equivariant Geometry-Informed Fourier Neural Operator for 3D PDEs

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://iclr.cc" alt="Conference">
        <img src="https://img.shields.io/badge/ICML'26-brightgreen" /></a>
    <a href="https://iclr.cc" alt="Conference">
        <img src="https://img.shields.io/badge/ICLR'26 AI&PDE Workshop-brightgreen" /></a>
<!--     <img src="https://img.shields.io/pypi/l/torch-rechub"> -->
</p>

Official code repository for **EqGINO** (Equivariant Geometry-Informed Fourier Neural Operator for 3D PDEs) at ICML 2026. [[paper]](https://arxiv.org/abs/2606.03260)


## Requirements

The code has been tested with the following environment:

- **OS**: Linux
- **Python**: 3.10.18

### Key Dependencies

| Package | Version |
|---------|---------|
| `torch` | 2.5.1+cu121 |
| `torch-geometric` | 2.6.1 |
| `lightning` | 2.5.2 |
| `trimesh` | 4.7.3 |
| `scikit-learn` | 1.6.1 |
| `numpy` | 2.1.2 |
| `tensorly` | 0.9.0 |
| `open3d` | 0.19.0 |
| `wandb` | 0.21.1 |

> **Note:** `neuralop` is included locally in this repository (modified for equivariance), so you do NOT need to install it separately.

### Installation

You can create a conda environment and install the dependencies:

```bash
conda create -n eqgino python=3.10
conda activate eqgino

# 1. Install PyTorch first (adjust CUDA version if needed)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Install PyG extensions (must match torch + CUDA version)
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric==2.6.1

# 3. Install remaining dependencies
pip install -r requirements.txt
```

## Datasets

### Ahmed Body
Original dataset sourced from [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_ahmed_body_dataset/files?version=v1).

### ShapeNetCar
Original dataset sourced from [Zenodo](https://zenodo.org/records/13936501).

### Download

**AhmedBody & ShapeNetCar:** Download the preprocessed datasets from [Google Drive](https://drive.google.com/file/d/1iObwzvIyBIg-FAAdmZK8etY0Ccv7-9EL/view?usp=sharing) and extract the `data` folder in the same directory as `main.py`.

After downloading, the directory structure should look like:
```
data/
    ahmedbody/
        train/    # 458 cases (case{id}.pkl)
        test/     # 50 cases
    shapenetcar/
        train/    # 500 cases ({id}.pkl)
        test/     # 100 cases
```

### Data Structure

Each `.pkl` file is a PyG `Data` object. Below are the key fields for each dataset.

#### AhmedBody (`case{id}.pkl`)

| Field | Shape | Description |
|-------|-------|-------------|
| `x` | [N, 3] | Mesh point coordinates |
| `edge_index` | [2, E] | Graph edge connectivity |
| `faces` | [F, 3] | Triangle face indices |
| `conds_feat` | [1, 8] | Conditional features (Length, Width, Height, GroundClearance, SlantAngle, FilletRadius, Velocity, Re) |
| `inlet_vel_direction` | [1, 3] | Inlet velocity unit vector |
| `y_p` | [N, 1] | Pressure |
| `y_U` | [N, 3] | Velocity |
| `y_wallShearStress` | [N, 3] | Wall shear stress (default target) |
| `y_k` | [N, 1] | Turbulent kinetic energy |
| `y_omega` | [N, 1] | Specific dissipation rate |
| `y_nut` | [N, 1] | Turbulent viscosity |
| `y_yPlus` | [N, 1] | Y+ value |

#### ShapeNetCar (`{id}.pkl`)

| Field | Shape | Description |
|-------|-------|-------------|
| `x` | [N, 3] | Mesh point coordinates |
| `edge_index` | [2, E] | Graph edge connectivity |
| `faces` | [F, 3] | Triangle face indices |
| `vertex_normals` | [N, 3] | Surface normals at each vertex |
| `conds_feat` | [1, C] | Conditional features |
| `inlet_vel_direction` | [1, 3] | Inlet velocity unit vector |
| `y` | [N, 1] | Surface pressure (target) |

## Usage

The main training script is `main.py`. Each dataset has a YAML config file under `configs/` with pre-tuned hyperparameters.

### Quick Start

```bash
# Train with a config file (recommended)
python main.py --config configs/ahmedbody.yaml --devices 0
```

### Config Files

| Config | Dataset | Target |
|--------|---------|--------|
| `configs/ahmedbody.yaml` | AhmedBody | `3d_ab_wss` |
| `configs/shapenetcar.yaml` | ShapeNetCar | `3d_snc_press` |

Each config contains dataset-specific hyperparameters (model architecture, learning rate, GNO radius, etc.). See `configs/*.yaml` for details.

### Arguments

CLI arguments for controlling the training run:

- `--config`: Path to YAML config file (required)
- `--model`: Model architecture. Choices: `eqgino` (default: `eqgino`)
- `--devices`: GPU device IDs (e.g., `0` or `0,1`)
- `--seed`: Random seed (default: 0)
- `--num_seed`: Number of seeds to run (default: 1)
- `--aug_type`: Rotation augmentation type.
    - `canonical`: No rotation (default)
    - `discrete`: Random 90-degree rotations
    - `arbitrary`: Random continuous rotations
- `--num_workers`: DataLoader workers (default: 0)
- `--val_interval`: Validation every N epochs (default: 1)
- `--log_name`: Custom W&B run name
- `--summary`: W&B project prefix

### Training Examples

**AhmedBody - Wall Shear Stress**
```bash
python main.py --config configs/ahmedbody.yaml --devices 0
```

**AhmedBody - Wall Shear Stress with Arbitrary Rotation**
```bash
python main.py --config configs/ahmedbody.yaml --aug_type arbitrary --devices 0
```

**ShapeNetCar - Surface Pressure**
```bash
python main.py --config configs/shapenetcar.yaml --devices 0
```

## Release Notes

See [`RELEASES.md`](RELEASES.md) for the changelog. The latest release (**v1.1.0**)
makes training **1.64× faster / 2.47× lighter** and inference **2.87× faster /
1.79× lighter** with numerically equivalent outputs (no architecture, weight, or
hyper-parameter changes).

## Acknowledgement

This project utilizes code from the [neuraloperator](https://github.com/neuraloperator/neuraloperator) library. We have modified specific components to implement the equivariant features required for EqGINO. We thank the authors of `neuralop` for their open-source contribution.

