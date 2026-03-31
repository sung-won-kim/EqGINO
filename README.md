# EqGINO

Official code repository for **EqGINO** (Equivariant Geometry-Informed Fourier Neural Operator for 3D PDEs).

This repository contains the implementation of EqGINO, designed for learning 3D physics simulations with equivariance.



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

### Preprocessed Data
The preprocessed versions of the datasets used in this project are available for download here: [Google Drive Link](https://drive.google.com/file/d/1iObwzvIyBIg-FAAdmZK8etY0Ccv7-9EL/view?usp=sharing)

After downloading, please place the `data` folder in the same directory as `main.py` with the following structure:
```
data/
    ahmedbody/
        train/
        test/
    shapenetcar/
        train/
        test/
    deepjeb/
        train/
        test/
```

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
| `configs/deepjeb_deflection.yaml` | DeepJeb | `3d_dj_deflection` |
| `configs/deepjeb_stress.yaml` | DeepJeb | `3d_dj_stress` |

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

**DeepJeb - Deflection**
```bash
python main.py --config configs/deepjeb_deflection.yaml --devices 0
```

**DeepJeb - Stress**
```bash
python main.py --config configs/deepjeb_stress.yaml --devices 0
```

## Acknowledgement

This project utilizes code from the [neuraloperator](https://github.com/neuraloperator/neuraloperator) library. We have modified specific components to implement the equivariant features required for EqGINO. We thank the authors of `neuralop` for their open-source contribution.
