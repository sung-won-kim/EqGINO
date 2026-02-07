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
| `torch` | 2.5.1+cu118 |
| `torch-geometric` | 2.6.1 |
| `neuraloperator` | 1.0.2 |
| `lightning` | 2.5.0.post0 |
| `trimesh` | 4.7.1 |
| `scikit-learn` | 1.6.1 |
| `numpy` | 2.1.2 |

### Installation

You can create a conda environment and install the dependencies:

```bash
conda create -n eqgino python=3.10
conda activate eqgino

# Install PyTorch (adjust cuda version if needed)
pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install torch-geometric neuraloperator lightning wandb trimesh scikit-learn numpy
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
```

## Usage

The main training script is `main.py`.

### Arguments

- `--model`: Model architecture (default: `eqgino`)
- `--data_fname`: Dataset name. Choices: `['ahmedbody', 'shapenetcar']`
- `--tgt_y`: Target variable to predict.
    - **AhmedBody**: `3d_ab_wss`, `3d_ab_p`, `3d_ab_k`, `3d_ab_omega`, `3d_ab_nut`
    - **ShapeNetCar**: `3d_snc_press`
- `--aug_type`: Rotation augmentation type.
    - `canonical`: No rotation. (Canonical)
    - `discrete`: Random discrete rotations.
    - `arbitrary`: Random continuous rotations.
- `--batch_size`: Batch size (default: 1)
- `--hidden_dim`: Hidden dimension size (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--gno_radius`: GNO radius (default: Ahmedbody 0.1, ShapeNetCar 0.15)
- `--fno_n_layers`: Number of FNO layers
- `--fno_n_mode`: Number of FNO modes
- `--mesh_subsample_rate`: Subsampling rate for training mesh (default: 5 -> 1/5 nodes)
- `--mesh_subsample_rate_valid`: Subsampling rate for validation/test mesh (default: 1)

### Training Examples

**1. Train on Ahmed Body for Turbulent Kinetic Energy**
```bash
python main.py --data_fname ahmedbody --tgt_y 3d_ab_k
```

**2. Train on Ahmed Body for Wall Shear Stress (WSS) with Arbitrary Rotation**
```bash
python main.py --data_fname ahmedbody --tgt_y 3d_ab_wss --aug_type arbitrary
```

## Acknowledgement

This project utilizes code from the [neuraloperator](https://github.com/neuraloperator/neuraloperator) library. We have modified specific components to implement the equivariant features required for EqGINO. We thank the authors of `neuralop` for their open-source contribution.