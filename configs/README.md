# Configuration Files

This directory contains YAML configuration files for training and validation scripts. The scripts require a YAML configuration file to run.

## Usage

### Training
```bash
# Use a configuration file
python train.py configs/train_dit_imagenet.yaml

# Or use any custom config
python train.py path/to/your/config.yaml
```

### Validation
```bash
# Use a configuration file
python val.py configs/val_dit_imagenet.yaml

# Or use any custom config
python val.py path/to/your/config.yaml
```

## Configuration Files

### Training Configurations
- `train_dit_imagenet.yaml` - DiT training on ImageNet with default settings
- `train_vit_cifar100.yaml` - ViT training on CIFAR100 with quantization

### Validation Configurations
- `val_dit_imagenet.yaml` - Validation setup for DiT on ImageNet

## Configuration Parameters

### Training Parameters
```yaml
# Model and dataset
module: "dit"                    # Model type: "dit", "mlp", "vit"
dataset: "imagenet"              # Dataset: "imagenet", "cifar100"
checkpoint: null                 # Path to checkpoint or null

# Training hyperparameters
batch_size: 8
max_epochs: 20
accumulate_grad_batches: 4
learning_rate: 1e-6
reg_scale: 0.5

# Quantization
quant_type: "none"               # "tsvdlinear", "trqlinear", "nbitlinear", "none"
quant_rank: 192

# Training setup
precision: 32                    # 16 or 32
accelerator: "gpu"               # "cpu", "gpu"
devices: "1"                     # Number or list of GPU IDs
strategy: "auto"                 # "ddp", "dp", "auto"

# Logging and checkpointing
project: "dit-imagenet"          # Wandb project name
save_dir: "checkpoints"          # Checkpoint directory
enable_progress_bar: false       # Show progress bar
```

### Validation Parameters
```yaml
# Model and dataset
module: "dit"
dataset: "imagenet"
checkpoint: "path/to/model.ckpt" # Required for validation

# Validation settings
batch_size: 16
test: false                      # Use test set instead of validation set

# Model settings (for fresh model without checkpoint)
learning_rate: 1e-6
reg_scale: 0.5

# Quantization
quant_type: "none"
quant_rank: 192

# Hardware setup
precision: 32
accelerator: "gpu"
devices: "1"
```

## Creating Custom Configurations

1. Copy an existing configuration file
2. Modify the parameters as needed
3. Save with a descriptive name
4. Use with `python train.py path/to/your/config.yaml`

## Configuration Loading

Configuration values are loaded in this order:
1. Default values (built into the code) are loaded first
2. YAML configuration file values override defaults

All parameters must be specified in the YAML file. If a parameter is not in the YAML file, the default value will be used. 