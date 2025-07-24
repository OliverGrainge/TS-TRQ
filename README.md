# PyTorch Lightning Trainer for ImageNet Finetuning

This repository contains a comprehensive PyTorch Lightning trainer for finetuning various models on the ImageNet dataset.

## Features

- **Multiple Model Support**: Supports ResNet, VGG, DenseNet, EfficientNet, and DiT models
- **ImageNet Data Loading**: Automatic data loading with proper transforms and augmentation
- **Transfer Learning**: Option to freeze backbone layers for transfer learning
- **Mixed Precision Training**: Support for 16-bit mixed precision training
- **Automatic Checkpointing**: Saves best models based on validation loss
- **Early Stopping**: Prevents overfitting with early stopping
- **Learning Rate Scheduling**: StepLR scheduler with configurable parameters
- **TensorBoard Logging**: Automatic logging of training metrics

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### DiT Training (Updated)

For training DiT models with checkpointing and wandb logging:

```bash
# Install additional dependencies
pip install wandb python-dotenv datasets

# Set up wandb (first time only)
wandb login

# Run training
python train.py
```

The updated trainer includes:
- **Wandb Logging**: Automatic experiment tracking and model logging
- **Checkpointing**: Saves best models based on validation loss
- **Learning Rate Scheduling**: Cosine annealing scheduler
- **Mixed Precision**: 16-bit training for faster training
- **Gradient Accumulation**: Effective larger batch sizes
- **Gradient Clipping**: Prevents gradient explosion

### Command Line Interface

The trainer can be used directly from the command line:

```bash
python trainers/tstrq_trainer.py \
    --model resnet50 \
    --data_dir /path/to/imagenet \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --pretrained \
    --freeze_backbone
```

### Python API

You can also use the trainer programmatically:

```python
from trainers.tstrq_trainer import train_model

# Train a ResNet50 model
model, trainer = train_model(
    model_name="resnet50",
    data_dir="/path/to/imagenet",
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-4,
    pretrained=True,
    freeze_backbone=False
)
```

### Example Script

Use the provided example script:

```bash
python scripts/train_example.py
```

## Supported Models

### Standard Classification Models
- `resnet50` - ResNet-50
- `resnet101` - ResNet-101
- `resnet152` - ResNet-152
- `vgg16` - VGG-16
- `densenet121` - DenseNet-121
- `efficientnet_b0` - EfficientNet-B0

### Diffusion Models
- `dit-xl-2-256` - DiT-XL-2-256 (requires special handling)

## Data Structure

The ImageNet dataset should be organized as follows:

```
imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

## Configuration Options

### Model Configuration
- `model_name`: Name of the model to train
- `num_classes`: Number of output classes (default: 1000 for ImageNet)
- `pretrained`: Whether to use pretrained weights
- `freeze_backbone`: Whether to freeze backbone layers for transfer learning

### Training Configuration
- `learning_rate`: Initial learning rate
- `weight_decay`: Weight decay for optimizer
- `batch_size`: Batch size for training
- `num_epochs`: Number of training epochs
- `scheduler_gamma`: Learning rate decay factor
- `scheduler_step_size`: Epochs between learning rate decay

### Data Configuration
- `data_dir`: Path to ImageNet dataset
- `image_size`: Input image size
- `num_workers`: Number of data loading workers

### Hardware Configuration
- `accelerator`: Training accelerator ("auto", "gpu", "cpu")
- `devices`: Number of devices to use
- `precision`: Training precision ("16-mixed", "32", "bf16-mixed")

## Training Features

### Data Augmentation
The trainer includes comprehensive data augmentation for training:
- Random resized crop
- Random horizontal flip
- Color jittering (brightness, contrast, saturation, hue)

### Validation Transforms
- Resize to 256x256
- Center crop to 224x224
- Normalization with ImageNet statistics

### Callbacks
- **ModelCheckpoint**: Saves the best 3 models based on validation loss
- **LearningRateMonitor**: Logs learning rate changes
- **EarlyStopping**: Stops training if validation loss doesn't improve for 10 epochs

### Logging
- TensorBoard logging with training and validation metrics
- Automatic logging of loss, accuracy, and learning rate
- Progress bars with real-time metrics

## Output

The trainer creates the following outputs:
- `checkpoints/`: Saved model checkpoints
- `lightning_logs/`: TensorBoard logs
- Console output with training progress

## Example Training Runs

### Transfer Learning (Freeze Backbone)
```bash
python trainers/tstrq_trainer.py \
    --model resnet50 \
    --data_dir /path/to/imagenet \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-3 \
    --pretrained \
    --freeze_backbone
```

### Full Finetuning
```bash
python trainers/tstrq_trainer.py \
    --model resnet101 \
    --data_dir /path/to/imagenet \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --pretrained
```

### Multi-GPU Training
```bash
python trainers/tstrq_trainer.py \
    --model resnet50 \
    --data_dir /path/to/imagenet \
    --batch_size 128 \
    --epochs 100 \
    --lr 1e-4 \
    --pretrained \
    --devices 4 \
    --accelerator gpu
```

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir lightning_logs
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Increase num_workers or use mixed precision
3. **Poor Performance**: Try different learning rates or freeze backbone
4. **Data Loading Issues**: Check data directory structure and permissions

### Performance Tips

- Use mixed precision training (`--precision 16-mixed`)
- Increase batch size if memory allows
- Use multiple GPUs for faster training
- Adjust learning rate based on model size

## License

This project is licensed under the MIT License. 