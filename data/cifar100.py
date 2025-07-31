import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_workers=4,
        image_size=32,  # CIFAR-100 images are 32x32
        transform=None,
        data_dir="./data/raw",
        download=True,  # Add a download flag for clarity
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.data_dir = data_dir
        self.download = download

        # The transforms are NOT in the correct order.
        # Resize should come BEFORE ToTensor, because ToTensor expects a PIL image, but Resize only works on PIL images (not tensors).
        # The correct order is: augmentations (on PIL), Resize (on PIL), ToTensor, Normalize (on tensor).
        train_transform = [
            transforms.RandomCrop(32, padding=4),  # Random crop with padding
            transforms.RandomHorizontalFlip(p=0.5),  # Standard flip probability
            transforms.RandomRotation(15),  # Small random rotation
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.Resize((224, 224)),  # Resize for ViT
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)
            ),  # Regularization
        ]

        val_transform = [
            transforms.Resize((224, 224)),  # <-- Move Resize before ToTensor
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]

        self.train_transform = transforms.Compose(train_transform)
        self.val_transform = transforms.Compose(val_transform)

    def setup(self, stage=None):
        self.train_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=self.download,
            transform=self.train_transform,  # Use train-specific transform
        )
        self.val_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=self.download,
            transform=self.val_transform,  # Use val-specific transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )
