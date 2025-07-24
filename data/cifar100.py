import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_workers=4,
        image_size=256,  # CIFAR-100 images are 32x32
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

        # Default transform
        default_transform = [
            transforms.Resize((image_size, image_size)),  # no-op for CIFAR but kept for generality
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]

        self.transform = transform or transforms.Compose(default_transform)

    def setup(self, stage=None):
        # The following will download the data if not already present in self.data_dir
        self.train_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=self.download,
            transform=self.transform,
        )
        self.val_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=self.download,
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )