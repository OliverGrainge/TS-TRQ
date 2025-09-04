import os

import pytorch_lightning as pl
import torch
from datasets import DownloadConfig, load_dataset
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

load_dotenv()


class HuggingFaceLSUNChurchesDataset(Dataset):
    """Wrapper to make HuggingFace CIFAR100 dataset compatible with PyTorch DataLoader"""

    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]  # This is already a PIL Image
        label = item["label"]  # CIFAR100 uses 'fine_label' for the 100 classes

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "labels": label}


class LSUNChurchesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_workers=4,
        image_size=224,  # Changed to 224 for ViT compatibility (was 32)
        cache_dir=None,  # HuggingFace cache directory
        download=True,
    ):
        load_dotenv()
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.cache_dir = cache_dir or os.getenv("HF_DATASETS_CACHE", None)
        self.download = download

        # Get HuggingFace token if available
        self.hf_token = os.getenv("HF_TOKEN")
        self.cache_dir = cache_dir or os.getenv("HF_DATASETS_CACHE", None)

        # Train transforms with data augmentation
        train_transform = [
            transforms.RandomCrop(32, padding=4),  # Standard for CIFAR-100
            transforms.RandomHorizontalFlip(),  # Standard for CIFAR-100
            transforms.Resize((self.image_size, self.image_size)),  # For ViT
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]

        # Validation transforms: just resize and normalize
        val_transform = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]

        self.train_transform = transforms.Compose(train_transform)
        self.val_transform = transforms.Compose(val_transform)

    def setup(self, stage=None):
        """Setup datasets for training and validation"""

        # Load training dataset
        ds = load_dataset("tglcourse/lsun_church_train", cache_dir=self.cache_dir)

        # Wrap HuggingFace datasets with PyTorch Dataset wrapper
        self.train_dataset = HuggingFaceLSUNChurchesDataset(
            ds["train"], transform=self.train_transform
        )
        self.test_dataset = HuggingFaceLSUNChurchesDataset(
            ds["test"], transform=self.val_transform
        )

        print(f"Training dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        """Same as test_dataloader for testing"""
        return self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if self.num_workers > 0 else False,
        )


# Example usage
if __name__ == "__main__":
    # Initialize the datamodule
    dm = LSUNChurchesDataModule(batch_size=32, num_workers=4)

    # Setup the datasets
    dm.setup()

    # Get a sample batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Pixel values shape: {batch['pixel_values'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Labels: {batch['labels']}")
