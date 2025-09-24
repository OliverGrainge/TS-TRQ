import os
from dotenv import load_dotenv
load_dotenv()
import pytorch_lightning as pl
import torch
from datasets import DownloadConfig, load_dataset
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class HuggingFaceLSUNBedroomDataset(Dataset):
    """Wrapper to make HuggingFace LSUN Bedroom dataset compatible with PyTorch DataLoader"""

    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]  # This is already a PIL Image


        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image}


class LSUNBedroomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_workers=4,
        cache_dir=None,  # HuggingFace cache directory
        download=True,
    ):
        load_dotenv()
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir or os.getenv("HF_DATASETS_CACHE", None)
        self.download = download

        # Get HuggingFace token if available
        self.hf_token = os.getenv("HF_TOKEN")
        self.cache_dir = cache_dir or os.getenv("HF_DATASETS_CACHE", None)

        # Train transforms with data augmentation
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),  # or RandomCrop for more augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Scale to [-1, 1]
        ])


    def setup(self, stage=None):
        """Setup datasets for training and validation"""

        cache_dir = self.cache_dir or os.getenv("HF_DATASETS_CACHE", None)
        ds = load_dataset(
            "pcuenq/lsun-bedrooms",
            cache_dir=cache_dir,
            download_config=DownloadConfig(delete_extracted=True),
            trust_remote_code=True,
        )

        # Wrap HuggingFace datasets with PyTorch Dataset wrapper
        self.train_dataset = HuggingFaceLSUNBedroomDataset(
            ds["train"], transform=self.transform
        )
        self.test_dataset = HuggingFaceLSUNBedroomDataset(
            ds["test"], transform=self.transform
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
    dm = LSUNBedroomDataModule(batch_size=32, num_workers=4)

    # Setup the datasets
    dm.setup()

    # Get a sample batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Pixel values shape: {batch['pixel_values'].shape}, min: {batch['pixel_values'].min()}, max: {batch['pixel_values'].max()}")
