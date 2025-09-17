import os
from dotenv import load_dotenv
load_dotenv()
import pytorch_lightning as pl
import torch
from datasets import DownloadConfig, load_dataset
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class HuggingFaceFFHQDataset(Dataset):
    """Wrapper to make HuggingFace FFHQ dataset compatible with PyTorch DataLoader"""

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


class FFHQDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_workers=4,
        cache_dir=None,  # HuggingFace cache directory
        download=True,
        train_split=0.9,  # FFHQ doesn't have predefined splits
        val_split=0.05,   # Remaining 0.05 goes to test
    ):
        load_dotenv()
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir or os.getenv("HF_DATASETS_CACHE", None)
        self.download = download
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1.0 - train_split - val_split

        # Get HuggingFace token if available
        self.hf_token = os.getenv("HF_TOKEN")
        self.cache_dir = cache_dir or os.getenv("HF_DATASETS_CACHE", None)

        # Train transforms with data augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(p=0.5),  # Common for face datasets
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Scale to [-1, 1]
        ])

        # Val/test transforms without augmentation
        self.eval_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Scale to [-1, 1]
        ])

    def setup(self, stage=None):
        """Setup datasets for training, validation, and testing"""

        cache_dir = self.cache_dir or os.getenv("HF_DATASETS_CACHE", None)
        
        # Load the FFHQ dataset
        ds = load_dataset(
            "bitmind/ffhq-256",
            cache_dir=cache_dir,
            download_config=DownloadConfig(delete_extracted=True),
            trust_remote_code=True,
        )

        # FFHQ typically comes as a single split, so we need to split it manually
        full_dataset = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        total_size = len(full_dataset)
        
        # Calculate split sizes
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size

        print(f"Total dataset size: {total_size}")
        print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

        # Split the dataset
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_size))

        # Create subset datasets
        train_subset = full_dataset.select(train_indices)
        val_subset = full_dataset.select(val_indices)
        test_subset = full_dataset.select(test_indices)

        # Wrap HuggingFace datasets with PyTorch Dataset wrapper
        self.train_dataset = HuggingFaceFFHQDataset(
            train_subset, transform=self.train_transform
        )
        self.val_dataset = HuggingFaceFFHQDataset(
            val_subset, transform=self.eval_transform
        )
        self.test_dataset = HuggingFaceFFHQDataset(
            test_subset, transform=self.eval_transform
        )

        print(f"Training dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
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
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if self.num_workers > 0 else False,
        )

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
    dm = FFHQDataModule(batch_size=32, num_workers=4)

    # Setup the datasets
    dm.setup()

    # Get a sample batch from each loader
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    train_batch = next(iter(train_loader))
    print(f"Train batch keys: {train_batch.keys()}")
    print(f"Train pixel values shape: {train_batch['pixel_values'].shape}")
    print(f"Train pixel values range: [{train_batch['pixel_values'].min():.3f}, {train_batch['pixel_values'].max():.3f}]")

    val_batch = next(iter(val_loader))
    print(f"Val batch keys: {val_batch.keys()}")
    print(f"Val pixel values shape: {val_batch['pixel_values'].shape}")
    print(f"Val pixel values range: [{val_batch['pixel_values'].min():.3f}, {val_batch['pixel_values'].max():.3f}]")