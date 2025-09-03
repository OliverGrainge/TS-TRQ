import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset, DownloadConfig
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


class HuggingFaceLSUNBedroomDataset(Dataset):
    """Wrapper to make HuggingFace LSUN Bedroom dataset compatible with PyTorch DataLoader"""
    
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']  # This is already a PIL Image
        
        # LSUN bedroom dataset typically doesn't have labels, so we'll use a dummy label
        # or you can modify this based on your specific needs
        label = 0  # Dummy label for bedroom class
        
        if self.transform:
            image = self.transform(image)
            
        return {"pixel_values": image, "labels": label}


class LSUNBedroomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_workers=4,
        image_size=256,  # LSUN bedrooms are typically 256x256
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

        # Train transforms with data augmentation for bedrooms
        train_transform = [
            transforms.RandomHorizontalFlip(),     # Standard augmentation
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Standard normalization
        ]

        # Validation transforms: just resize and normalize
        val_transform = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]

        self.train_transform = transforms.Compose(train_transform)
        self.val_transform = transforms.Compose(val_transform)

    def setup(self, stage=None):
        """Setup datasets for training and validation"""
        
        cache_dir = self.cache_dir or os.getenv("HF_DATASETS_CACHE", None)
        ds = load_dataset(
            "tglcourse/lsun_church_train",
            cache_dir=cache_dir,
            download_config=DownloadConfig(delete_extracted=True),
            trust_remote_code=True,
        )

        
        # Wrap HuggingFace datasets with PyTorch Dataset wrapper
        self.train_dataset = HuggingFaceLSUNBedroomDataset(ds["train"], transform=self.train_transform)
        self.test_dataset = HuggingFaceLSUNBedroomDataset(ds["test"], transform=self.val_transform)
        
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
    print(f"Pixel values shape: {batch['pixel_values'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Labels: {batch['labels']}")
