import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset, DownloadConfig
from dotenv import load_dotenv
from PIL import Image


class HuggingFaceCIFAR10Dataset(Dataset):
    """Wrapper to make HuggingFace CIFAR10 dataset compatible with PyTorch DataLoader"""
    
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['img']  # This is already a PIL Image
        label = item['label']  # This is the class index
        
        if self.transform:
            image = self.transform(image)
            
        return {"pixel_values": image, "labels": label}


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_workers=4,
        image_size=32,  # Changed to 224 for ViT compatibility
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

        # Train transforms with data augmentation
        train_transform = [
            transforms.RandomCrop(32, padding=4),  # Standard for CIFAR-100
            transforms.RandomHorizontalFlip(),     # Standard for CIFAR-100
            transforms.Resize((self.image_size, self.image_size)),  # For ViT
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ]

        # Validation transforms: just resize and normalize
        val_transform = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ]


        self.train_transform = transforms.Compose(train_transform)
        self.val_transform = transforms.Compose(val_transform)

    def setup(self, stage=None):
        """Setup datasets for training and validation"""
        
        # Load training dataset
        train_hf_dataset = load_dataset(
            "cifar10",
            split="train",
            cache_dir=self.cache_dir,
            download_config=DownloadConfig(delete_extracted=True) if self.download else None,
        )
        
        # Load test dataset (used as validation)
        val_hf_dataset = load_dataset(
            "cifar10",
            split="test",
            cache_dir=self.cache_dir,
            download_config=DownloadConfig(delete_extracted=True) if self.download else None,
        )
        
        # Wrap HuggingFace datasets with PyTorch Dataset wrapper
        self.train_dataset = HuggingFaceCIFAR10Dataset(
            train_hf_dataset, 
            transform=self.train_transform
        )
        self.val_dataset = HuggingFaceCIFAR10Dataset(
            val_hf_dataset, 
            transform=self.val_transform
        )
        
        print(f"Training dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")

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
        """Same as val_dataloader for testing"""
        return self.val_dataloader()


# Example usage
if __name__ == "__main__":
    # Initialize the datamodule
    dm = CIFAR10DataModule(batch_size=32, num_workers=4)
    
    # Setup the datasets
    dm.setup()
    
    # Get a sample batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Pixel values shape: {batch['pixel_values'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Labels: {batch['labels']}")