from dotenv import load_dotenv
load_dotenv()
import os, numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_dataset, DownloadConfig
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import warnings

# Suppress PIL EXIF corruption warnings
warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Train transforms (no augmentation as discussed)
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),  # or RandomCrop for more augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Scale to [-1, 1]
        ])

    def setup(self, stage=None):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN", None)
        cache_dir = os.getenv("HF_DATASETS_CACHE", None)

        common_kwargs = dict(
            cache_dir=cache_dir,
            download_config=DownloadConfig(delete_extracted=True),
            token=hf_token,
            streaming=False,
        )
        self.train_dataset = load_dataset("timm/imagenet-1k-wds", split="train", **common_kwargs)
        self.val_dataset = load_dataset("timm/imagenet-1k-wds", split="validation", **common_kwargs)

    def _collate_fn(self, batch, transform):
        images, labels = [], []
        for item in batch:
            # WDS-backed HF dataset exposes image under 'jpg'
            img = item["jpg"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            img = transform(img.convert("RGB"))
            images.append(img)
            labels.append(int(item["cls"]))

        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        return {"pixel_values": images, "labels": labels}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda batch: self._collate_fn(batch, self.transform),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda batch: self._collate_fn(batch, self.transform),
        )


    # Example usage
if __name__ == "__main__":
    # Initialize the datamodule
    dm = ImageNetDataModule(batch_size=32, num_workers=4)

    # Setup the datasets
    dm.setup()

    # Get a sample batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Pixel values shape: {batch['pixel_values'].shape}, min: {batch['pixel_values'].min()}, max: {batch['pixel_values'].max()}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Labels: {batch['labels']}")