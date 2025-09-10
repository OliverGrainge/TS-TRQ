from dotenv import load_dotenv

load_dotenv()

import os, numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_dataset, DownloadConfig
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, image_size=256, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.class_names = None  # optional; may remain None for WDS
        default_transform = [
            transforms.Lambda(lambda im: center_crop_arr(im, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], inplace=True),
        ]
        self.transform = transform or transforms.Compose(default_transform)

    def setup(self, stage=None):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN", None)
        cache_dir = os.getenv("HF_DATASETS_CACHE", None)

        # NOTE: This is the WDS-backed ImageNet-1k you downloaded
        common_kwargs = dict(
            cache_dir=cache_dir,
            download_config=DownloadConfig(delete_extracted=True),
            token=hf_token,
            streaming=False,  # you downloaded to local cache; set True if you want iterable streaming
        )
        self.train_dataset = load_dataset("timm/imagenet-1k-wds", split="train", **common_kwargs)
        self.val_dataset   = load_dataset("timm/imagenet-1k-wds", split="validation", **common_kwargs)

        # (Optional) try to populate class_names if metadata provides it
        # Many WDS shards include 'json' with a 'synset' or 'label' string; fall back to None.
        try:
            probe = self.train_dataset[0]
            if isinstance(probe.get("json", None), dict) and "synset" in probe["json"]:
                # Build a simple index->synset list by scanning once (cheap vs dataset size)
                # If not desired, you can remove this and just skip captions downstream.
                index_to_name = {}
                for item in self.train_dataset.select(range(min(5000, len(self.train_dataset)))):
                    idx = int(item["cls"])
                    name = item["json"].get("synset", str(idx))
                    index_to_name.setdefault(idx, name)
                max_idx = max(index_to_name) if index_to_name else -1
                self.class_names = [index_to_name.get(i, str(i)) for i in range(max_idx+1)]
        except Exception:
            self.class_names = None  # safe default

    def _collate_fn(self, batch):
        images, labels, captions = [], [], []
        for item in batch:
            # WDS-backed HF dataset exposes image under 'jpg'
            img = item["jpg"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            img = self.transform(img.convert("RGB"))
            images.append(img)

            label_idx = int(item["cls"])
            labels.append(label_idx)

            if self.class_names and 0 <= label_idx < len(self.class_names):
                captions.append(self.class_names[label_idx])
            else:
                # try to use JSON synset/label if present; else just the index
                meta = item.get("json", None)
                name = meta.get("synset", None) if isinstance(meta, dict) else None
                captions.append(name if name is not None else str(label_idx))

        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        return {"pixel_values": images, "labels": labels, "captions": captions}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate_fn,
        )

    def get_class_names(self):
        if self.class_names is None:
            raise ValueError("Class names not available for this dataset variant.")
        return self.class_names

# --- Script to use the module ---
if __name__ == "__main__":
    # Example usage of ImageNetDataModule
    # You can adjust batch_size, num_workers, image_size as needed
    datamodule = ImageNetDataModule(batch_size=8, num_workers=2, image_size=128)
    datamodule.setup()

    print("Number of training samples:", len(datamodule.train_dataset))
    print("Number of validation samples:", len(datamodule.val_dataset))

    # Fetch a batch from the training dataloader
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        break
    print("Batch keys:", batch.keys())
    print("Pixel values shape:", batch["pixel_values"].shape)
    print("Pixel values type:", type(batch["pixel_values"]))
    print("Labels shape:", batch["labels"].shape)
    print("Captions example:", batch["labels"][:5])

    # Optionally, print class names if available
    try:
        class_names = datamodule.get_class_names()
        print("Number of classes:", len(class_names))
        print("First 5 class names:", class_names[:5])
    except Exception as e:
        print("Class names not available:", e)