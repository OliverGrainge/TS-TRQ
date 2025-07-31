import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, image_size=256, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # Default transform
        default_transform = [
            transforms.Lambda(
                lambda pil_image: center_crop_arr(pil_image, self.image_size)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose(default_transform)

    def setup(self, stage=None):
        self.train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train")
        self.val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")

    def _collate_fn(self, batch):
        images = []
        labels = []
        for item in batch:
            # The image is a PIL Image or a numpy array depending on the dataset
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            img = self.transform(img.convert("RGB"))
            images.append(img)
            labels.append(item["label"])
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels

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
