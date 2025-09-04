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
        self.class_names = None  # Will be populated in setup()

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

        # Get class names from the dataset features
        label_feature = self.train_dataset.features["label"]
        self.class_names = label_feature.names

    def _collate_fn(self, batch):
        images = []
        labels = []
        class_names = []

        for item in batch:
            # The image is a PIL Image or a numpy array depending on the dataset
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            img = self.transform(img.convert("RGB"))
            images.append(img)

            label_idx = item["label"]
            labels.append(label_idx)
            class_names.append(self.class_names[label_idx])

        images = torch.stack(images)
        labels = torch.tensor(labels)

        return {
            "pixel_values": images,
            "labels": labels,
            "captions": class_names,  # List of class name strings
        }

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
        """Return list of all class names"""
        if self.class_names is None:
            raise ValueError("DataModule not set up yet. Call setup() first.")
        return self.class_names


# Example usage
if __name__ == "__main__":
    # Create the data module
    datamodule = ImageNetDataModule(batch_size=16, num_workers=2, image_size=256)

    # Setup the data module (this loads the datasets)
    datamodule.setup()

    # Get training and validation dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Example: Iterate through a few batches
    print("Training DataLoader Example:")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  Images shape: {batch['pixel_values'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Sample captions: {batch['captions'][:3]}")  # First 3 class names
        print(
            f"  Sample labels: {batch['labels'][:3].tolist()}"
        )  # First 3 label indices
        if i >= 2:  # Only show first 3 batches
            break

    print("\nValidation DataLoader Example:")
    for i, batch in enumerate(val_loader):
        print(f"Batch {i+1}:")
        print(f"  Images shape: {batch['pixel_values'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Sample captions: {batch['captions'][:3]}")
        if i >= 1:  # Only show first 2 batches
            break

    # Get class names
    class_names = datamodule.get_class_names()
    print(f"\nTotal number of classes: {len(class_names)}")
    print(f"First 10 class names: {class_names[:10]}")

    # Example with custom transform
    print("\n" + "=" * 50)
    print("Example with custom transform:")

    custom_transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 224)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    custom_datamodule = ImageNetDataModule(
        batch_size=8, num_workers=1, image_size=224, transform=custom_transform
    )

    custom_datamodule.setup()
    custom_loader = custom_datamodule.train_dataloader()

    # Test custom dataloader
    batch = next(iter(custom_loader))
    print(f"Custom transform batch shape: {batch['pixel_values'].shape}")
    print(f"Custom transform sample captions: {batch['captions'][:3]}")
