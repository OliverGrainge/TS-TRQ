from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from transformers import (ResNetConfig, ResNetForImageClassification,
                          ResNetModel)

from quant import get_all_conv2d_names, quantize_model
from .base import QBaseModule  # Import the base class


def replace_classifier(model: ResNetForImageClassification, num_classes: int = 100):
    """
    Replaces the classifier layer of the model in-place to have the specified number of output classes.

    This function modifies the model's classifier module directly (in-place).
    """
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    # This is an in-place operation: model is modified directly.
    return model


def load_resnet(
    model_name: str = "resnet18", num_classes: int = 100, image_size: int = 224
):
    """Load ResNet model with support for both pretrained and untrained variants"""

    if model_name == "resnet18":
        return ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
    elif model_name == "resnet18_untrained":
        # Create untrained ResNet18 with proper config
        config = ResNetConfig(
            num_labels=num_classes,
            num_channels=3,
            embedding_size=64,
            hidden_sizes=[64, 128, 256, 512],
            depths=[2, 2, 2, 2],  # ResNet18 layer structure
            layer_type="basic",  # Use basic blocks for ResNet18
            hidden_act="relu",
            downsample_in_first_stage=False,
        )
        return ResNetForImageClassification(config)
    elif model_name == "resnet50":
        return ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    elif model_name == "resnet50_untrained":
        # Create untrained ResNet50 with proper config
        config = ResNetConfig(
            num_labels=num_classes,
            num_channels=3,
            embedding_size=64,
            hidden_sizes=[256, 512, 1024, 2048],
            depths=[3, 4, 6, 3],  # ResNet50 layer structure
            layer_type="bottleneck",  # Use bottleneck blocks for ResNet50
            hidden_act="relu",
            downsample_in_first_stage=False,
        )
        return ResNetForImageClassification(config)
    else:
        raise ValueError(
            f"Model {model_name} not supported. Supported models: resnet18, resnet18_untrained, resnet50, resnet50_untrained"
        )


class ResNetModule(QBaseModule):
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 100,
        learning_rate: float = 1e-3,
        image_size: int = 224,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.image_size = image_size

        # Load the ResNet model and set it as the base class model
        resnet_model = load_resnet(
            model_name=model_name, num_classes=num_classes, image_size=image_size
        )
        self.resnet = resnet_model
        self.model = resnet_model  # Set for base class

    def setup_model(self) -> None:
        """
        Initialize the main model components (ResNet).
        Implementation of abstract method from QBaseModule.
        """
        if self.model is None:
            resnet_model = load_resnet(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                image_size=self.image_size
            )
            self.resnet = resnet_model
            self.model = resnet_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet model."""
        return self.resnet(x).logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step with cross-entropy loss and regularization."""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self(pixel_values)
        ce_loss = F.cross_entropy(logits, labels)
        reg_loss = self.reg_loss()
        train_loss = ce_loss + reg_loss
        
        self.log("ce_loss", ce_loss)
        self.log("reg_loss", reg_loss)
        self.log("train_loss", train_loss)

        if self.global_step % 50 == 0:
            self.log_stats()
        
        return train_loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step with accuracy computation."""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self(pixel_values)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)
        
        # Call parent validation_step for quantization statistics logging
        super().validation_step(batch, batch_idx)
        
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step with accuracy computation."""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self(pixel_values)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("test_loss", loss)
        self.log("test_acc", acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure AdamW optimizer."""
        return torch.optim.AdamW(self.resnet.parameters(), lr=self.learning_rate)

    # ------------- Layer Selection Overrides -------------

    def _get_quant_conv2d_layer_names(self) -> List[str]:
        """
        Get Conv2d layer names for quantization, excluding embedder layers.
        Override of base class method.
        """
        if self.model is None:
            return []
        
        all_convs = get_all_conv2d_names(self.model)
        convs = [l for l in all_convs if "embedder.embedder" not in l]
        return convs

    def _get_ignore_conv2d_patterns(self) -> List[str]:
        """
        Get patterns for Conv2d layer names to ignore during quantization.
        Override of base class method.
        """
        return ["embedder.embedder"]


if __name__ == "__main__":
    model = ResNetModule(model_name="resnet18", num_classes=100, image_size=224)
    model.apply_quantization("tsvdconv2d", rank=8)
    print(model.ternary_vs_lr_ratio())
