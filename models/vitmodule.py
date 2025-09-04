from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTForImageClassification

from quant import quantize_model

# Constants
CIFAR_IMAGE_SIZE = 32
VIT_DEFAULT_IMAGE_SIZE = 224
EPSILON_STABILIZER = 1e-8
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_EPS = 1e-8

# Layer names for quantization
QUANTIZABLE_LAYER_KEYWORDS = ["query", "key", "value", "dense", "intermediate"]


class ViTModule(pl.LightningModule):
    """
    Vision Transformer (ViT) PyTorch Lightning module with quantization support.

    A Lightning module wrapping a Vision Transformer for image classification,
    with support for various quantization methods and training/inference.
    """

    def __init__(
        self,
        model_name: str = "Ahmed9275/Vit-Cifar100",
        num_classes: int = 100,
        learning_rate: float = 2e-3,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
    ) -> None:
        """
        Initialize the ViT module.

        Args:
            model_name: Name/path of the pretrained model to load
            num_classes: Number of output classes
            learning_rate: Learning rate for optimization
            image_size: Size of input images
            patch_size: Size of image patches
            num_channels: Number of input channels
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model_name = model_name

        # Load pre-trained ViT model or create a new one
        self._load_pretrained_model()

        # Adjust classifier if needed
        self._adjust_classifier_if_needed()

        # Track quantization state
        self.is_quantized = False
        self.quant_type: Optional[str] = None
        self.quant_kwargs: Optional[Dict[str, Any]] = None

    def _load_pretrained_model(self) -> None:
        """Load the pretrained ViT model."""
        print(f"Loading pre-trained model: {self.model_name}")
        try:
            self.vit = ViTForImageClassification.from_pretrained(self.model_name)
            print(
                f"Successfully loaded pre-trained model with {self.vit.config.num_labels} classes"
            )
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            raise

    def _adjust_classifier_if_needed(self) -> None:
        """Adjust the classifier head if the number of classes doesn't match."""
        if self.vit.config.num_labels != self.num_classes:
            print(
                f"Adjusting classifier head from {self.vit.config.num_labels} to {self.num_classes} classes"
            )
            self.vit.classifier = nn.Linear(
                self.vit.config.hidden_size, self.num_classes
            )

    def apply_quantization(self, quant_type: str, **quant_kwargs: Any) -> None:
        """
        Apply quantization to the model after construction/loading.

        Args:
            quant_type: Type of quantization to apply
            **quant_kwargs: Additional keyword arguments for quantization
        """
        if not quant_type:
            print(
                "No quantization type specified or quantization disabled. Skipping quantization."
            )
            return

        if self.is_quantized:
            print(f"Model is already quantized with {self.quant_type}. Skipping.")
            return

        print(
            f"Applying {quant_type} quantization to model with kwargs: {quant_kwargs}"
        )

        try:
            # Get linear layer names for quantization
            layer_names = self._get_quant_layer_names()
            self.vit = quantize_model(
                self.vit, layer_names=layer_names, quant_type=quant_type, **quant_kwargs
            )

            # Update quantization state
            self.is_quantized = True
            self.quant_type = quant_type
            self.quant_kwargs = quant_kwargs

            print(f"Quantization applied successfully.")
        except Exception as e:
            print(f"Failed to apply quantization: {e}")
            raise

    def _get_quant_layer_names(self) -> List[str]:
        """
        Get names of linear layers that can be quantized.

        Returns:
            List of layer names suitable for quantization
        """
        layer_names = []
        for name, module in self.vit.named_modules():
            if isinstance(module, nn.Linear):
                # Include attention and MLP linear layers, but typically exclude the final classifier
                if any(keyword in name for keyword in QUANTIZABLE_LAYER_KEYWORDS):
                    layer_names.append(name)
        return layer_names

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ViT model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # ViT expects images in (batch_size, channels, height, width) format
        # For CIFAR-100, we might need to resize from 32x32 to 224x224
        x = self._resize_if_needed(x)

        outputs = self.vit(x)
        return outputs.logits

    def _resize_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        """Resize images if they don't match the expected ViT input size."""
        if x.shape[-1] == CIFAR_IMAGE_SIZE:  # CIFAR-100 size
            x = F.interpolate(
                x, size=VIT_DEFAULT_IMAGE_SIZE, mode="bilinear", align_corners=False
            )
        return x

    def reg_loss(self, reduction: str = "mean") -> torch.Tensor:
        """
        Compute regularization loss from quantized layers.

        Args:
            reduction: Reduction method ("mean" or "sum")

        Returns:
            Regularization loss tensor
        """
        if not self.is_quantized:
            return torch.tensor(0.0, device=self.device)

        losses = []
        for m in self.vit.modules():
            fn = getattr(m, "layer_reg_loss", None)
            if callable(fn):
                losses.append(fn())

        if not losses:
            return torch.tensor(0.0, device=self.device)

        losses = torch.stack([torch.as_tensor(l, device=self.device) for l in losses])
        return losses.mean() if reduction == "mean" else losses.sum()

    def lr_scalars_magnitude(self, reduction: str = "mean") -> torch.Tensor:
        """
        Compute magnitude statistics of lr_scalars from TSVDLinear layers.

        Args:
            reduction: Reduction method ("mean", "max", "std")

        Returns:
            Magnitude statistics tensor
        """
        if not self.is_quantized or self.quant_type != "tsvdlinear":
            return torch.tensor(0.0, device=self.device)

        magnitudes = []
        for m in self.vit.modules():
            if self._is_tsvd_linear_layer(m):
                mag = m.lr_scalars.abs()
                magnitudes.append(mag.flatten())

        if not magnitudes:
            return torch.tensor(0.0, device=self.device)

        return self._compute_magnitude_reduction(magnitudes, reduction)

    def _is_tsvd_linear_layer(self, module: torch.nn.Module) -> bool:
        """Check if a module is a TSVDLinear layer."""
        return hasattr(module, "lr_scalars") and hasattr(module, "rank")

    def _compute_magnitude_reduction(
        self, magnitudes: List[torch.Tensor], reduction: str
    ) -> torch.Tensor:
        """Compute reduction over magnitude tensors."""
        all_magnitudes = torch.cat(magnitudes)

        if reduction == "mean":
            return all_magnitudes.mean()
        elif reduction == "max":
            return all_magnitudes.max()
        elif reduction == "std":
            return all_magnitudes.std()
        else:
            return all_magnitudes.mean()

    def ternary_vs_lr_ratio(self) -> torch.Tensor:
        """
        Estimate the relative contribution of the ternary (quantized) weights vs. the low-rank correction
        in TSVDLinear layers, by comparing the mean absolute value of the ternary part (alpha * q)
        to the mean absolute value of the low-rank correction (lr_scalars * L @ R).

        Returns:
            Ratio of ternary to low-rank contributions
        """
        if not self.is_quantized or self.quant_type != "tsvdlinear":
            return torch.tensor(1.0, device=self.device)

        ternary_vals, lowrank_vals = self._compute_ternary_and_lowrank_values()

        if not ternary_vals or not lowrank_vals:
            return torch.tensor(1.0, device=self.device)

        mean_ternary = torch.cat(ternary_vals).mean()
        mean_lowrank = torch.cat(lowrank_vals).mean()
        return mean_ternary / (mean_lowrank + EPSILON_STABILIZER)

    def _compute_ternary_and_lowrank_values(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute ternary and low-rank values for ratio calculation."""
        ternary_vals = []
        lowrank_vals = []

        for m in self.vit.modules():
            if self._has_tsvd_components(m):
                ternary_part = self._compute_ternary_part(m)
                if ternary_part is not None:
                    ternary_vals.append(ternary_part.flatten())

                lowrank_part = self._compute_lowrank_part(m)
                if lowrank_part is not None:
                    lowrank_vals.append(lowrank_part.flatten())

        return ternary_vals, lowrank_vals

    def _has_tsvd_components(self, module: torch.nn.Module) -> bool:
        """Check if module has all required TSVD components."""
        required_attrs = ["alpha", "L", "R", "lr_scalars"]
        return all(hasattr(module, attr) for attr in required_attrs)

    def _compute_ternary_part(self, module: torch.nn.Module) -> Optional[torch.Tensor]:
        """Compute the ternary quantization part."""
        with torch.no_grad():
            q_nograd, alpha, _, _ = (
                module.ternary_quantize(module.weight, module.thresh_ratio)
                if hasattr(module, "ternary_quantize")
                else (torch.sign(module.weight), module.alpha, None, None)
            )
            return (alpha * q_nograd).abs()

    def _compute_lowrank_part(self, module: torch.nn.Module) -> Optional[torch.Tensor]:
        """Compute the low-rank correction part."""
        L = self._to_device_and_dtype(module.L)
        R = self._to_device_and_dtype(module.R)
        lr_scalars = self._to_device_and_dtype(module.lr_scalars)

        if L.numel() > 0 and R.numel() > 0:
            lowrank = (lr_scalars * L) @ R
            return lowrank.abs()
        return None

    def _to_device_and_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to appropriate device and dtype."""
        target_dtype = self.dtype if hasattr(self, "dtype") else tensor.dtype
        return tensor.to(device=self.device, dtype=target_dtype)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step for the ViT model.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index

        Returns:
            Total training loss
        """
        images, labels = batch

        # Forward pass
        logits = self(images)

        # Compute classification loss
        loss = F.cross_entropy(logits, labels)

        # Add regularization loss (only if quantized)
        rloss = self.reg_loss()
        total_loss = loss + rloss

        # Compute accuracy
        acc = self._compute_accuracy(logits, labels)

        # Get current learning rate from optimizer
        current_lr = self.optimizers().param_groups[0]["lr"]

        # Log metrics
        self._log_training_metrics(loss, total_loss, rloss, acc, current_lr)

        return total_loss

    def _compute_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute classification accuracy."""
        preds = torch.argmax(logits, dim=1)
        return (preds == labels).float().mean()

    def _log_training_metrics(
        self,
        loss: torch.Tensor,
        total_loss: torch.Tensor,
        rloss: torch.Tensor,
        acc: torch.Tensor,
        current_lr: float,
    ) -> None:
        """Log training metrics."""
        log_dict = {
            "train_loss": loss,
            "total_loss": total_loss,
            "train_acc": acc,
            "learning_rate": current_lr,
        }

        # Only log reg_loss if model is quantized
        if self.is_quantized:
            log_dict["reg_loss"] = rloss

        # Log lr_scalars magnitude for tsvdlinear quantization
        log_dict.update(self._get_tsvd_metrics())

        self.log_dict(log_dict, on_step=True, prog_bar=False)

    def _get_tsvd_metrics(self) -> Dict[str, torch.Tensor]:
        """Get TSVD-related metrics for logging."""
        return {
            "lr_scalars_mean": self.lr_scalars_magnitude(reduction="mean"),
            "lr_scalars_max": self.lr_scalars_magnitude(reduction="max"),
            "lr_scalars_std": self.lr_scalars_magnitude(reduction="std"),
            "ternary_lr_ratio": self.ternary_vs_lr_ratio(),
        }

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step for the ViT model.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index

        Returns:
            Dictionary containing validation metrics
        """
        images, labels = batch

        # Forward pass
        logits = self(images)

        # Compute loss and accuracy
        loss = F.cross_entropy(logits, labels)
        acc = self._compute_accuracy(logits, labels)

        # Log metrics
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

        return {"val_loss": loss, "val_acc": acc}

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Test step for the ViT model.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index

        Returns:
            Dictionary containing test metrics
        """
        images, labels = batch

        # Forward pass
        logits = self(images)

        # Compute loss and accuracy
        loss = F.cross_entropy(logits, labels)
        acc = self._compute_accuracy(logits, labels)

        # Log metrics
        self.log_dict({"test_loss": loss, "test_acc": acc})

        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer for ViT training.

        Returns:
            Configured AdamW optimizer
        """
        optimizer = torch.optim.AdamW(
            self.vit.parameters(),
            lr=self.learning_rate,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            betas=(DEFAULT_BETA1, DEFAULT_BETA2),
            eps=DEFAULT_EPS,
        )

        # Optional: Add learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.trainer.max_epochs if hasattr(self, "trainer") and self.trainer is not None else 100, eta_min=1e-6
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_loss",
        #     }
        # }

        return optimizer


if __name__ == "__main__":
    # Example usage
    print("=== Creating ViT Module ===")
    vit_model = ViTModule(
        model_name="Ahmed9275/Vit-Cifar100", num_classes=100, learning_rate=1e-4
    )

    # Test with dummy CIFAR-100 data
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 32, 32)  # CIFAR-100 size
    dummy_labels = torch.randint(0, 100, (batch_size,))

    print("\n=== Full Precision Model ===")
    # Forward pass (full precision)
    logits = vit_model(dummy_images)
    print(f"Input shape: {dummy_images.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in vit_model.parameters()):,}")
    print(f"Is quantized: {vit_model.is_quantized}")

    # Apply quantization
    print("\n=== Applying Quantization ===")
    vit_model.apply_quantization("tsvdlinear", {"rank": 8})

    # Test quantized model
    print("\n=== Quantized Model ===")
    logits_quant = vit_model(dummy_images)
    print(f"Output shape after quantization: {logits_quant.shape}")
    print(f"Is quantized: {vit_model.is_quantized}")
    print(f"Quantization type: {vit_model.quant_type}")
    print(f"Quantization args: {vit_model.quant_kwargs}")

    # Compare outputs
    diff = (logits - logits_quant).abs().mean()
    print(f"Mean absolute difference: {diff:.6f}")

    print("\n=== Usage Example for Checkpoint Loading ===")
    print("# 1. Train unquantized model and save checkpoint")
    print("# 2. Load checkpoint:")
    print("model = ViTModule.load_from_checkpoint('checkpoint.ckpt')")
    print("# 3. Apply quantization:")
    print("model.apply_quantization('tsvdlinear', {'rank': 8})")
    print("# 4. Continue training or inference with quantized model")

    print("\n=== Alternative: Create from scratch ===")
    print("# Create a new ViT model from scratch:")
    vit_scratch = ViTModule(
        model_name="new_model",  # This will create a new model
        num_classes=100,
        learning_rate=1e-4,
        image_size=224,
        patch_size=16,
    )
