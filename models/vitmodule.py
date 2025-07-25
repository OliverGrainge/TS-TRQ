import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTForImageClassification

from quant import quantize


class ViTModule(pl.LightningModule):
    def __init__(
        self,
        model_name="Ahmed9275/Vit-Cifar100",
        num_classes=100,
        learning_rate=2e-3,
        reg_scale=0.5,
        image_size=224,
        patch_size=16,
        num_channels=3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.reg_scale = reg_scale
        self.model_name = model_name

        # Load pre-trained ViT model or create a new one
        try:
            print(f"Loading pre-trained model: {model_name}")
            self.vit = ViTForImageClassification.from_pretrained(model_name)
            print(
                f"Successfully loaded pre-trained model with {self.vit.config.num_labels} classes"
            )

            # Adjust classifier if needed
            if self.vit.config.num_labels != num_classes:
                print(
                    f"Adjusting classifier head from {self.vit.config.num_labels} to {num_classes} classes"
                )
                self.vit.classifier = nn.Linear(
                    self.vit.config.hidden_size, num_classes
                )

        except Exception as e:
            print(f"Could not load pre-trained model {model_name}: {e}")
            print("Creating new ViT model from scratch...")

            # Create new ViT configuration
            config = ViTConfig(
                image_size=image_size,
                patch_size=patch_size,
                num_channels=num_channels,
                num_labels=num_classes,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
            )

            self.vit = ViTForImageClassification(config)
            print(f"Created new ViT model with {num_classes} classes")

        # Track quantization state
        self.is_quantized = False
        self.quant_type = None
        self.quant_kwargs = None

    def apply_quantization(self, quant_type, quant_kwargs=None):
        """
        Apply quantization to the model after construction/loading.

        Args:
            quant_type: Type of quantization to apply
            quant_kwargs: Arguments for quantization
        """
        if quant_kwargs is None:
            quant_kwargs = {}

        if self.is_quantized:
            print(f"Model is already quantized with {self.quant_type}. Skipping.")
            return

        print(f"Applying {quant_type} quantization to model...")

        # Get linear layer names for quantization
        layer_names = self._get_quant_layer_names()
        self.vit = quantize(
            self.vit, layer_names=layer_names, quant_type=quant_type, **quant_kwargs
        )

        # Update quantization state
        self.is_quantized = True
        self.quant_type = quant_type
        self.quant_kwargs = quant_kwargs

        print(f"Quantization applied successfully.")

    def _get_quant_layer_names(self):
        """Get names of linear layers that can be quantized"""
        layer_names = []
        for name, module in self.vit.named_modules():
            if isinstance(module, nn.Linear):
                # Include attention and MLP linear layers, but typically exclude the final classifier
                if any(
                    key in name
                    for key in ["query", "key", "value", "dense", "intermediate"]
                ):
                    layer_names.append(name)
        return layer_names

    def forward(self, x):
        """Forward pass through the ViT model"""
        # ViT expects images in (batch_size, channels, height, width) format
        # For CIFAR-100, we might need to resize from 32x32 to 224x224
        if x.shape[-1] == 32:  # CIFAR-100 size
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)

        outputs = self.vit(x)
        return outputs.logits

    def reg_loss(self, reduction="mean"):
        """Compute regularization loss from quantized layers"""
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

    def lr_scalars_magnitude(self, reduction="mean"):
        """Compute magnitude statistics of lr_scalars from TSVDLinear layers"""
        if not self.is_quantized or self.quant_type != "tsvdlinear":
            return torch.tensor(0.0, device=self.device)

        magnitudes = []
        for m in self.vit.modules():
            # Check if this is a TSVDLinear layer by looking for lr_scalars attribute
            if hasattr(m, "lr_scalars") and hasattr(m, "rank"):
                # Get the magnitude of lr_scalars
                mag = m.lr_scalars.abs()
                magnitudes.append(mag.flatten())
        
        if not magnitudes:
            return torch.tensor(0.0, device=self.device)
        
        # Concatenate all magnitudes and compute statistics
        all_magnitudes = torch.cat(magnitudes)
        if reduction == "mean":
            return all_magnitudes.mean()
        elif reduction == "max":
            return all_magnitudes.max()
        elif reduction == "std":
            return all_magnitudes.std()
        else:
            return all_magnitudes.mean()

    def ternary_vs_lr_ratio(self):
        """Compute the ratio of ternary alpha magnitudes vs lr_scalars magnitudes"""
        if not self.is_quantized or self.quant_type != "tsvdlinear":
            return torch.tensor(1.0, device=self.device)

        alpha_mags = []
        lr_mags = []
        
        for m in self.vit.modules():
            if hasattr(m, "lr_scalars") and hasattr(m, "alpha") and hasattr(m, "rank"):
                # Get magnitudes
                alpha_mag = m.alpha.abs().flatten()
                lr_mag = m.lr_scalars.abs().flatten()
                
                alpha_mags.append(alpha_mag)
                lr_mags.append(lr_mag)
        
        if not alpha_mags or not lr_mags:
            return torch.tensor(1.0, device=self.device)
        
        # Compute mean magnitudes
        mean_alpha = torch.cat(alpha_mags).mean()
        mean_lr = torch.cat(lr_mags).mean()
        
        # Return ratio (ternary / low-rank)
        # Higher ratio means ternary weights are doing more work
        return mean_alpha / (mean_lr + 1e-8)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        logits = self(images)

        # Compute classification loss
        loss = F.cross_entropy(logits, labels)

        # Add regularization loss (only if quantized)
        rloss = self.reg_scale * self.reg_loss()
        total_loss = loss + rloss

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Get current learning rate from optimizer
        current_lr = self.optimizers().param_groups[0]["lr"]

        # Log metrics
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
            if self.quant_type == "tsvdlinear":
                log_dict["lr_scalars_mean"] = self.lr_scalars_magnitude(reduction="mean")
                log_dict["lr_scalars_max"] = self.lr_scalars_magnitude(reduction="max")
                log_dict["lr_scalars_std"] = self.lr_scalars_magnitude(reduction="std")
                log_dict["ternary_lr_ratio"] = self.ternary_vs_lr_ratio()

        self.log_dict(log_dict, on_step=True, prog_bar=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        logits = self(images)

        # Compute loss and accuracy
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        logits = self(images)

        # Compute loss and accuracy
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log_dict({"test_loss": loss, "test_acc": acc})

        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        """Configure optimizer for ViT training"""
        optimizer = torch.optim.AdamW(
            self.vit.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
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
