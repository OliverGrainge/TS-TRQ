from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ResNetForImageClassification, ResNetConfig, ResNetModel
from typing import Tuple, Any, List
from quant import get_all_conv2d_names, quantize_model
from typing import Optional

def replace_classifier(model: ResNetForImageClassification, num_classes: int = 100): 
    """
    Replaces the classifier layer of the model in-place to have the specified number of output classes.

    This function modifies the model's classifier module directly (in-place).
    """
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    # This is an in-place operation: model is modified directly.
    return model

def load_resnet(model_name: str = "resnet18", num_classes: int = 100, image_size: int = 224): 
    if model_name == "resnet18":
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
    elif model_name == "resnet50":
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    elif model_name == "resnet50_untrained":
        model = ResNetForImageClassification(ResNetConfig(num_labels=num_classes))
    else: 
        raise ValueError(f"Model {model_name} not supported")

    replace_classifier(model, num_classes=num_classes)
    return model


class ResNetModule(LightningModule):
    def __init__(self, model_name: str = "resnet18", num_classes: int = 100, learning_rate: float = 1e-3, image_size: int = 224):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.is_quantized = False

        self.model = load_resnet(model_name=model_name, num_classes=num_classes, image_size=image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.model(x).logits 

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

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
            self.model = quantize_model(
                self.model, layer_names=layer_names, quant_type=quant_type, **quant_kwargs
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
        all_convs = get_all_conv2d_names(self.model)
        convs = [l for l in all_convs if "embedder.embedder" not in l]
        return convs

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
        for m in self.transformer.modules():
            fn = getattr(m, "layer_reg_loss", None)
            if callable(fn):
                losses.append(fn())
                
        if not losses:
            return torch.tensor(0.0, device=self.device)
            
        losses = torch.stack([torch.as_tensor(l, device=self.device) for l in losses])
        return losses.mean() if reduction == "mean" else losses.sum()


    def ternary_vs_lr_ratio(self) -> dict:
        """
        Compute the ratio of ternary vs low-rank contributions across all TSVD layers.
        
        Returns:
            Dictionary with global statistics and raw values for histogram logging:
            {
                'mean': float,
                'median': float, 
                'min': float,
                'max': float,
                'values': List[float]  # Raw ratio values for histogram logging
            }
        """
        if not self.is_quantized or "tsvd" not in self.quant_type:
            return {
                'mean': 1.0, 'median': 1.0, 'min': 1.0, 'max': 1.0,
                'values': [1.0]
            }
        
        ratios = []
        
        # Compute ratio for each TSVD layer
        for module in self.model.modules():
            if self._is_tsvd_layer(module):
                ratio = self._compute_layer_ratio(module)
                if ratio is not None:
                    ratios.append(ratio.item())
        
        if not ratios:
            return {
                'mean': 1.0, 'median': 1.0, 'min': 1.0, 'max': 1.0,
                'values': [1.0]
            }
        
        # Convert to tensor for easy statistics
        ratios_tensor = torch.tensor(ratios, device=self.device)
        
        return {
            'mean': ratios_tensor.mean().item(),
            'median': ratios_tensor.median().item(),
            'min': ratios_tensor.min().item(),
            'max': ratios_tensor.max().item(),
            'values': ratios  # Raw values for histogram logging
        }

    def _is_tsvd_layer(self, module: torch.nn.Module) -> bool:
        """Check if module is a TSVD layer with required attributes."""
        required_attrs = ["alpha", "L", "R", "lr_scalars", "weight"]
        return all(hasattr(module, attr) for attr in required_attrs)

    def _compute_layer_ratio(self, module: torch.nn.Module) -> torch.Tensor:
        """Compute ternary vs low-rank ratio for a single layer."""
        with torch.no_grad():
            # Compute ternary part: |alpha * quantized_weights|
            if hasattr(module, "ternary_quantize"):
                q_weights, alpha, _, _ = module.ternary_quantize(module.weight, module.thresh_ratio)
            else:
                q_weights = torch.sign(module.weight)
                alpha = module.alpha
            
            ternary_part = (alpha * q_weights).abs().mean()
            
            # Compute low-rank part: |lr_scalars * L @ R|
            L = module.L.to(device=self.device, dtype=module.weight.dtype)
            R = module.R.to(device=self.device, dtype=module.weight.dtype)
            lr_scalars = module.lr_scalars.to(device=self.device, dtype=module.weight.dtype)
            
            if L.numel() > 0 and R.numel() > 0:
                lowrank_part = (lr_scalars * L @ R).abs().mean()
            else:
                lowrank_part = torch.tensor(1e-8, device=self.device)
            
            # Return ratio with numerical stability
            epsilon = 1e-8
            return ternary_part / (lowrank_part + epsilon)

    # Example usage for histogram logging:
    def log_stats(self, logger=None, step=None):
        """
        Helper method to log ternary vs low-rank ratios with histogram.

        Args:
            logger: Your logging framework (e.g., WandbLogger from pytorch-lightning)
            step: Training step for logging
        """
        ratios = self.ternary_vs_lr_ratio()

        # If using WandbLogger from pytorch-lightning, use its log methods
        if logger is not None:
            log_dict = {
                "ternary_vs_lr/mean": ratios['mean'],
                "ternary_vs_lr/median": ratios['median'],
                "ternary_vs_lr/min": ratios['min'],
                "ternary_vs_lr/max": ratios['max'],
            }

            # WandbLogger expects log_dict and step as arguments
            if hasattr(logger, "log_metrics"):
                logger.log_metrics(log_dict, step=step)
            elif hasattr(logger, "log"):
                # fallback for other loggers
                logger.log(log_dict, step=step)

        return ratios


if __name__ == "__main__":
    model = ResNetModule(model_name="resnet18", num_classes=100, image_size=224)
    model.apply_quantization("tsvdconv2d", rank=8)
    print(model.ternary_vs_lr_ratio())
    

    


