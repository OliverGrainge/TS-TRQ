import os
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
import pytorch_lightning as pl
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from pytorch_lightning.loggers import WandbLogger
from diffusers import (
    UNet2DModel,
    DDPMScheduler,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)
from data import ImageNetDataModule
from quant import quantize_model, get_all_conv2d_names, get_all_linear_names


class QBaseModule(pl.LightningModule, ABC):
    """
    Base class for quantization-aware PyTorch Lightning modules.
    
    This class provides common quantization functionality and requires
    subclasses to implement model initialization and core training logic.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        # Quantization state
        self.is_quantized: bool = False
        self.quant_type: Optional[str] = None
        self.quant_kwargs: Dict[str, Any] = {}
        
        # Model components - to be initialized by subclasses
        self.model: Optional[torch.nn.Module] = None
        
    
    @abstractmethod
    def setup_model(self) -> None:
        """
        Initialize the main model components (unet, etc.).
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass implementation.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step implementation.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def configure_optimizers(self) -> Any:
        """
        Optimizer configuration.
        Must be implemented by subclasses.
        """
        pass
    
    def on_fit_start(self) -> None:
        """Initialize model if not already done."""
        if self.model is None:
            self.setup_model()
    
    def apply_quantization(self, quant_type: str, **quant_kwargs: Any) -> None:
        """
        Apply quantization to layers (and update internal state).
        Keeps original mapping and ignores if already quantized.
        """
        if not quant_type:
            print("No quantization type specified or quantization disabled. Skipping quantization.")
            return

        if self.is_quantized:
            print(f"Model is already quantized with {self.quant_type}. Skipping.")
            return
        
        if self.model is None:
            raise RuntimeError("Model must be initialized before applying quantization. Call setup_model() first.")

        print(f"Applying {quant_type} quantization to model with kwargs: {quant_kwargs}")

        # Conv2d layers
        conv_layers = self._get_quant_conv2d_layer_names()
        if conv_layers:
            self.model = quantize_model(
                self.model,
                layer_names=conv_layers,
                quant_type=self._update_quant_type(quant_type, "conv2d"),
                **quant_kwargs,
            )

        # Linear layers
        linear_layers = self._get_quant_linear_layer_names()
        if linear_layers:
            self.model = quantize_model(
                self.model,
                layer_names=linear_layers,
                quant_type=self._update_quant_type(quant_type, "linear"),
                **quant_kwargs,
            )

        self.is_quantized = True
        self.quant_type = quant_type
        self.quant_kwargs = quant_kwargs
        
        print(f"Quantization applied successfully. Conv2d layers: {len(conv_layers)}, Linear layers: {len(linear_layers)}")

    @staticmethod
    def _update_quant_type(quant_type: str, layer_type: str) -> str:
        """Update quantization type based on layer type."""
        if "tsvd" in quant_type and layer_type == "conv2d":
            return "tsvdconv2d"
        if "tsvd" in quant_type and layer_type == "linear":
            return "tsvdlinear"
        if "t" in quant_type and layer_type == "conv2d":
            return "tconv2d"
        if "t" in quant_type and layer_type == "linear":
            return "tlinear"
        return quant_type

    def _get_quant_conv2d_layer_names(self) -> List[str]:
        """
        Get names of Conv2d layers to quantize.
        Override in subclasses to customize layer selection.
        """
        if self.model is None:
            return []
        
        # Default: quantize all conv2d layers except those in ignore list
        ignore = self._get_ignore_conv2d_patterns()
        all_convs = get_all_conv2d_names(self.model)
        return [name for name in all_convs if not any(pattern in name for pattern in ignore)]
    
    def _get_quant_linear_layer_names(self) -> List[str]:
        """
        Get names of Linear layers to quantize.
        Override in subclasses to customize layer selection.
        """
        if self.model is None:
            return []
        
        # Default: quantize all linear layers except those in ignore list
        ignore = self._get_ignore_linear_patterns()
        all_linear = get_all_linear_names(self.model)
        return [name for name in all_linear if not any(pattern in name for pattern in ignore)]
    
    def _get_ignore_conv2d_patterns(self) -> List[str]:
        """
        Get patterns for Conv2d layer names to ignore during quantization.
        Override in subclasses to customize.
        """
        return []
    
    def _get_ignore_linear_patterns(self) -> List[str]:
        """
        Get patterns for Linear layer names to ignore during quantization.
        Override in subclasses to customize.
        """
        return []

    # ------------- Regularization -------------

    def reg_loss(self, reduction: str = "mean") -> torch.Tensor:
        """
        Aggregate per-layer regularization losses (if any) emitted by quantized modules.
        
        Args:
            reduction: Either "mean" or "sum" for loss aggregation
            
        Returns:
            Aggregated regularization loss tensor
        """
        if not self.is_quantized or self.model is None:
            return torch.tensor(0.0, device=self.device)

        losses: List[torch.Tensor] = []
        for m in self.model.modules():
            fn = getattr(m, "layer_reg_loss", None)
            if callable(fn):
                try:
                    loss = fn()
                    if loss is not None:
                        losses.append(loss)
                except Exception as e:
                    print(f"Warning: Failed to compute reg loss for module {type(m)}: {e}")

        if not losses:
            return torch.tensor(0.0, device=self.device)

        losses_t = torch.stack([torch.as_tensor(l, device=self.device) for l in losses])
        return losses_t.mean() if reduction == "mean" else losses_t.sum()

    # ------------- Diagnostics -------------

    def ternary_vs_lr_ratio(self) -> Dict[str, Union[float, List[float]]]:
        """
        Compute ternary vs. low-rank contribution ratios across TSVD layers.
        Returns aggregate stats and raw values for histogram logging.
        """
        default_stats = {"mean": 1.0, "median": 1.0, "min": 1.0, "max": 1.0, "values": [1.0]}
        
        if not self.is_quantized or not getattr(self, "quant_type", None) or "tsvd" not in self.quant_type:
            return default_stats
        
        if self.model is None:
            return default_stats

        ratios: List[float] = []
        for module in self.model.modules():
            if self._is_tsvd_layer(module):
                try:
                    ratio = self._compute_layer_ratio(module)
                    if ratio is not None and torch.isfinite(ratio):
                        ratios.append(float(ratio))
                except Exception as e:
                    print(f"Warning: Failed to compute ratio for module {type(module)}: {e}")

        if not ratios:
            return default_stats

        t = torch.tensor(ratios, device=self.device)
        return {
            "mean": float(t.mean().item()),
            "median": float(t.median().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "values": ratios,
        }

    @staticmethod
    def _is_tsvd_layer(module: torch.nn.Module) -> bool:
        """
        Check if module is a TSVD quantized layer.
        Heuristic: TSVD layer if it exposes expected attributes.
        """
        required = ["alpha", "L", "R", "lr_scalars", "weight"]
        return all(hasattr(module, attr) for attr in required)

    def _compute_layer_ratio(self, module: torch.nn.Module) -> Optional[torch.Tensor]:
        """
        Compute |alpha * ternary(W)| / |lr_scalars * (L @ R)| for a TSVD layer.
        
        Args:
            module: TSVD quantized module
            
        Returns:
            Ratio tensor or None if computation fails
        """
        try:
            with torch.no_grad():
                if hasattr(module, "ternary_quantize"):
                    q_w, alpha, _, _ = module.ternary_quantize(module.weight, module.thresh_ratio)
                else:
                    q_w = torch.sign(module.weight)
                    alpha = module.alpha

                ternary_part = (alpha * q_w).abs().mean()

                L = module.L.to(device=self.device, dtype=module.weight.dtype)
                R = module.R.to(device=self.device, dtype=module.weight.dtype)
                lr_scalars = module.lr_scalars.to(device=self.device, dtype=module.weight.dtype)

                if L.numel() > 0 and R.numel() > 0:
                    lowrank_part = (lr_scalars * (L @ R)).abs().mean()
                else:
                    lowrank_part = torch.tensor(1e-8, device=self.device)

                eps = 1e-8
                return ternary_part / (lowrank_part + eps)
        except Exception as e:
            print(f"Error computing layer ratio: {e}")
            return None

    def log_stats(self) -> Dict[str, Union[float, List[float]]]:
        """
        Log ternary vs. low-rank ratios to Lightning logger.
        
        Returns:
            Dictionary of computed statistics
        """
        ratios = self.ternary_vs_lr_ratio()
        for k, v in ratios.items():
            if "value" not in k:
                self.log(f"ternary_vs_LR_ratio/{k}", v, prog_bar=False, logger=True)
        return ratios
    
    def validation_step(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        """
        Default validation step - can be overridden by subclasses.
        Logs quantization statistics during validation.
        """
        if self.is_quantized and batch_idx == 0:  # Only log once per validation epoch
            self.log_stats()
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model state.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            "is_quantized": self.is_quantized,
            "quant_type": self.quant_type,
            "quant_kwargs": self.quant_kwargs,
            "model_initialized": self.model is not None,
        }
        
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "parameter_ratio": trainable_params / max(total_params, 1),
            })
        
        return info