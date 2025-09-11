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

    @torch.no_grad()
    def global_ternary_vs_lr_ratio(self) -> float:
        """
        Compute global ternary vs. low-rank contribution ratio across all TSVD layers.
        Memory-optimized version with streaming computation.
        """
        if not self.is_quantized or not getattr(self, "quant_type", None) or "tsvd" not in self.quant_type:
            return 1.0
        
        if self.model is None:
            return 1.0

        total_ternary = 0.0
        total_lowrank = 0.0
        layer_count = 0
        
        # Process one layer at a time to minimize memory usage
        for module in self.model.modules():
            if self._is_tsvd_layer(module):
                try:
                    # Compute ratio directly without storing intermediate values
                    ratio_contribution = self._compute_layer_ratio_contribution(module)
                    if ratio_contribution is not None:
                        ternary_contrib, lowrank_contrib = ratio_contribution
                        total_ternary += ternary_contrib
                        total_lowrank += lowrank_contrib
                        layer_count += 1
                        
                        # Clear any cached computations
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                            
                except Exception as e:
                    print(f"Warning: Failed to compute contributions for module {type(module)}: {e}")

        if layer_count == 0 or total_lowrank == 0:
            return 1.0

        return total_ternary / total_lowrank

    @torch.no_grad()
    def _compute_layer_ratio_contribution(self, module: torch.nn.Module) -> Optional[tuple[float, float]]:
        """
        Memory-efficient computation of layer contributions using sampling and norms.
        """
        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype
        
        # Ternary contribution via norm
        alpha_norm = float(module.alpha.abs().norm().item())
        weight_norm = float(module.weight.norm().item())
        ternary_contrib = alpha_norm * weight_norm

        # Low-rank contribution via norm
        L_norm = float(module.L.norm().item())
        R_norm = float(module.R.norm().item()) 
        lr_scalar_norm = float(module.lr_scalars.norm().item())
        lowrank_contrib = lr_scalar_norm * L_norm * R_norm / (module.L.shape[0] ** 0.5)
        return ternary_contrib, max(lowrank_contrib, 1e-8)

    @staticmethod
    def _is_tsvd_layer(module: torch.nn.Module) -> bool:
        """
        Check if module is a TSVD quantized layer.
        Heuristic: TSVD layer if it exposes expected attributes.
        """
        required = ["alpha", "L", "R", "lr_scalars", "weight"]
        return all(hasattr(module, attr) for attr in required)

    @torch.no_grad()
    def log_stats(self) -> float:
        """
        Log global ternary vs. low-rank ratio to Lightning logger.
        
        Returns:
            The computed global ratio
        """
        ratio = self.global_ternary_vs_lr_ratio()
        self.log("ternary_vs_LR_ratio/global", ratio, prog_bar=False, logger=True)
        return ratio

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