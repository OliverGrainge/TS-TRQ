import os
from typing import Optional, List, Union, Tuple, Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, DiTPipeline, DPMSolverMultistepScheduler
from PIL import Image
from models.thirdparty.diffusion import create_diffusion

from quant import quantize_model


# Constants
DEFAULT_TIMESTEPS = 1000
DEFAULT_BETA_START = 1e-4
DEFAULT_BETA_END = 2e-2
VAE_SCALE_FACTOR = 0.18215
DEFAULT_GUIDANCE_SCALE = 4.0
DEFAULT_SEED = 45
EPSILON_STABILIZER = 1e-8

# Model loading constants
DEFAULT_PRETRAINED_REPO = "facebook/DiT-XL-2-256"
DEFAULT_IMAGE_SIZE = 256


def _get_device() -> str:
    """Get the appropriate device for model loading."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model_loading_kwargs(
    dtype: torch.dtype, 
    local_path: Optional[str] = None, 
    force_safe: bool = False
) -> Dict[str, Any]:
    """Get common kwargs for model loading."""
    kwargs = {
        "torch_dtype": dtype,
        "cache_dir": os.getenv("HF_HUB_CACHE"),
        "local_files_only": bool(local_path),
    }
    
    if force_safe:
        kwargs["allow_pickle"] = False
    else:
        kwargs["use_safetensors"] = False
        
    return kwargs


def _get_untrained_model(
    model_name: str, 
    image_size: int = DEFAULT_IMAGE_SIZE, 
    dtype: torch.dtype = torch.float16
) -> torch.nn.Module:
    """
    Load an untrained DiT model.
    
    Args:
        model_name: Model name like "dit-xl-2", "dit-l-4", etc.
        image_size: Image size (currently supports 256)
        dtype: torch dtype for the model weights
    
    Returns:
        Untrained DiT model
    """
    from .thirdparty.models import DiT_models
    
    # Normalize model name and map to the correct key format
    model_name_lower = model_name.lower().replace("-", "").replace("_", "")
    
    # Handle the "dit-xl-2-256" format by removing the image size suffix
    if model_name_lower.endswith("256"):
        model_name_lower = model_name_lower[:-3]
    
    # Map common variations to the correct format
    model_mapping = {
        "ditxl2": "DiT-XL/2",
        "ditxl4": "DiT-XL/4", 
        "ditxl8": "DiT-XL/8",
        "ditl2": "DiT-L/2",
        "ditl4": "DiT-L/4",
        "ditl8": "DiT-L/8",
        "ditb2": "DiT-B/2",
        "ditb4": "DiT-B/4", 
        "ditb8": "DiT-B/8",
        "dits2": "DiT-S/2",
        "dits4": "DiT-S/4",
        "dits8": "DiT-S/8"
    }
    
    if model_name_lower not in model_mapping:
        raise ValueError(
            f"Model {model_name} not found. Available models: {list(model_mapping.keys())}"
        )
    
    model_key = model_mapping[model_name_lower]
    
    if model_key not in DiT_models:
        raise ValueError(f"Model {model_key} not found in DiT_models")
    
    # Get the model constructor and create model
    model_constructor = DiT_models[model_key]
    patch_size = int(model_key.split('/')[-1])
    
    # VAE downsamples by 8, so latents are image_size/8
    latent_size = image_size // 8
    model = model_constructor(input_size=latent_size)
    return model.to(dtype=dtype)


def _get_pretrained_model(
    model_name: str, 
    image_size: int = DEFAULT_IMAGE_SIZE, 
    dtype: torch.dtype = torch.float16, 
    device: Optional[str] = None
) -> DiTPipeline:
    """
    Load a pretrained DiT model pipeline.
    
    Args:
        model_name: Model name
        image_size: Image size
        dtype: Data type for model weights
        device: Target device
        
    Returns:
        Loaded DiT pipeline
    """
    if model_name.lower() != "dit-xl" and image_size != DEFAULT_IMAGE_SIZE:
        raise ValueError(
            f"Model {model_name} with image size {image_size} not found. "
            f"Only DiT-XL with size {DEFAULT_IMAGE_SIZE} is supported for pretrained models."
        )

    kwargs = _get_model_loading_kwargs(dtype)
    pipe = DiTPipeline.from_pretrained(DEFAULT_PRETRAINED_REPO, **kwargs)
    
    # Replace scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move to device
    device = device or _get_device()
    return pipe.to(device)


def _get_model(
    model_name: str,
    dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,
    local_path: Optional[str] = None,
    force_safe: bool = False,
    pretrained: bool = True,
    image_size: int = DEFAULT_IMAGE_SIZE
) -> DiTPipeline:
    """
    Load a DiT model pipeline.

    Args:
        model_name: Model name like "dit-xl-2-256", "dit-xl-2", "dit-l-4", etc.
        dtype: torch dtype to load weights in.
        device: e.g. "cuda", "cpu", or "cuda:0". Auto if None.
        local_path: path to an already-downloaded HF snapshot (optional).
        force_safe: if True, require safetensors (allow_pickle=False).
        pretrained: if True, load pretrained weights. If False, use untrained transformer.
        image_size: Image size for the model.
        
    Returns:
        Loaded DiT pipeline
    """
    try:
        if pretrained:
            pipe = _load_pretrained_pipeline(
                model_name, image_size, dtype, local_path, force_safe
            )
        else:
            pipe = _load_untrained_pipeline(
                model_name, image_size, dtype, force_safe
            )
        
        # Replace scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Move to device
        device = device or _get_device()
        return pipe.to(device)
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}") from e


def _load_pretrained_pipeline(
    model_name: str,
    image_size: int,
    dtype: torch.dtype,
    local_path: Optional[str],
    force_safe: bool
) -> DiTPipeline:
    """Load a pretrained DiT pipeline."""
    if model_name.lower() != "dit-xl" and image_size != DEFAULT_IMAGE_SIZE:
        raise ValueError(
            f"Pretrained model {model_name} not found. "
            f"Only 'dit-xl-2-256' is supported for pretrained=True"
        )

    repo_or_path = local_path or DEFAULT_PRETRAINED_REPO
    kwargs = _get_model_loading_kwargs(dtype, local_path, force_safe)
    
    return DiTPipeline.from_pretrained(repo_or_path, **kwargs)


def _load_untrained_pipeline(
    model_name: str,
    image_size: int,
    dtype: torch.dtype,
    force_safe: bool
) -> DiTPipeline:
    """Load an untrained DiT pipeline with pretrained VAE and scheduler."""
    # Load pretrained pipeline for VAE and other components
    kwargs = _get_model_loading_kwargs(dtype, force_safe=force_safe)
    pipe = DiTPipeline.from_pretrained(DEFAULT_PRETRAINED_REPO, **kwargs)
    
    # Replace transformer with untrained model
    untrained_transformer = _get_untrained_model(model_name, image_size, dtype)
    pipe.transformer = untrained_transformer
    
    return pipe


class DiTModule(pl.LightningModule):
    """
    DiT (Diffusion Transformer) PyTorch Lightning module with quantization support.
    
    A Lightning module wrapping a diffusion transformer for image generation,
    with support for various quantization methods and training/inference.
    """
    
    def __init__(
        self, 
        model_name: str = "ditxl2", 
        pretrained: bool = True, 
        image_size: int = 256, 
        learning_rate: float = 1e-4
    ) -> None:
        """
        Initialize the DiT module.
        
        Args:
            model_name: Name of the model to load
            pretrained: Whether to use pretrained weights
            image_size: Size of input images
            learning_rate: Learning rate for optimization
        """
        super().__init__()

        self.diffusion = create_diffusion(timestep_respacing="")
        self.save_hyperparameters()

        model = _get_model(model_name, pretrained=pretrained, image_size=image_size)
        self.vae = model.vae
        self.transformer = model.transformer
        self.pipeline = model
        self.pretrained = pretrained
        self.image_size = image_size

        self.scheduler = DDPMScheduler(
            num_train_timesteps=DEFAULT_TIMESTEPS,
            beta_start=DEFAULT_BETA_START,
            beta_end=DEFAULT_BETA_END,
            beta_schedule="linear",
            prediction_type="epsilon",
            clip_sample=False,
        )

        self.learning_rate = learning_rate
        self.is_quantized = False

    def apply_quantization(self, quant_type: str, **quant_kwargs: Any) -> None:
        """
        Apply quantization to the model after construction/loading.

        Args:
            quant_type: Type of quantization to apply
            **quant_kwargs: Additional arguments for quantization
        """
        if not quant_type or not quant_kwargs:
            print(
                "No quantization type specified or quantization disabled. Skipping quantization."
            )
            return

        if self.is_quantized:
            print(f"Model is already quantized with {self.quant_type}. Skipping.")
            return

        print(f"Applying {quant_type} quantization to model...")

        try:
            # Get linear layer names for quantization
            layer_names = self._get_quant_layer_names()
            self.transformer = quantize_model(
                self.transformer,
                layer_names=layer_names,
                quant_type=quant_type,
                **quant_kwargs,
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
        """Get layer names suitable for quantization."""
        quant_layer_keywords = ["fc1", "fc2", "qkv", "proj", "to_q", "to_k", "to_v", "to_out", "ff.net"]
        return [
            n
            for n, _ in self.transformer.named_modules()
            if any(keyword in n for keyword in quant_layer_keywords)
        ]

    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor
            t: Timestep tensor
            y: Optional class labels
            
        Returns:
            Tuple of (epsilon, logvar) tensors
        """
        if self.pretrained:
            out = self.transformer(x, t, class_labels=y).sample
        else: 
            out = self.transformer(x, t, y)
        epsilon, logvar = out.chunk(2, dim=1)
        return epsilon, logvar

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        class_labels: Optional[List[int]] = None,
        num_steps: int = 50,
        device: Optional[torch.device] = None,
        output_type: str = "tensor",
    ) -> Union[torch.Tensor, List[Image.Image]]:
        """
        Generate samples from the model.
        
        Args:
            batch_size: Number of samples to generate
            class_labels: Optional class labels for conditional generation
            num_steps: Number of inference steps
            device: Device to run inference on
            output_type: Output format ("tensor" or "pil")
            
        Returns:
            Generated images as tensors or PIL Images
        """
        generator = torch.manual_seed(DEFAULT_SEED)
        
        if class_labels is None:
            class_labels = self._generate_random_labels(batch_size, generator)
        
        if device is None:
            device = next(self.parameters()).device

        out = self.pipeline(
            class_labels=class_labels,
            num_inference_steps=num_steps,
            generator=generator,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
        )
        
        return self._format_output(out.images, output_type)

    def _generate_random_labels(
        self, 
        batch_size: int, 
        generator: torch.Generator
    ) -> List[int]:
        """Generate random class labels for unconditional sampling."""
        return torch.randint(
            0, 1000, (batch_size,), generator=generator
        ).tolist()

    def _format_output(
        self, 
        imgs: Union[List[Image.Image], torch.Tensor], 
        output_type: str
    ) -> Union[torch.Tensor, List[Image.Image]]:
        """Format the output images according to the requested type."""
        if output_type == "tensor":
            if isinstance(imgs, list):
                return torch.stack(
                    [torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in imgs]
                )
            return imgs
        elif output_type == "pil":
            if isinstance(imgs, list):
                return imgs
            return [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in imgs]
        else:
            raise ValueError(f"Unsupported output_type: {output_type}. Use 'tensor' or 'pil'.")

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
        for m in self.transformer.modules():
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
        self, 
        magnitudes: List[torch.Tensor], 
        reduction: str
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

    def _compute_ternary_and_lowrank_values(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute ternary and low-rank values for ratio calculation."""
        ternary_vals = []
        lowrank_vals = []
        
        for m in self.transformer.modules():
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
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for the diffusion model.
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        images, labels = batch
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample().mul_(VAE_SCALE_FACTOR)
            
        # Sample random timesteps
        t = torch.randint(
            0, self.diffusion.num_timesteps, (latents.shape[0],), device=images.device
        )
        
        # Prepare model kwargs based on pretrained status
        model_kwargs = self._prepare_model_kwargs(labels)
        
        # Compute losses
        loss_dict = self.diffusion.training_losses(
            self.transformer, latents, t, model_kwargs, pretrained=self.pretrained
        )
        
        current_lr = self.optimizers().param_groups[0]["lr"]
        loss = loss_dict["loss"].mean()
        rloss = self.reg_loss()
        total_loss = loss + rloss

        # Log metrics
        self._log_training_metrics(loss, total_loss, rloss, current_lr)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step for the diffusion model.
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        images, labels = batch
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample().mul_(VAE_SCALE_FACTOR)
            
        # Sample random timesteps
        t = torch.randint(
            0, self.diffusion.num_timesteps, (latents.shape[0],), device=images.device
        )
        
        # Prepare model kwargs based on pretrained status
        model_kwargs = self._prepare_model_kwargs(labels)
        
        # Compute losses
        loss_dict = self.diffusion.training_losses(
            self.transformer, latents, t, model_kwargs, pretrained=self.pretrained
        )
        
        loss = loss_dict["loss"].mean()
        
        # Log validation metrics
        self._log_validation_metrics(loss)
        
        return loss

    def _prepare_model_kwargs(self, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Prepare model kwargs based on pretrained status."""
        if self.pretrained:
            return dict(class_labels=labels)
        else:
            return dict(y=labels)

    def _log_training_metrics(
        self, 
        loss: torch.Tensor, 
        total_loss: torch.Tensor, 
        rloss: torch.Tensor, 
        current_lr: float
    ) -> None:
        """Log training metrics."""
        log_dict = {
            "train_loss": loss,
            "total_loss": total_loss,
            "learning_rate": current_lr,
        }
        
        # Only log reg_loss if model is quantized
        if self.is_quantized:
            log_dict["reg_loss"] = rloss

        self.log_dict(log_dict, on_step=True, prog_bar=False)

    def _log_validation_metrics(self, loss: torch.Tensor) -> None:
        """Log validation metrics."""
        log_dict = {
            "val_loss": loss,
        }
        
        # Log additional validation metrics if model is quantized
        if self.is_quantized:
            log_dict["val_reg_loss"] = self.reg_loss()
            log_dict["val_lr_scalars_magnitude"] = self.lr_scalars_magnitude()
            log_dict["val_ternary_vs_lr_ratio"] = self.ternary_vs_lr_ratio()

        self.log_dict(log_dict, on_step=False, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training."""
        opt = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0,
        )
        return opt


if __name__ == "__main__":
    import os

    os.makedirs("tmp", exist_ok=True)
    dit = DiTModule()
    images = dit.sample(batch_size=2, output_type="pil")
    for idx, image in enumerate(images):
        image.save(f"tmp/image_{idx}.png")
