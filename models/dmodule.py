from typing import Optional, Union, List, Tuple
import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from models.archs import load_diffusion_model
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from quant import get_all_conv2d_names, get_all_linear_names, quantize_model



class EMAModel:
    """Exponential Moving Average for model parameters with proper device handling."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        
        # Store shadow parameters on the same device as model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
    
    def update(self, model: nn.Module):
        """Update EMA parameters, ensuring device consistency."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Move shadow to same device as param if needed
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                
                # Update EMA
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Store original params
                self.original[name] = param.data.clone()
                
                # Move shadow to same device if needed
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                
                # Apply shadow
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module):
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original[name])
        self.original = {}
    
    def to(self, device):
        """Move all EMA shadow parameters to device."""
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)
        return self




class DiffusionBase(pl.LightningModule): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_quantized = False
        self.quant_type = None
        self.quant_kwargs = {}
        self.model = None
        

    def apply_quantization(self, quant_type: str, **quant_kwargs: Any) -> None:
        """
        Apply quantization to layers (and update internal state).
        Keeps original mapping and ignores if already quantized.
        """
        if not quant_type:
            print(
                "No quantization type specified or quantization disabled. Skipping quantization."
            )
            return

        if self.is_quantized:
            print(f"Model is already quantized with {self.quant_type}. Skipping.")
            return

        if self.model is None:
            raise RuntimeError(
                "Model must be initialized before applying quantization. Call setup_model() first."
            )

        print(
            f"Applying {quant_type} quantization to model with kwargs: {quant_kwargs}"
        )

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

        print(
            f"Quantization applied successfully. Conv2d layers: {len(conv_layers)}, Linear layers: {len(linear_layers)}"
        )

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
        return [
            name for name in all_convs if not any(pattern in name for pattern in ignore)
        ]

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
        return [
            name
            for name in all_linear
            if not any(pattern in name for pattern in ignore)
        ]

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
                    print(
                        f"Warning: Failed to compute reg loss for module {type(m)}: {e}"
                    )

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
        if (
            not self.is_quantized
            or not getattr(self, "quant_type", None)
            or "tsvd" not in self.quant_type
        ):
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
                        if hasattr(torch.cuda, "empty_cache"):
                            torch.cuda.empty_cache()

                except Exception as e:
                    print(
                        f"Warning: Failed to compute contributions for module {type(module)}: {e}"
                    )

        if layer_count == 0 or total_lowrank == 0:
            return 1.0

        return total_ternary / total_lowrank

    @torch.no_grad()
    def _compute_layer_ratio_contribution(
        self, module: torch.nn.Module
    ) -> Optional[tuple[float, float]]:
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
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            info.update(
                {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "parameter_ratio": trainable_params / max(total_params, 1),
                }
            )

        return info


class DiffusionModule(DiffusionBase):
    """
    PyTorch Lightning module for training diffusion models.
    Supports both pixel-space and latent-space diffusion at multiple resolutions.
    
    Resolutions supported: 32, 64, 128, 256
    """
    
    def __init__(
        self,
        # Model configuration
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        model_size: str = "tiny",
        model_id: Optional[str] = None,
        class_conditional: bool = False,
        num_classes: Optional[int] = None,
        
        # Image configuration
        sample_size: int = 32,
        pixel_space: bool = None,  # Auto-determined if None
        
        # Training configuration
        ema_decay: float = 0.9999,
        use_ema: bool = True,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
    ):
        """
        Args:
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            model_size: Size of UNet ("tiny", "small", "base", "large")
            model_id: Pretrained model ID (if any)
            class_conditional: Whether to use class conditioning
            num_classes: Number of classes for conditioning
            sample_size: Size of training images (32, 64, 128, 256)
            pixel_space: Use pixel-space (True) or latent-space (False)
                        If None, auto-determined: pixel for ≤64px, latent for ≥128px
            ema_decay: EMA decay rate for model parameters
            use_ema: Whether to use EMA
            warmup_steps: Number of linear warmup steps
            max_steps: Maximum training steps for cosine schedule
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model_size = model_size
        self.model_id = model_id
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.sample_size = sample_size
        self.use_ema = use_ema
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Load models
        self._load_models()
        
        # Initialize EMA
        if self.use_ema:
            self.ema = EMAModel(self.model, decay=ema_decay)
        else:
            self.ema = None
    
    def _load_models(self) -> None:
        """Load and initialize model components."""
        
        model_dict = load_diffusion_model(
            model_size=self.model_size,
            model_id=self.model_id,
            class_conditional=self.class_conditional,
            num_classes=self.num_classes,
            pixel_space=self.hparams.pixel_space,
            sample_size=self.sample_size,
        )
        
        self.model = model_dict["model"]
        self.vae = model_dict["vae"]
        self.train_noise_scheduler = model_dict["train_noise_scheduler"]
        self.inference_noise_scheduler = model_dict["inference_noise_scheduler"]
        self.pixel_space = model_dict["pixel_space"]
        
        # Freeze VAE if using latent space
        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.vae.eval()
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode to latents (if needed), add noise, and predict it.
        
        Args:
            pixel_values: Images in range [-1, 1], shape [B, 3, H, W]
            labels: Class labels (optional)
            
        Returns:
            Tuple of (model_prediction, target)
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # Encode to latent space if using VAE
        if self.vae is not None:
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
        else:
            # Already in pixel space
            latents = pixel_values
        
        # Generate random noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.train_noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device
        ).long()
        
        # Add noise to clean latents/pixels
        noisy_latents = self.train_noise_scheduler.add_noise(
            latents, noise, timesteps
        )
        
        # Predict noise (or v-prediction depending on config)
        if self.class_conditional and labels is not None:
            model_pred = self.model(
                noisy_latents, 
                timesteps, 
                class_labels=labels
            ).sample
        else:
            model_pred = self.model(noisy_latents, timesteps).sample
        
        # Determine target based on prediction type
        if self.train_noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.train_noise_scheduler.config.prediction_type == "v_prediction":
            target = self.train_noise_scheduler.get_velocity(
                latents, noise, timesteps
            )
        else:
            raise ValueError(
                f"Unknown prediction type: {self.train_noise_scheduler.config.prediction_type}"
            )
        
        return model_pred, target
    
    def training_step(self, batch, batch_idx):
        """Training step with proper loss calculation."""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"] if "labels" in batch else None
        
        # Normalize to [-1, 1] if needed
        if pixel_values.min() >= 0.0 and pixel_values.max() <= 1.0:
            pixel_values = pixel_values * 2.0 - 1.0
        
        # Forward pass
        model_pred, target = self(pixel_values, labels)
        
        # Calculate loss (simple MSE for diffusion)
        loss = F.mse_loss(model_pred, target, reduction="mean")
        
        # Log metrics
        self.log("train/train-loss", loss, prog_bar=True)
        
        # Update EMA
        if self.use_ema and self.global_step % 10 == 0:
            self.ema.update(self.model)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Unpack batch
        pixel_values = batch["pixel_values"]
        labels = batch["labels"] if "labels" in batch else None


        # Normalize to [-1, 1] if needed
        if pixel_values.min() >= 0.0 and pixel_values.max() <= 1.0:
            pixel_values = pixel_values * 2.0 - 1.0
        
        # Forward pass
        model_pred, target = self(pixel_values, labels)
        
        # Calculate loss
        loss = F.mse_loss(model_pred, target, reduction="mean")
        
        # Log metrics
        self.log("train/val-loss", loss, prog_bar=True)
        
        return loss

    def on_validation_epoch_end(self):
        """Generate and log sample images at the end of validation."""
        # Only generate samples periodically to save time
    
        
        # Generate samples
        if self.class_conditional:
            # Generate 2 samples per class (up to 10 classes for visualization)
            num_classes_to_show = min(10, self.num_classes)
            samples_per_class = 2
            batch_size = num_classes_to_show * samples_per_class
            
            # Create class labels [0,0,1,1,2,2,...]
            class_labels = []
            for c in range(num_classes_to_show):
                class_labels.extend([c] * samples_per_class)
        else:
            # Unconditional: generate 16-20 samples
            batch_size = 16
            class_labels = None
        
        try:
            # Generate images (using fewer inference steps for speed during training)
            images = self.generate(
                batch_size=batch_size,
                num_inference_steps=50,  # Faster sampling during training
                class_labels=class_labels,
                use_ema=True,
                pbar=True,
                return_pil=False,  # Get tensors for easier logging
            )
            
            # Create a grid
            grid = torchvision.utils.make_grid(
                images, 
                nrow=num_classes_to_show if self.class_conditional else 4,
                normalize=False,  # Already normalized to [0,1]
                padding=2,
                pad_value=1.0,  # White padding
            )
            
            # Log to tensorboard/wandb
            if self.logger is not None:
                # Try different logger types
                logger_name = self.logger.__class__.__name__
                
                if "TensorBoard" in logger_name:
                    # TensorBoard logger
                    self.logger.experiment.add_image(
                        "generated_samples",
                        grid,
                        global_step=self.global_step
                    )
                
                elif "WandbLogger" in logger_name:
                    # WandB logger
                    import wandb
                    # Convert to PIL for wandb
                    grid_pil = torchvision.transforms.ToPILImage()(grid)
                    self.logger.experiment.log({
                        "generated_samples": wandb.Image(grid_pil),
                        "global_step": self.global_step
                    })
                
                else:
                    # Generic logger - try to log as image
                    try:
                        self.logger.log_image(
                            key="generated_samples",
                            images=[grid],
                            step=self.global_step
                        )
                    except:
                        # Fallback: save to file
                        save_path = f"samples_step_{self.global_step}.png"
                        torchvision.utils.save_image(grid, save_path)
                        print(f"Saved samples to {save_path}")
            
            # Also log some metrics about the generated images
            self.log("generated/mean_pixel_value", images.mean(), on_epoch=True)
            self.log("generated/std_pixel_value", images.std(), on_epoch=True)
            
        except Exception as e:
            print(f"Failed to generate samples: {e}")
            # Don't crash training if generation fails

    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer with weight decay
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        
        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_steps,
            eta_min=self.learning_rate * 0.01  # Minimum LR is 1% of initial
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Custom optimizer step with warmup."""
        # Linear warmup
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.learning_rate
        
        # Update parameters
        optimizer.step(closure=optimizer_closure)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log learning rate after each batch."""
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/lr', current_lr, prog_bar=True, on_step=True, on_epoch=False)
    
    @torch.no_grad()
    def generate(
        self,
        height: int = None,
        width: int = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        return_pil: bool = True,
        batch_size: int = 1,
        pbar: bool = True,
        class_labels: Optional[Union[int, List[int]]] = None,
        use_ema: bool = True,
    ) -> Union[List, torch.Tensor]:
        """
        Generate images using the trained diffusion model.
        
        Args:
            height: Image height (defaults to training size)
            width: Image width (defaults to training size)
            num_inference_steps: Number of denoising steps
            generator: Random generator
            return_pil: Return PIL images or tensors
            batch_size: Number of images to generate
            pbar: Show progress bar
            class_labels: Class labels for conditional generation
            use_ema: Use EMA model for generation
            
        Returns:
            Generated images
        """
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        
        # Default to training size if not specified
        if height is None:
            height = self.sample_size
        if width is None:
            width = self.sample_size
        
        # Use EMA model if available
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)
        
        try:
            # Set up scheduler
            scheduler = self.inference_noise_scheduler or self.train_noise_scheduler
            scheduler.set_timesteps(num_inference_steps, device=device)
            
            # Determine latent size for VAE models
            if self.vae is not None:
                latent_height = height // 8
                latent_width = width // 8
                num_channels = 4
            else:
                latent_height = height
                latent_width = width
                num_channels = 3
            
            # Initialize random noise
            shape = (batch_size, num_channels, latent_height, latent_width)
            latents = torch.randn(shape, generator=generator, device=device)
            latents = latents * scheduler.init_noise_sigma
            
            # Prepare class labels
            if self.class_conditional:
                if class_labels is None:
                    class_labels = [0] * batch_size
                elif isinstance(class_labels, int):
                    class_labels = [class_labels] * batch_size
                labels = torch.tensor(class_labels, device=device, dtype=torch.long)
            else:
                labels = None
            
            # Denoising loop
            for t in tqdm(scheduler.timesteps, disable=not pbar, desc="Generating"):
                # Scale input
                latent_input = scheduler.scale_model_input(latents, t)
                
                # Predict noise
                if self.class_conditional and labels is not None:
                    noise_pred = self.model(
                        latent_input, 
                        t, 
                        class_labels=labels
                    ).sample
                else:
                    noise_pred = self.model(latent_input, t).sample
                
                # Compute previous sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # Decode latents to pixels if using VAE
            if self.vae is not None:
                latents = latents / self.vae.config.scaling_factor
                images = self.vae.decode(latents).sample
            else:
                images = latents
            
            # Denormalize from [-1, 1] to [0, 1]
            images = (images + 1.0) / 2.0
            images = images.clamp(0, 1)
            
            if return_pil:
                # Convert to PIL
                images_np = (
                    images.cpu()
                    .permute(0, 2, 3, 1)
                    .numpy() * 255
                ).round().astype("uint8")
                
                from PIL import Image
                return [Image.fromarray(img) for img in images_np]
            
            return images
            
        finally:
            # Restore original model parameters
            if use_ema and self.ema is not None:
                self.ema.restore(self.model)
            
            if was_training:
                self.train()

    
    def _get_ignore_conv2d_patterns(self) -> List[str]:
        """Get Conv2d layer patterns to ignore during quantization."""
        return [
            "conv_in",  # Input convolution
            "conv_out",  # Output convolution
            "conv_shortcut",  # Skip connections
            "c",
        ]

    def _get_ignore_linear_patterns(self) -> List[str]:
        """Get Linear layer patterns to ignore during quantization."""
        return [
            "time_embedding",  # Time embeddings
            "time_emb_proj",  # Time projections
            "linear_1",  # Time embedding layers
            "linear_2",
            "class_embedding",  # Class embeddings
        ]