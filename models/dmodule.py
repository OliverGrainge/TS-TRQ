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
from typing import List, Optional, Tuple, Union, Any
from models.base import DiffusionBase
import wandb 



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
        ema_decay: float = 0.999,
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
        self.ema_decay = ema_decay
        
        # Load models
        self._load_models()
        
        # Initialize EMA
        if self.use_ema:
            self.ema = EMAModel(self.model, decay=ema_decay)
        else:
            self.ema = None

    def apply_quantization(self, quant_type: str, **quant_kwargs: Any) -> None:
        """Apply quantization and reinitialize EMA with new parameters."""
        super().apply_quantization(quant_type, **quant_kwargs)
        
        # Reinitialize EMA after quantization to include new parameters (alpha, lr_scalars, etc.)
        if hasattr(self, 'ema') and self.ema is not None:
            self.ema = EMAModel(self.model, decay=self.ema_decay)


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
        pixel_values = batch["pixel_values"]
        labels = batch["labels"] if "labels" in batch else None
        
        if pixel_values.min() >= 0.0 and pixel_values.max() <= 1.0:
            pixel_values = pixel_values * 2.0 - 1.0
        
        # Encode to latents if using VAE
        if self.vae is not None:
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
        else:
            latents = pixel_values
        
        # Generate noise and timesteps
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.train_noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=latents.device
        ).long()
        
        # Add noise
        noisy_latents = self.train_noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict
        if self.class_conditional and labels is not None:
            model_pred = self.model(noisy_latents, timesteps, class_labels=labels).sample
        else:
            model_pred = self.model(noisy_latents, timesteps).sample
        
        # Determine target
        if self.train_noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.train_noise_scheduler.config.prediction_type == "v_prediction":
            target = self.train_noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = latents
        
        # *** NEW: Compute loss per spatial location ***
        loss = F.mse_loss(model_pred, target, reduction='none')  # [B, C, H, W]
        loss = loss.mean(dim=[1, 2, 3])  # Mean over spatial dims -> [B]
        
        # *** NEW: Apply timestep-dependent weighting ***
        # Calculate LVLB weights similar to reference
        alphas_cumprod = self.train_noise_scheduler.alphas_cumprod.to(device=latents.device)
        alphas = self.train_noise_scheduler.alphas.to(device=latents.device)
        betas = self.train_noise_scheduler.betas.to(device=latents.device)
        
        # Get posterior variance (similar to reference line 149-150)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        
        # Calculate LVLB weights (reference line 161-162)
        lvlb_weights = betas ** 2 / (
            2 * posterior_variance * alphas * (1 - alphas_cumprod)
        )
        lvlb_weights[0] = lvlb_weights[1]  # Fix first timestep
        
        # Apply weights
        loss_simple = loss.mean()
        loss_vlb = (lvlb_weights[timesteps] * loss).mean()
        
        # Combine losses (reference uses original_elbo_weight, typically 0)
        mse_loss = loss_simple + 0.0 * loss_vlb  # Adjust weight if needed
        
        # Add regularization
        reg_loss = self.reg_loss(reduction="mean")
        total_loss = mse_loss + reg_loss
        
        # Log metrics
        self.log("train/mse-loss", mse_loss, prog_bar=True)
        self.log("train/loss_vlb", loss_vlb, prog_bar=False)
        self.log("train/reg-loss", reg_loss, prog_bar=False)
        self.log("train/train-loss", total_loss, prog_bar=True)
        self.log_stats()
        
        # Update EMA every step
        if self.use_ema:
            self.ema.update(self.model)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with EMA comparison."""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"] if "labels" in batch else None
        
        if pixel_values.min() >= 0.0 and pixel_values.max() <= 1.0:
            pixel_values = pixel_values * 2.0 - 1.0
        
        # Evaluate without EMA
        model_pred, target = self(pixel_values, labels)
        loss_no_ema = F.mse_loss(model_pred, target)
        self.log("val/loss_no_ema", loss_no_ema, prog_bar=False, on_epoch=True)
        
        # Evaluate with EMA
        if self.use_ema:
            self.ema.apply_shadow(self.model)
            model_pred_ema, target_ema = self(pixel_values, labels)
            loss_ema = F.mse_loss(model_pred_ema, target_ema)
            self.log("val/loss_ema", loss_ema, prog_bar=True, on_epoch=True)
            self.ema.restore(self.model)
        return loss_no_ema

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
                num_inference_steps=100,  # Faster sampling during training
                class_labels=class_labels,
                use_ema=True,
                pbar=False,
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
                
                grid_pil = torchvision.transforms.ToPILImage()(grid)
                self.logger.experiment.log({
                    "generated_samples": wandb.Image(grid_pil),
                    "global_step": self.global_step
                })
                
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
            weight_decay=0.0,
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
        self.log('train/lr', current_lr, on_step=True, on_epoch=False)
    
    @torch.no_grad()
    def generate(
        self,
        height: int = None,
        width: int = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        return_pil: bool = True,
        batch_size: int = 1,
        pbar: bool = False,
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
        ]

    def _get_ignore_linear_patterns(self) -> List[str]:
        """Get Linear layer patterns to ignore during quantization."""
        return [
            "time_embedding",  # Time embeddings
            "time_emb_proj",  # Time projections
            "linear_1",  # Time embedding layers
            "linear_2",
            "class_embedding",  # Class embeddings

            "to_q",  # Attention query projections
            "to_k",  # Attention key projectionsc
            "to_v",  # Attention value projections
            "to_out",  # Attention output projections
        ]