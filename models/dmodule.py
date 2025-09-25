import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger
from diffusers import (
    UNet2DModel,
    DDPMScheduler,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    DDPMPipeline
)

from data import ImageNetDataModule
from quant import quantize_model, get_all_conv2d_names, get_all_linear_names
from .base import QBaseModule


def _load_unet(
    model_type: str, 
    in_channels: int = 4, 
    class_conditional: bool = False, 
    num_classes: Optional[int] = None
) -> UNet2DModel:
    """
    Load UNet with different size configurations.
    
    Args:
        model_type: Size of the model ("tiny", "small", "base")
        in_channels: Number of input channels
        class_conditional: Whether to use class conditioning
        num_classes: Number of classes for class conditioning
    
    Returns:
        UNet2DModel configured for the specified size
    """
    configs = {
        "tiny": {
            "sample_size": 32,
            "block_out_channels": (128, 128, 256, 256),
            "down_block_types": ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            "up_block_types": ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            "attention_head_dim": 8,
            "layers_per_block": 1,
        },
        "small": {
            "sample_size": 32,
            "block_out_channels": (128, 256, 512, 512),
            "down_block_types": ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            "up_block_types": ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            "attention_head_dim": 8,
            "layers_per_block": 2,
        },
        "base": {
            "sample_size": 32,
            "block_out_channels": (320, 640, 1280, 1280),
            "down_block_types": ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            "up_block_types": ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            "attention_head_dim": 8,
            "layers_per_block": 2,
        },
    }
    
    if model_type not in configs:
        raise ValueError(f"model_type must be one of {list(configs.keys())}, got {model_type}")
    
    config = configs[model_type]
    
    # Build UNet configuration
    unet_kwargs = {
        "sample_size": config["sample_size"],
        "in_channels": in_channels,
        "out_channels": in_channels,
        "down_block_types": config["down_block_types"],
        "up_block_types": config["up_block_types"],
        "block_out_channels": config["block_out_channels"],
        "layers_per_block": config["layers_per_block"],
        "attention_head_dim": config["attention_head_dim"],
        "norm_num_groups": 32,
    }
    
    # Add class conditioning if requested
    if class_conditional:
        if num_classes is None:
            raise ValueError("num_classes must be provided when class_conditional=True")
        unet_kwargs.update({
            "class_embed_type": "timestep",
            "num_class_embeds": num_classes,
        })
    
    return UNet2DModel(**unet_kwargs)



def load_pretrained_diffusion(model_id: str) -> Dict[str, Any]: 
    if model_id in ["google/ddpm-ema-church-256", "google/ddpm-ema-bedroom-256", "google/ddpm-cifar10-32"]:
        model_dict = {} 
        pipeline = DDPMPipeline.from_pretrained(model_id)
        model_dict["unet"] = pipeline.unet
        model_dict["train_noise_scheduler"] = pipeline.scheduler 
        schd = EulerAncestralDiscreteScheduler.from_config(
            model_dict["train_noise_scheduler"].config
        )
        schd.config.use_karras_sigmas = True        # smoother noise schedule -> higher quality/fewer steps
        schd.config.timestep_spacing = "trailing"
        schd.set_timesteps(40)
        model_dict["inference_noise_scheduler"] = schd
        model_dict["vae"] = None
        return model_dict
    else:
        raise ValueError(f"Invalid model ID: {model_id}")

def load_diffusion_model(
    model_id: Union[str, None] = None,
    model_size: str = "base", 
    in_channels: int = 4, 
    class_conditional: bool = False, 
    num_classes: Optional[int] = None,
    pixel_space: bool = False
) -> Dict[str, Any]:
    """
    Load VAE, UNet, and schedulers for Stable Diffusion.

    Args:
        model_size: Size of UNet model ("tiny", "small", "base")
        in_channels: Number of input channels for UNet
        class_conditional: Whether to use class conditioning
        num_classes: Number of classes for conditioning

    Returns:
        Dictionary with model components
    """
    cache_dir = os.getenv("HF_TRANSFORMERS_CACHE")
    if model_id is not None: 
        return load_pretrained_diffusion(model_id)

    # Load VAE
    if not pixel_space:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", cache_dir=cache_dir)
    else: 
        vae = None

    # Load UNet with specified configuration
    unet = _load_unet(
        model_type=model_size,
        in_channels=in_channels,
        class_conditional=class_conditional,
        num_classes=num_classes,
    )

    train_noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="scaled_linear",  # keep as-is if that's how you trained
    beta_start=0.00085,
    beta_end=0.012,
    clip_sample=False,
    prediction_type="epsilon",      # <-- make this explicit; change only if you truly trained with "v_prediction"
)

    # Inference scheduler: start from the train config, then override *inference-only* goodies
    inference_noise_scheduler = EulerAncestralDiscreteScheduler.from_config(
        train_noise_scheduler.config
    )
    # Recommended inference-only tweaks for Euler-A:
    inference_noise_scheduler.config.use_karras_sigmas = True        # smoother noise schedule -> higher quality/fewer steps
    inference_noise_scheduler.config.timestep_spacing = "trailing"   # denser steps at the end; often best for Euler/DPM samplers
    # (Alternative: "leading" or "linspace". "trailing" tends to improve detail/texture with few steps.)

    # When sampling:
    num_inference_steps = 40  # try 20–40 for 256x256 LSUN
    inference_noise_scheduler.set_timesteps(num_inference_steps)

    return {
        "vae": vae,
        "unet": unet,
        "train_noise_scheduler": train_noise_scheduler,
        "inference_noise_scheduler": inference_noise_scheduler,
    }


class LatentDiffusionModule(QBaseModule):
    """
    PyTorch Lightning module for training latent diffusion models.
    
    This module wraps UNet training for diffusion models with optional
    class conditioning and quantization support.
    """
    
    def __init__(
        self,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        model_size: str = "small", 
        model_id: Union[str, None] = None,
        class_conditional: bool = False, 
        num_classes: Optional[int] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.model_size = model_size
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.model_id = model_id

        # Load model components
        self._load_models(model_size, class_conditional, num_classes)

    def _load_models(self, model_size: str, class_conditional: bool, num_classes: Optional[int]) -> None:
        """Load and initialize all model components."""
        model_dict = load_diffusion_model(
            model_size=model_size, 
            model_id=self.model_id,
            in_channels=4, 
            class_conditional=class_conditional, 
            num_classes=num_classes
        )
        
        self.vae = model_dict["vae"]
        self.model = model_dict["unet"]
        
        self.train_noise_scheduler = model_dict["train_noise_scheduler"]
        self.inference_noise_scheduler = model_dict["inference_noise_scheduler"]
        
        # Freeze VAE during training
        self.vae.requires_grad_(False)

    def forward(
        self, 
        pixel_values: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training: VAE encode -> add noise -> UNet predict noise.
        
        Args:
            pixel_values: Input images
            labels: Class labels (optional)
            
        Returns:
            Tuple of (predicted_noise, target_noise)
        """
        batch_size = pixel_values.shape[0]

        # Encode to latent space
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = latents.to(self.device)

        # Generate noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.train_noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=latents.device,
        ).long()

        # Add noise to latents
        noisy_latents = self.train_noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        if labels is None:
            model_pred = self.model(noisy_latents, timesteps).sample
        else:
            model_pred = self.model(noisy_latents, timesteps, class_labels=labels).sample

        return model_pred, noise

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        pixel_values = batch["pixel_values"]
        labels = batch.get("labels")

        model_pred, target = self.forward(pixel_values, labels)
        
        # Calculate losses
        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        reg_loss = self.reg_loss()
        total_loss = mse_loss + reg_loss

        # Log metrics
        self.log("mse_loss", mse_loss, prog_bar=True)
        self.log("reg_loss", reg_loss)
        self.log("train_loss", total_loss)

        if self.global_step % 500 == 0:
            self.log_stats()

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        pixel_values = batch["pixel_values"]
        self.image_height = pixel_values.shape[2]
        self.image_width = pixel_values.shape[3]
        labels = batch.get("labels")

        model_pred, target = self.forward(pixel_values, labels)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        self.log("val_loss", loss)
        
        # Call parent for quantization statistics
        super().validation_step(batch, batch_idx)
        
        return loss

    def on_validation_epoch_end(self) -> None:
        """Generate sample images at end of validation epoch."""
        if self.class_conditional:
            class_type = torch.randint(0, self.num_classes, (1,))
        else:
            class_type = None

        imgs = self.generate(
            batch_size=1,
            height=self.image_height * 2,
            width=self.image_width * 2,
            num_inference_steps=100,
            class_type=class_type
        )
        if self.class_conditional:
            self.logger.log_image(key="samples", images=imgs, caption=class_type)
        else: 
            self.logger.log_image(key="samples", images=imgs)


    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
            eps=1e-8,
        )

    def _get_ignore_conv2d_patterns(self) -> List[str]:
        """Get Conv2d layer patterns to ignore during quantization."""
        return [
            "conv_in",           # Input convolution
            "conv_out",          # Output convolution
            "conv_shortcut",     # Skip connections
        ]

    def _get_ignore_linear_patterns(self) -> List[str]:
        """Get Linear layer patterns to ignore during quantization."""
        return [
            "time_embedding",    # Time embeddings
            "time_emb_proj",     # Time projections
            "linear_1",          # Time embedding layers
            "linear_2",
            "class_embedding",   # Class embeddings
        ]

    @torch.no_grad()
    def generate(
        self,
        height: int = 128,
        width: int = 128,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        return_pil: bool = True,
        batch_size: int = 1,
        pbar: bool = False,
        class_type: int = 1,
    ) -> Union[List["Image.Image"], torch.Tensor]:
        """
        Generate images using the trained UNet model.
        
        Args:
            height: Generated image height
            width: Generated image width
            num_inference_steps: Number of denoising steps
            generator: Random generator for reproducibility
            return_pil: Whether to return PIL images or tensors
            batch_size: Number of images to generate
            pbar: Whether to show progress bar
            class_type: Class label for conditional generation
            
        Returns:
            Generated images as PIL Images or tensors
        """
        device = next(self.parameters()).device
        self.eval()

        scheduler = self.inference_noise_scheduler or self.train_noise_scheduler

        # Initialize latents
        latents = torch.randn(
            (batch_size, self.model.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        latents = latents * scheduler.init_noise_sigma

        # Class labels for conditional generation (if needed)
        labels = None
        if self.class_conditional: 
            labels = torch.tensor([class_type] * batch_size, device=device)

        # Denoising loop
        for t in tqdm(timesteps, disable=not pbar, desc="Generating"):
            latent_in = scheduler.scale_model_input(latents, timestep=t)
            
            # Conditional model call
            if self.class_conditional:
                noise_pred = self.model(latent_in, t, class_labels=labels).sample
            else:
                noise_pred = self.model(latent_in, t).sample
            
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode to images
        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        images = self.vae.decode(latents / scaling_factor).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        if not return_pil:
            return images

        # Convert to PIL images
        np_imgs = (images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype("uint8")
        from PIL import Image
        return [Image.fromarray(img) for img in np_imgs]






class PixelDiffusionModule(LatentDiffusionModule):
    """
    PyTorch Lightning module for training pixel-space diffusion models.
    
    This module wraps UNet training for diffusion models working directly
    on pixel values without VAE encoding/decoding.
    """
    
    def __init__(
        self,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        model_size: str = "small", 
        model_id: Union[str, None] = None,
        class_conditional: bool = False, 
        num_classes: Optional[int] = None,
        *args,
        **kwargs
    ):
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, model_size=model_size, class_conditional=class_conditional, num_classes=num_classes, model_id=model_id, *args, **kwargs)

    def _load_models(self, model_size: str, class_conditional: bool, num_classes: Optional[int]) -> None:
        """Load and initialize all model components."""
        model_dict = load_diffusion_model(
            model_size=model_size, 
            model_id=self.model_id,
            in_channels=3,  # UNet now works directly with pixel channels
            class_conditional=class_conditional, 
            num_classes=num_classes,
            pixel_space=True  # Add this flag to your model loading function
        )
        
        self.model = model_dict["unet"]
        
        self.train_noise_scheduler = model_dict["train_noise_scheduler"]
        self.inference_noise_scheduler = model_dict["inference_noise_scheduler"]

    def forward(
        self, 
        pixel_values: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training: normalize pixels -> add noise -> UNet predict noise.
        
        Args:
            pixel_values: Input images [B, C, H, W] in range [0, 1]
            labels: Class labels (optional)
            
        Returns:
            Tuple of (predicted_noise, target_noise)
        """
        batch_size = pixel_values.shape[0]

        pixels = pixel_values  # Assume normalization to [-1, 1] is already done
        assert pixels.min() >= -1.0 and pixels.max() <= 1.0 # pixels must be in [-1, 1] range
        pixels = pixels.to(self.device)

        # Generate noise and timesteps
        noise = torch.randn_like(pixels)
        timesteps = torch.randint(
            0,
            self.train_noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=pixels.device,
        ).long()

        # Add noise to pixels
        noisy_pixels = self.train_noise_scheduler.add_noise(pixels, noise, timesteps)

        # Predict noise
        if labels is None:
            model_pred = self.model(noisy_pixels, timesteps).sample
        else:
            model_pred = self.model(noisy_pixels, timesteps, class_labels=labels).sample

        return model_pred, noise


    def _get_ignore_linear_patterns(self) -> List[str]:
        """Get Linear layer patterns to ignore during quantization."""
        return [
            "time_embedding",    # Time embeddings
            "time_emb_proj",     # Time projections
            "linear_1",          # Time embedding layers
            "linear_2",
            "class_embedding",   # Class embeddings
        ]

    @torch.no_grad()
    def generate(
        self,
        height: int = 128,
        width: int = 128,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        return_pil: bool = True,
        batch_size: int = 1,
        pbar: bool = False,
        class_type: int = 1,
    ) -> Union[List["Image.Image"], torch.Tensor]:
        """
        Generate images using the trained UNet model in pixel space.
        
        Args:
            height: Generated image height
            width: Generated image width
            num_inference_steps: Number of denoising steps
            generator: Random generator for reproducibility
            return_pil: Whether to return PIL images or tensors
            batch_size: Number of images to generate
            pbar: Whether to show progress bar
            class_type: Class label for conditional generation
            
        Returns:
            Generated images as PIL Images or tensors
        """
        device = next(self.parameters()).device
        self.eval()

        scheduler = self.inference_noise_scheduler or self.train_noise_scheduler

        # Initialize pixel noise directly (no latent space)
        pixels = torch.randn(
            (batch_size, 3, height, width),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        pixels = pixels * scheduler.init_noise_sigma

        # Class labels for conditional generation
        if self.class_conditional:
            labels = torch.tensor([class_type] * batch_size, device=device)
        else:
            labels = None

        # Denoising loop
        for t in tqdm(timesteps, disable=not pbar, desc="Generating"):
            pixel_in = scheduler.scale_model_input(pixels, timestep=t)
            
            if labels is None:
                noise_pred = self.model(pixel_in, t).sample
            else:
                noise_pred = self.model(pixel_in, t, class_labels=labels).sample
                
            pixels = scheduler.step(noise_pred, t, pixels).prev_sample

        # Convert from [-1, 1] back to [0, 1]
        images = (pixels + 1.0) / 2.0
        images = images.clamp(0, 1)

        if not return_pil:
            return images

        # Convert to PIL images
        np_imgs = (images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype("uint8")
        from PIL import Image
        return [Image.fromarray(img) for img in np_imgs]