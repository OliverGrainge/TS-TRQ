from typing import Optional, Dict, Any
import torch.nn as nn
from diffusers import UNet2DModel
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, EulerAncestralDiscreteScheduler
import os


def _load_model(
    model_type: str,
    in_channels: int = 3,
    class_conditional: bool = False,
    num_classes: Optional[int] = None,
    sample_size: int = 32,
) -> nn.Module:
    """
    Load UNet optimized for diffusion at different resolutions.
    
    Automatically adjusts architecture depth based on image size:
    - 32x32: 4 downsample blocks (→ 2x2 bottleneck)
    - 64x64: 4 downsample blocks (→ 4x4 bottleneck)
    - 128x128: 5 downsample blocks (→ 4x4 bottleneck)
    - 256x256: 5 downsample blocks (→ 8x8 bottleneck)
    
    Args:
        model_type: Size of the model ("tiny", "small", "base", "large")
        in_channels: Number of input channels (3 for RGB pixel, 4 for latent)
        class_conditional: Whether to use class conditioning
        num_classes: Number of classes for class conditioning
        sample_size: Size of input images (32, 64, 128, 256)
    
    Returns:
        UNet2DModel configured for the specified resolution
    """
    # Determine number of downsample blocks based on resolution
    if sample_size <= 32:
        num_blocks = 4
    elif sample_size == 64:
        num_blocks = 4
    elif sample_size == 128:
        num_blocks = 5
    else:  # 256+
        num_blocks = 5
    
    # Configuration for different model sizes
    # Format: (base_channels, channel_multipliers, attention_start_layer)
    size_configs = {
        "tiny": (128, (1, 1, 2, 2, 4)[:num_blocks], 2),
        "small": (128, (1, 2, 2, 4, 4)[:num_blocks], 1),
        "base": (128, (1, 2, 4, 4, 8)[:num_blocks], 1),
        "large": (192, (1, 2, 3, 4, 4)[:num_blocks], 1),
    }
    
    if model_type not in size_configs:
        raise ValueError(
            f"model_type must be one of {list(size_configs.keys())}, got {model_type}"
        )
    
    base_channels, multipliers, attn_start = size_configs[model_type]
    
    # Calculate block output channels
    block_out_channels = tuple(base_channels * m for m in multipliers)
    
    # Build block types - add attention from attn_start layer onward
    down_block_types = []
    up_block_types = []
    
    for i in range(num_blocks):
        if i >= attn_start:
            down_block_types.append("AttnDownBlock2D")
            up_block_types.insert(0, "AttnUpBlock2D")  # Reverse order for up
        else:
            down_block_types.append("DownBlock2D")
            up_block_types.insert(0, "UpBlock2D")
    
    # Adjust attention head dimension based on image size
    if sample_size <= 64:
        attention_head_dim = 8
        layers_per_block = 2
    elif sample_size == 128:
        attention_head_dim = 8
        layers_per_block = 2
    else:  # 256+
        attention_head_dim = 8 if model_type == "tiny" else 16
        layers_per_block = 2
    
    
    if model_type not in size_configs:
        raise ValueError(
            f"model_type must be one of {list(size_configs.keys())}, got {model_type}"
        )
    
    base_channels, multipliers, attn_start = size_configs[model_type]
    
    # Calculate block output channels
    block_out_channels = tuple(base_channels * m for m in multipliers)
    
    # Build block types - add attention from attn_start layer onward
    down_block_types = []
    up_block_types = []
    
    for i in range(num_blocks):
        if i >= attn_start:
            down_block_types.append("AttnDownBlock2D")
            up_block_types.insert(0, "AttnUpBlock2D")  # Reverse order for up
        else:
            down_block_types.append("DownBlock2D")
            up_block_types.insert(0, "UpBlock2D")
    
    # Adjust attention head dimension based on image size
    if sample_size <= 64:
        attention_head_dim = 8
        layers_per_block = 2
    elif sample_size == 128:
        attention_head_dim = 8
        layers_per_block = 2
    else:  # 256+
        attention_head_dim = 8 if model_type == "tiny" else 16
        layers_per_block = 2
    
    config = {
        "block_out_channels": block_out_channels,
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "attention_head_dim": attention_head_dim,
        "layers_per_block": layers_per_block,
        "norm_num_groups": 32,
    }
    
    # Build UNet configuration
    unet_kwargs = {
        "sample_size": sample_size,
        "in_channels": in_channels,
        "out_channels": in_channels,
        "down_block_types": config["down_block_types"],
        "up_block_types": config["up_block_types"],
        "block_out_channels": config["block_out_channels"],
        "layers_per_block": config["layers_per_block"],
        "attention_head_dim": config["attention_head_dim"],
        "norm_num_groups": config["norm_num_groups"],
        "norm_eps": 1e-5,
        "resnet_time_scale_shift": "default",
        "add_attention": True,
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


def load_diffusion_model(
    model_id: Optional[str] = None,
    model_size: str = "tiny",
    in_channels: int = 3,
    class_conditional: bool = False,
    num_classes: Optional[int] = None,
    pixel_space: bool = None,  # Auto-determine if None
    sample_size: int = 32,
) -> Dict[str, Any]:
    """
    Load diffusion model components optimized for different resolutions.
    
    Automatically determines optimal settings based on image size:
    - 32-64px: Pixel-space with linear schedule
    - 128-256px: Can do pixel-space OR latent-space (latent recommended for speed)
    
    Args:
        model_id: Pretrained model ID (if loading pretrained)
        model_size: Size of UNet model ("tiny", "small", "base", "large")
        in_channels: Number of input channels (auto-set if None)
        class_conditional: Whether to use class conditioning
        num_classes: Number of classes for conditioning
        pixel_space: Whether to use pixel space (auto-determined if None)
        sample_size: Size of input images (32, 64, 128, 256)
    
    Returns:
        Dictionary with model components
    """
    cache_dir = os.getenv("HF_TRANSFORMERS_CACHE")
    
    # Load pretrained if specified
    if model_id is not None:
        return load_pretrained_diffusion(model_id, cache_dir=cache_dir)
    
    # Auto-determine pixel_space if not specified
    if pixel_space is None:
        # Use pixel space for smaller images, latent for larger
        pixel_space = sample_size <= 64
    
    # Load VAE only if using latent space
    vae = None
    if not pixel_space:
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-ema",
            cache_dir=cache_dir
        )
        # VAE downsamples by 8x and produces 4 channels
        actual_sample_size = sample_size // 8
        actual_in_channels = 4
    else:
        actual_sample_size = sample_size
        actual_in_channels = in_channels
    
    # Load UNet with specified configuration
    model = _load_model(
        model_type=model_size,
        in_channels=actual_in_channels,
        class_conditional=class_conditional,
        num_classes=num_classes,
        sample_size=actual_sample_size,
    )
    
    # Adjust noise schedule based on resolution and pixel/latent space
    # In load_diffusion_model function, replace the scheduler section:

    # Adjust noise schedule based on resolution and pixel/latent space
    if pixel_space:
        # Pixel space: linear schedule, lower betas
        if sample_size <= 64:
            beta_start, beta_end = 0.0001, 0.01  # ← FIXED: Lower beta_end
            beta_schedule = "linear"
        else:  # 128-256 pixel space
            beta_start, beta_end = 0.0001, 0.015
            beta_schedule = "linear"
    else:
        # Latent space: scaled_linear or squaredcos, higher betas
        beta_start, beta_end = 0.00085, 0.012
        beta_schedule = "scaled_linear"

    # Training scheduler
    train_noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        variance_type="fixed_small" if pixel_space else "learned_range",
        prediction_type="epsilon",
        clip_sample=True,           # ← CRITICAL: Enable clipping
        clip_sample_range=1.0,       # ← ADDED: Clip to [-1, 1]
    )

    # Inference scheduler - DDIM with more steps for 32x32
    if sample_size <= 64:
        inference_noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type="epsilon",
            clip_sample=True,        # ← Enable for inference too
            set_alpha_to_one=False,
            steps_offset=1,
        )
    else:
        # DPM-Solver or Euler-A for larger images
        inference_noise_scheduler = EulerAncestralDiscreteScheduler.from_config(
            train_noise_scheduler.config
        )
        inference_noise_scheduler.config.use_karras_sigmas = True
        inference_noise_scheduler.config.timestep_spacing = "trailing"
    
    return {
        "vae": vae,
        "model": model,
        "train_noise_scheduler": train_noise_scheduler,
        "inference_noise_scheduler": inference_noise_scheduler,
        "pixel_space": pixel_space,
        "sample_size": sample_size,
    }


def load_pretrained_diffusion(
    model_id: str,
    cache_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Load pretrained diffusion models."""
    from diffusers import DDPMPipeline, DiTPipeline
    
    if model_id in ["google/ddpm-cifar10-32"]:
        pipeline = DDPMPipeline.from_pretrained(model_id, cache_dir=cache_dir)
        
        return {
            "model": pipeline.unet,
            "train_noise_scheduler": pipeline.scheduler,
            "inference_noise_scheduler": DDIMScheduler.from_config(
                pipeline.scheduler.config
            ),
            "vae": None,
            "pixel_space": False,
            "sample_size": 32,
        }
    
    elif model_id in ["google/ddpm-ema-church-256", "google/ddpm-ema-bedroom-256"]:
        pipeline = DDPMPipeline.from_pretrained(model_id, cache_dir=cache_dir)
        
        # Use DDIM for faster inference on larger images
        inference_scheduler = DDIMScheduler.from_config(
            pipeline.scheduler.config
        )
        
        return {
            "model": pipeline.unet,
            "train_noise_scheduler": pipeline.scheduler,
            "inference_noise_scheduler": inference_scheduler,
            "vae": None,
            "pixel_space": False, 
            "sample_size": 256,
        }
    
    elif model_id in ["facebook/DiT-XL-2-256"]:
        pipeline = DiTPipeline.from_pretrained(model_id, cache_dir=cache_dir)
        
        train_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        
        return {
            "model": pipeline.transformer,
            "train_noise_scheduler": train_scheduler,
            "inference_noise_scheduler": pipeline.scheduler,
            "vae": pipeline.vae,
            "pixel_space": False, 
            "sample_size": 256,
        }
    
    else:
        raise ValueError(f"Unknown model ID: {model_id}")
