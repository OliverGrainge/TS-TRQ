
import os
from typing import Any, Dict, List, Optional, Tuple, Union

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
from .base import QBaseModule  # Import the base class



# -------------------------
# Stable Diffusion loaders
# -------------------------
def load_diffusion_model() -> Dict[str, Any]:
    """
    Load VAE, text encoder, tokenizer, UNet, and schedulers for Stable Diffusion.

    Args:
        pretrained: Whether to load a pretrained UNet or create from config.

    Returns:
        Dictionary with keys:
        ['vae', 'text_encoder', 'tokenizer', 'unet',
         'train_noise_scheduler', 'inference_noise_scheduler']
    """
    cache_dir = os.getenv("HF_TRANSFORMERS_CACHE")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", cache_dir=cache_dir)

    unet = UNet2DModel(
        class_embed_type="timestep",
        num_class_embeds=1000,
        
        sample_size=128,  # Match your training resolution
        in_channels=4,
        out_channels=4,
        
        # Larger architecture for better quality
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(128, 256, 512, 1024), 
        
        layers_per_block=2,
        attention_head_dim=16,  # Increased from 8
        norm_num_groups=32,
    )


    train_noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",  # Better than linear
        beta_start=0.00085,
        beta_end=0.012,
        clip_sample=False,
    )
    inference_noise_scheduler = EulerAncestralDiscreteScheduler.from_config(train_noise_scheduler.config)

    return {
        "vae": vae,
        "unet": unet,
        "train_noise_scheduler": train_noise_scheduler,
        "inference_noise_scheduler": inference_noise_scheduler,
    }


# ---------------------------------
# Lightning training module
# ---------------------------------
class DiffusionModule(QBaseModule):
    def __init__(
        self,
        batch_size: int,
        learning_rate: float = 2e-4,
        *args,
        **kwargs
    ):
        """
        Minimal Stable Diffusion training wrapper around UNet + text encoder.

        Args:
            learning_rate: Optimizer LR.
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Load core components
        model_dict = load_diffusion_model()
        self.vae = model_dict["vae"]
        self.unet = model_dict["unet"]

        # Set the model for the base class
        self.model = self.unet

        # Schedulers
        self.train_noise_scheduler = model_dict["train_noise_scheduler"]
        self.inference_noise_scheduler = model_dict["inference_noise_scheduler"]

        # Freeze VAE
        self.vae.requires_grad_(False)

    def setup_model(self) -> None:
        """
        Initialize the main model components (unet, etc.).
        Implementation of abstract method from QBaseModule.
        """
        if self.model is None:
            model_dict = load_diffusion_model()
            self.vae = model_dict["vae"]
            self.unet = model_dict["unet"]
            self.model = self.unet
            
            self.train_noise_scheduler = model_dict["train_noise_scheduler"]
            self.inference_noise_scheduler = model_dict["inference_noise_scheduler"]
            
            # Freeze VAE
            self.vae.requires_grad_(False)

    # ------------- Forward / Steps -------------

    def forward(
        self, pixel_values: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One training forward pass:
          VAE encode -> add noise -> UNet predict noise
        """
        bsz = pixel_values.shape[0]

        # Latents
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = latents.to(self.device)
            assert latents.shape[1] == self.unet.config.in_channels

        # Noise + timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.train_noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        ).long()

        noisy_latents = self.train_noise_scheduler.add_noise(latents, noise, timesteps)

        # UNet predicts noise residual
        model_pred = self.unet(
            noisy_latents, timesteps, class_labels=labels
        ).sample

        return model_pred, noise

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        model_pred, target = self.forward(pixel_values, labels)
        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        reg_loss = self.reg_loss()
        loss = mse_loss + reg_loss

        self.log("mse_loss", mse_loss, prog_bar=True)  # keep original logging keys/values
        self.log("reg_loss", reg_loss)
        self.log("train_loss", loss)

        if self.global_step % 500 == 0:
            self.log_stats()

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        # Support both 'caption' and 'captions' without changing existing behavior.

        model_pred, target = self.forward(pixel_values, labels)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        self.log("val_loss", loss)
        
        # Call parent validation_step for quantization statistics logging
        super().validation_step(batch, batch_idx)
        
        return loss

    def on_validation_epoch_end(self) -> None:
        imgs = self.generate(
            batch_size=1,
            height=128,
            width=128,
            num_inference_steps=100,
        )
        self.logger.log_image(key="samples", images=imgs)

    # ------------- Optimizer -------------

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.unet.parameters()),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        )


    def _get_ignore_conv2d_patterns(self) -> List[str]:
        """
        Get patterns for Conv2d layer names to ignore during quantization.
        These layers are critical for model stability and should remain full precision.
        """
        return [
            # Input/Output layers - critical for maintaining data flow
            "conv_in",           # Initial input convolution
            "conv_out",          # Final output convolution

            # Deeper layers in the network - more sensitive to quantization
            #"down_blocks.2",     # Third down block (512 channels)
            "down_blocks.3",     # Fourth down block (1024 channels) 
            "up_blocks.0",       # First up block (1024 channels)
            #"up_blocks.1",       # Second up block (512 channels)
            "mid_block",         # Middle block - bottleneck, most critical
            
            
            # Shortcut connections - maintain residual flow
            "conv_shortcut",     # Skip connections in ResNet blocks
            
            # Small kernel convolutions that are already efficient
            # Note: 1x1 convs are often used for dimension matching and are lightweight
        ]

    def _get_ignore_linear_patterns(self) -> List[str]:
        """
        Get patterns for Linear layer names to ignore during quantization.
        These layers handle embeddings and attention which need higher precision.
        """
        return [
            # Time embedding layers - critical for diffusion timestep conditioning
            "time_embedding",    # Timestep embeddings
            "time_emb_proj",
            "linear_1",          # First linear layer in time embeddings
            "linear_2",          # Second linear layer in time embeddings
            
            # Class embedding layers - for conditional generation
            "class_embedding",   # Class conditioning embeddings
            
            # Time projection layers
            "time_emb_proj",     # Time embedding projections into ResNet blocks
            
            # First and last layers of attention (most sensitive)
            # You might want to be more selective here based on performance
            # "to_q",            # Query projection - consider keeping these
            # "to_k",            # Key projection - consider keeping these  
            # "to_v",            # Value projection - consider keeping these
            # "to_out.0",        # Output projection - consider keeping these
        ]

    # ------------- Inference (Generation) -------------

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
        Unconditional image generation using the current UNet model.
        """
        # Keep original device move behavior (even though Lightning manages device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        self.to(device)
        self.vae.to(device)
        self.unet.to(device)

        scheduler = self.inference_noise_scheduler or self.train_noise_scheduler

        # Initial noise (latent space is 8x downsampled)
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=device,
            dtype=torch.float32,  # Use explicit dtype since we don't have text_embeddings
        )

        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        latents = latents * scheduler.init_noise_sigma

        labels = torch.tensor([class_type] * batch_size, device=device)

        # Denoising loop - unconditional generation
        for _, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=not pbar):
            latent_in = scheduler.scale_model_input(latents.to(device), timestep=t)
            
            # Unconditional UNet forward pass (no encoder_hidden_states)
            noise_pred = self.unet(latent_in, t, class_labels=labels).sample
            
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode to images
        latents = latents.to(device)
        scaling = getattr(self.vae.config, "scaling_factor", 0.18215)
        images = self.vae.decode(latents / scaling).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        if not return_pil:
            return images

        # To PIL
        np_imgs = (images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype("uint8")
        from PIL import Image  # local import to avoid top-level dependency when unused
        return [Image.fromarray(img) for img in np_imgs]