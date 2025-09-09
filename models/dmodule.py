
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
        sample_size=16,  # 128/8 = 16 for VAE latents
        in_channels=4, 
        out_channels=4,
        block_out_channels=(128, 256, 512),  # More capacity
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        layers_per_block=2,  # More layers per block
        attention_head_dim=8,
        num_class_embeds=None,
    )

    train_noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
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
        self, pixel_values: torch.Tensor
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
            noisy_latents, timesteps
        ).sample

        return model_pred, noise

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pixel_values = batch["pixel_values"]

        model_pred, target = self.forward(pixel_values)
        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        reg_loss = self.reg_loss()
        loss = mse_loss + reg_loss

        self.log("mse_loss", mse_loss, prog_bar=True)  # keep original logging keys/values
        self.log("reg_loss", reg_loss)
        self.log("train_loss", loss)

        if self.global_step % 100 == 0:
            self.log_stats()

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        # Support both 'caption' and 'captions' without changing existing behavior.

        model_pred, target = self.forward(pixel_values)
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

    # ------------- Layer Selection Overrides -------------

    def _get_quant_conv2d_layer_names(self) -> List[str]:
        """Only quantize small layers, avoid big ones."""
        ignore = [""]
        all_convs = get_all_conv2d_names(self.unet)
        return [name for name in all_convs if not any(x in name for x in ignore)]

    def _get_quant_linear_layer_names(self) -> List[str]:
        ignore = [""]
        all_linear = get_all_linear_names(self.unet)
        return [name for name in all_linear if not any(x in name for x in ignore)]

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

        # Denoising loop - unconditional generation
        for _, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=not pbar):
            latent_in = scheduler.scale_model_input(latents.to(device), timestep=t)
            
            # Unconditional UNet forward pass (no encoder_hidden_states)
            noise_pred = self.unet(latent_in, t).sample
            
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