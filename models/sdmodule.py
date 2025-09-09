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
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
    LMSDiscreteScheduler,
)
from data import ImageNetDataModule
from quant import quantize_model, get_all_conv2d_names, get_all_linear_names
from .base import QBaseModule  # Import the base class



# -------------------------
# Stable Diffusion loaders
# -------------------------
def load_stable_diffusion(pretrained: bool = True) -> Dict[str, Any]:
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

    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae", cache_dir=cache_dir
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=cache_dir
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=cache_dir
    )

    if pretrained:
        unet = UNet2DConditionModel.from_pretrained(
            "nota-ai/bk-sdm-tiny-2m", subfolder="unet", cache_dir=cache_dir
        )
    else:
        config = UNet2DConditionModel.load_config(
            "nota-ai/bk-sdm-tiny-2m", subfolder="unet", cache_dir=cache_dir
        )
        unet = UNet2DConditionModel.from_config(config)

    train_noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    inference_noise_scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    return {
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "unet": unet,
        "train_noise_scheduler": train_noise_scheduler,
        "inference_noise_scheduler": inference_noise_scheduler,
    }


# ---------------------------------
# Lightning training module
# ---------------------------------

class StableDiffusionModule(QBaseModule):
    def __init__(
        self,
        learning_rate: float = 2e-4,
        gradient_checkpointing: bool = True,
        train_text_encoder: bool = False,
        pretrained: bool = True,
        *args,
        **kwargs
    ):
        """
        Minimal Stable Diffusion training wrapper around UNet + text encoder.

        Args:
            learning_rate: Optimizer LR.
            gradient_checkpointing: Enable gradient checkpointing for memory.
            train_text_encoder: Whether to unfreeze text encoder for training.
            pretrained: Load UNet weights vs. config-init.
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.train_text_encoder = train_text_encoder
        self.gradient_checkpointing = gradient_checkpointing
        self.pretrained = pretrained

        # Load core components
        model_dict = load_stable_diffusion(pretrained=pretrained)
        self.vae = model_dict["vae"]
        self.text_encoder = model_dict["text_encoder"]
        self.tokenizer = model_dict["tokenizer"]
        self.unet = model_dict["unet"]

        # Set the model for the base class
        self.model = self.unet

        # Schedulers
        self.train_noise_scheduler = model_dict["train_noise_scheduler"]
        self.inference_noise_scheduler = model_dict["inference_noise_scheduler"]

        # Freeze VAE and (optionally) text encoder
        self.vae.requires_grad_(False)
        if not self.train_text_encoder:
            self.text_encoder.requires_grad_(False)

        # Gradient checkpointing
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        # Eval mode for frozen parts
        self.vae.eval()
        if not self.train_text_encoder:
            self.text_encoder.eval()

    def setup_model(self) -> None:
        """
        Initialize the main model components (UNet, text encoder, etc.).
        Implementation of abstract method from QBaseModule.
        """
        if self.model is None:
            model_dict = load_stable_diffusion(pretrained=self.pretrained)
            self.vae = model_dict["vae"]
            self.text_encoder = model_dict["text_encoder"]
            self.tokenizer = model_dict["tokenizer"]
            self.unet = model_dict["unet"]
            self.model = self.unet
            
            self.train_noise_scheduler = model_dict["train_noise_scheduler"]
            self.inference_noise_scheduler = model_dict["inference_noise_scheduler"]
            
            # Freeze VAE and (optionally) text encoder
            self.vae.requires_grad_(False)
            if not self.train_text_encoder:
                self.text_encoder.requires_grad_(False)

            # Gradient checkpointing
            if self.gradient_checkpointing:
                self.unet.enable_gradient_checkpointing()
                if self.train_text_encoder:
                    self.text_encoder.gradient_checkpointing_enable()

            # Eval mode for frozen parts
            self.vae.eval()
            if not self.train_text_encoder:
                self.text_encoder.eval()

    # ------------- Utilities -------------

    def _encode_text(
        self, batch_size: int, captions: Optional[List[str]]
    ) -> torch.Tensor:
        """
        Tokenize and encode captions to CLIP hidden states.

        If captions is None, produces unconditional embeddings using empty strings
        with max_length=20 (kept to preserve original behavior).
        """
        if captions is None:
            uncond = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=20,  # preserves original behavior
                return_tensors="pt",
            )
            return self.text_encoder(uncond.input_ids.to(self.device))[0]

        text_input = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        ctx = torch.enable_grad() if self.train_text_encoder else torch.no_grad()
        with ctx:
            return self.text_encoder(text_input.input_ids.to(self.device))[0]

    def _get_params_to_optimize(self) -> List[torch.nn.Parameter]:
        params = list(self.unet.parameters())
        if self.train_text_encoder:
            params += list(self.text_encoder.parameters())
        return params

    # ------------- Forward / Steps -------------

    def forward(
        self, pixel_values: torch.Tensor, captions: Optional[List[str]] = None
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

        # Text embeddings (conditional or unconditional)
        encoder_hidden_states = self._encode_text(bsz, captions)

        # UNet predicts noise residual
        model_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
        ).sample

        return model_pred, noise

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        captions = batch.get("captions", None)

        model_pred, target = self.forward(pixel_values, captions)
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
        captions = batch.get("caption", batch.get("captions", None))

        model_pred, target = self.forward(pixel_values, captions)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        self.log("val_loss", loss)
        
        # Call parent validation_step for quantization statistics logging
        super().validation_step(batch, batch_idx)
        
        return loss

    def on_validation_epoch_end(self) -> None:
        imgs = self.generate(
            batch_size=4,
            height=512,
            width=512,
            num_inference_steps=100,
            guidance_scale=7.5,
        )
        self.logger.log_image(key="samples", images=imgs)

    # ------------- Optimizer -------------

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self._get_params_to_optimize(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        )

    # ------------- Layer Selection Overrides -------------

    def _get_quant_conv2d_layer_names(self) -> List[str]:
        """Only quantize small layers, avoid big ones."""
        if self.model is None:
            return []
        
        ignore = [
            # Your existing ignores
            "conv_in", "conv_out", "conv_norm_out",
            "time_proj", "time_embedding", "time_emb_proj", 
            "Downsample2D.conv", "Upsample2D.conv",
            "conv_shortcut", ".norm", "GroupNorm", "LayerNorm",
            "attn2.to_k", "attn2.to_v", "proj_in", "proj_out",
            
            # Avoid all deep blocks (pattern-based)
            "down_blocks.2",  # All 1280-channel blocks
            "up_blocks.0",    # All 1280-channel blocks  
            "down_blocks.1",  # All 640-channel blocks
            "up_blocks.1",    # All 640-channel blocks
        ]
        
        all_convs = get_all_conv2d_names(self.unet)
        return [name for name in all_convs if not any(x in name for x in ignore)]

    def _get_quant_linear_layer_names(self) -> List[str]:
        if self.model is None:
            return []
        
        ignore = [
            "conv_in", "conv_out", "conv_norm_out",
            "time_proj", "time_embedding", "time_emb_proj",
            "Downsample2D.conv", "Upsample2D.conv",
            "conv_shortcut",
            ".norm", "GroupNorm", "LayerNorm",
            "attn2.to_k", "attn2.to_v",
        ]
        all_linear = get_all_linear_names(self.unet)
        return [name for name in all_linear if not any(x in name for x in ignore)]

    def _get_ignore_conv2d_patterns(self) -> List[str]:
        """
        Get patterns for Conv2d layer names to ignore during quantization.
        Override of base class method.
        """
        return [
            "conv_in", "conv_out", "conv_norm_out",
            "time_proj", "time_embedding", "time_emb_proj", 
            "Downsample2D.conv", "Upsample2D.conv",
            "conv_shortcut", ".norm", "GroupNorm", "LayerNorm",
            "attn2.to_k", "attn2.to_v", "proj_in", "proj_out",
            "down_blocks.2",  # All 1280-channel blocks
            "up_blocks.0",    # All 1280-channel blocks  
            "down_blocks.1",  # All 640-channel blocks
            "up_blocks.1",    # All 640-channel blocks
        ]

    def _get_ignore_linear_patterns(self) -> List[str]:
        """
        Get patterns for Linear layer names to ignore during quantization.
        Override of base class method.
        """
        return [
            "conv_in", "conv_out", "conv_norm_out",
            "time_proj", "time_embedding", "time_emb_proj",
            "Downsample2D.conv", "Upsample2D.conv",
            "conv_shortcut",
            ".norm", "GroupNorm", "LayerNorm",
            "attn2.to_k", "attn2.to_v",
        ]

    # ------------- Inference (Generation) -------------

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str], None] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Union[str, List[str], None] = None,
        generator: Optional[torch.Generator] = None,
        return_pil: bool = True,
        batch_size: int = 1,
        pbar: bool = False,
    ) -> Union[List["Image.Image"], torch.Tensor]:
        """
        Text-to-image (or unconditional) generation using the current model.
        Preserves device moves and scheduler usage from the original code.
        """
        # Keep original device move behavior (even though Lightning manages device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        self.to(device)
        self.vae.to(device)
        self.unet.to(device)
        self.text_encoder.to(device)

        scheduler = self.inference_noise_scheduler or self.train_noise_scheduler

        if prompt is None:
            is_conditional = False
            effective_bsz = batch_size

            uncond = self.tokenizer(
                [""] * effective_bsz,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_embeddings = self.text_encoder(uncond["input_ids"].to(device))[0]
        else:
            is_conditional = True
            if isinstance(prompt, str):
                prompt = [prompt]
            effective_bsz = len(prompt)

            text_in = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = self.text_encoder(text_in["input_ids"].to(device))[0]

            if negative_prompt is not None:
                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt] * effective_bsz
                elif len(negative_prompt) != effective_bsz:
                    raise ValueError(
                        f"Length of negative_prompt ({len(negative_prompt)}) must match batch ({effective_bsz})"
                    )
                uncond_in = self.tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
            else:
                uncond_in = self.tokenizer(
                    [""] * effective_bsz,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )

            uncond_embeddings = self.text_encoder(uncond_in["input_ids"].to(device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Initial noise (latent space is 8x downsampled)
        latents = torch.randn(
            (effective_bsz, self.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=device,
            dtype=text_embeddings.dtype,
        )

        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        latents = latents * scheduler.init_noise_sigma

        # Denoising loop
        for _, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=not pbar):
            if is_conditional:
                latent_in = torch.cat([latents] * 2)
                latent_in = scheduler.scale_model_input(latent_in.to(device), timestep=t)

                noise_pred = self.unet(latent_in, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                latent_in = scheduler.scale_model_input(latents.to(device), timestep=t)
                noise_pred = self.unet(latent_in, t, encoder_hidden_states=text_embeddings).sample

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