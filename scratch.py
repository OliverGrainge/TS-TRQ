import torch
from dotenv import load_dotenv

load_dotenv()

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from typing import Any 
from typing import Union
from tqdm import tqdm
import torch 
from diffusers import (
    UNet2DConditionModel, 
    DDPMScheduler, 
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from diffusers import LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import torchvision.transforms as transforms
import os
from typing import Optional, List, Tuple
from data import LSUNBedroomDataModule
from diffusers import DiffusionPipeline
from quant import quantize_model, get_all_conv2d_names, get_all_linear_names



def load_stable_diffusion(): 
    cache_dir = os.getenv("HF_TRANSFORMERS_CACHE")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", cache_dir=cache_dir) 
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", cache_dir=cache_dir)
    train_noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
    inference_noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    return {"vae": vae, "text_encoder": text_encoder, "tokenizer": tokenizer, "unet": unet, "train_noise_scheduler": train_noise_scheduler, "inference_noise_scheduler": inference_noise_scheduler}



class StableDiffusionTrainingModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 500,
        max_train_steps: int = 500000,
        gradient_checkpointing: bool = True,
        train_text_encoder: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.max_train_steps = max_train_steps
        self.train_text_encoder = train_text_encoder
        self.is_quantized = False
        
        # Load the models
        model_dict = load_stable_diffusion()
        self.vae = model_dict["vae"]
        self.text_encoder = model_dict["text_encoder"]
        self.tokenizer = model_dict["tokenizer"]
        self.unet = model_dict["unet"]
        
        # Scheduler for noise
        self.train_noise_scheduler = model_dict["train_noise_scheduler"]
        self.inference_noise_scheduler = model_dict["inference_noise_scheduler"]
        
        # Freeze VAE and text encoder (unless specified otherwise)
        self.vae.requires_grad_(False)
        if not self.train_text_encoder:
            self.text_encoder.requires_grad_(False)
        
        # Enable gradient checkpointing to save memory
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()
        
        # Move VAE to eval mode
        self.vae.eval()
        if not self.train_text_encoder:
            self.text_encoder.eval()
    
    def encode_text(self, batch_size: int, captions: Union[None, List[str]]):
        """Encode text captions to embeddings"""
        if captions is None: 
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=20, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0] 
            return uncond_embeddings

        text_input = self.tokenizer(captions, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        
        with torch.no_grad() if not self.train_text_encoder else torch.enable_grad():
            text_embeddings = self.text_encoder(
                captions.input_ids.to(self.device)
            )[0]   
        return text_embeddings
    
    def forward(self, pixel_values, captions=None):
        """Forward pass for diffusion model training"""
        # Convert images to latent space
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = latents.to(self.device)
            assert latents.shape[1] == self.unet.config.in_channels
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.train_noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=latents.device
        ).long()
        
        # Add noise to latents according to noise magnitude at each timestep
        noisy_latents = self.train_noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings - handle both conditional and unconditional cases
        # Use actual captions instead of hardcoded string
        encoder_hidden_states = self.encode_text(len(pixel_values),captions)
        
        # Predict noise with UNet
        model_pred = self.unet(
            noisy_latents, 
            timesteps, 
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        return model_pred, noise
    
    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        if 'caption' not in batch.keys(): 
            captions = None
        else:
            captions = batch['caption']
        
        
        model_pred, target = self.forward(pixel_values, captions)
        
        # Calculate loss
        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        reg_loss = self.reg_loss()
        loss = mse_loss + reg_loss
        
        self.log("mse_loss", loss, prog_bar=True)
        self.log("reg_loss", reg_loss)
        self.log("train_loss", loss)
        if self.global_step % 100 == 0:
            self.log_stats()
        return loss
    
    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        if 'caption' not in batch.keys(): 
            captions = None
        else:
            captions = batch['caption']
        
        
        model_pred, target = self.forward(pixel_values, captions)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self): 
        imgs = self.generate(batch_size=4, height=512, width=512, num_inference_steps=100, guidance_scale=7.5)
        # generate() returns a list of PIL Images, so we pass it directly
        self.logger.log_image(key="samples", images=imgs)
    
    def configure_optimizers(self):
        # Determine which parameters to optimize
        params_to_optimize = list(self.unet.parameters())
        if self.train_text_encoder:
            params_to_optimize += list(self.text_encoder.parameters())
        
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        )
        
        scheduler = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.max_train_steps,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def apply_quantization(self, quant_type: str, **quant_kwargs: Any) -> None:
        """
        Apply quantization to the model after construction/loading.

        Args:
            quant_type: Type of quantization to apply
            **quant_kwargs: Additional keyword arguments for quantization
        """
        if not quant_type:
            print(
                "No quantization type specified or quantization disabled. Skipping quantization."
            )
            return

        if self.is_quantized:
            print(f"Model is already quantized with {self.quant_type}. Skipping.")
            return

        print(
            f"Applying {quant_type} quantization to model with kwargs: {quant_kwargs}"
        )

        layer_names = self._get_quant_conv2d_layer_names()
        layer_names = ["up_blocks.2.resnets.2.conv1"]
        self.unet = quantize_model(
            self.unet, layer_names=layer_names, quant_type=self._update_quant_type(quant_type, "conv2d"), **quant_kwargs
        )
        #layer_names = self._get_quant_linear_layer_names()[
        #self.unet = quantize_model(
        #    self.unet, layer_names=layer_names, quant_type=self._update_quant_type(quant_type, "linear"), **quant_kwargs
        #)
        # Update quantization state
        self.is_quantized = True
        self.quant_type = quant_type
        self.quant_kwargs = quant_kwargs
    
    def _update_quant_type(self, quant_type: str, layer_type: str):
        if "tsvd" in quant_type and layer_type == "conv2d":
            quant_type = "tsvdconv2d"
        elif "tsvd" in quant_type and layer_type == "linear":
            quant_type = "tsvdlinear"
        elif "t" in quant_type and layer_type == "conv2d":
            quant_type = "tconv2d"
        elif "t" in quant_type and layer_type == "linear":
            quant_type = "tlinear"
        return quant_type

    def _get_quant_conv2d_layer_names(self) -> List[str]: 
        ignore_layers = [
            "conv_in", "conv_out", "conv_norm_out",
            "time_proj", "time_embedding", "time_emb_proj",
            "Downsample2D.conv", "Upsample2D.conv",
            "conv_shortcut",
            ".norm", "GroupNorm", "LayerNorm",
            "attn2.to_k", "attn2.to_v"
        ]
        all_convs = get_all_conv2d_names(self.unet)
        convs = [l for l in all_convs if not any(x in l for x in ignore_layers)]
        return convs

    def _get_quant_linear_layer_names(self) -> List[str]: 
        """
        Returns a list of linear layer names in self.unet to be quantized,
        excluding those that match any of the ignore patterns.
        """
        ignore_layers = [
            "conv_in", "conv_out", "conv_norm_out",
            "time_proj", "time_embedding", "time_emb_proj",
            "Downsample2D.conv", "Upsample2D.conv",
            "conv_shortcut",
            ".norm", "GroupNorm", "LayerNorm",
            "attn2.to_k", "attn2.to_v"
        ]
        all_lines = get_all_linear_names(self.unet)
        lines = [l for l in all_lines if not any(x in l for x in ignore_layers)]
        return lines

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
        for m in self.unet.modules():
            fn = getattr(m, "layer_reg_loss", None)
            if callable(fn):
                losses.append(fn())
                
        if not losses:
            return torch.tensor(0.0, device=self.device)
            
        losses = torch.stack([torch.as_tensor(l, device=self.device) for l in losses])
        return losses.mean() if reduction == "mean" else losses.sum()


    def ternary_vs_lr_ratio(self) -> dict:
        """
        Compute the ratio of ternary vs low-rank contributions across all TSVD layers.
        
        Returns:
            Dictionary with global statistics and raw values for histogram logging:
            {
                'mean': float,
                'median': float, 
                'min': float,
                'max': float,
                'values': List[float]  # Raw ratio values for histogram logging
            }
        """
        if not self.is_quantized or "tsvd" not in self.quant_type:
            return {
                'mean': 1.0, 'median': 1.0, 'min': 1.0, 'max': 1.0,
                'values': [1.0]
            }
        
        ratios = []
        
        # Compute ratio for each TSVD layer
        for module in self.unet.modules():
            if self._is_tsvd_layer(module):
                ratio = self._compute_layer_ratio(module)
                if ratio is not None:
                    ratios.append(ratio.item())
        
        if not ratios:
            return {
                'mean': 1.0, 'median': 1.0, 'min': 1.0, 'max': 1.0,
                'values': [1.0]
            }
        
        # Convert to tensor for easy statistics
        ratios_tensor = torch.tensor(ratios, device=self.device)
        
        return {
            'mean': ratios_tensor.mean().item(),
            'median': ratios_tensor.median().item(),
            'min': ratios_tensor.min().item(),
            'max': ratios_tensor.max().item(),
            'values': ratios  # Raw values for histogram logging
        }

    def _is_tsvd_layer(self, module: torch.nn.Module) -> bool:
        """Check if module is a TSVD layer with required attributes."""
        required_attrs = ["alpha", "L", "R", "lr_scalars", "weight"]
        return all(hasattr(module, attr) for attr in required_attrs)

    def _compute_layer_ratio(self, module: torch.nn.Module) -> torch.Tensor:
        """Compute ternary vs low-rank ratio for a single layer."""
        with torch.no_grad():
            # Compute ternary part: |alpha * quantized_weights|
            if hasattr(module, "ternary_quantize"):
                q_weights, alpha, _, _ = module.ternary_quantize(module.weight, module.thresh_ratio)
            else:
                q_weights = torch.sign(module.weight)
                alpha = module.alpha
            
            ternary_part = (alpha * q_weights).abs().mean()
            
            # Compute low-rank part: |lr_scalars * L @ R|
            L = module.L.to(device=self.device, dtype=module.weight.dtype)
            R = module.R.to(device=self.device, dtype=module.weight.dtype)
            lr_scalars = module.lr_scalars.to(device=self.device, dtype=module.weight.dtype)
            
            if L.numel() > 0 and R.numel() > 0:
                lowrank_part = (lr_scalars * L @ R).abs().mean()
            else:
                lowrank_part = torch.tensor(1e-8, device=self.device)
            
            # Return ratio with numerical stability
            epsilon = 1e-8
            return ternary_part / (lowrank_part + epsilon)

    # Example usage for histogram logging:
    def log_stats(self):
        """
        Helper method to log ternary vs low-rank ratios with histogram.

        Args:
            logger: Your logging framework (e.g., WandbLogger from pytorch-lightning)
            step: Training step for logging
        """
        ratios = self.ternary_vs_lr_ratio()
        for key, value in ratios.items():
            if "value" not in key:
                self.log(f"ternary_vs_LR_ratio/{key}", value)
        return ratios

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
        pbar: bool=False
    ):
        """
        Generate images from text prompts using the trained model.
        
        Args:
            prompt: Text prompt(s) to generate images from. If None, performs unconditional generation
            height: Height of generated images (default: 512)
            width: Width of generated images (default: 512)  
            num_inference_steps: Number of denoising steps (default: 50)
            guidance_scale: Scale for classifier-free guidance (default: 7.5, ignored for unconditional)
            negative_prompt: Negative prompt(s) for guidance (optional, ignored for unconditional)
            generator: Random generator for reproducible results (optional)
            return_pil: Whether to return PIL images or tensor (default: True)
            batch_size: Number of images to generate (only used when prompt is None)
            
        Returns:
            List of PIL Images or tensor depending on return_pil parameter
        """
        # Ensure we're in eval mode
        self.eval()

        # Determine device: use GPU if available, else CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model components to the correct device if not already there
        self.to(device)
        if hasattr(self, "vae"):
            self.vae.to(device)
        if hasattr(self, "unet"):
            self.unet.to(device)
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(device)

        # Use inference scheduler if available, otherwise use training scheduler
        scheduler = self.inference_noise_scheduler if self.inference_noise_scheduler is not None else self.train_noise_scheduler

        # Determine if this is conditional or unconditional generation
        if prompt is None:
            # Unconditional generation
            is_conditional = False
            effective_batch_size = batch_size

            # Create unconditional embeddings (empty prompts)
            uncond_input = self.tokenizer(
                [""] * effective_batch_size, 
                padding="max_length", 
                max_length=self.tokenizer.model_max_length, 
                return_tensors="pt"
            )
            uncond_input = {k: v.to(device) for k, v in uncond_input.items()}
            text_embeddings = self.text_encoder(uncond_input["input_ids"])[0]
        else:
            # Conditional generation
            is_conditional = True

            # Handle single prompt
            if isinstance(prompt, str):
                prompt = [prompt]

            effective_batch_size = len(prompt)

            # Encode prompts
            text_input = self.tokenizer(
                prompt, 
                padding="max_length", 
                max_length=self.tokenizer.model_max_length, 
                truncation=True, 
                return_tensors="pt"
            )
            text_input = {k: v.to(device) for k, v in text_input.items()}
            text_embeddings = self.text_encoder(text_input["input_ids"])[0]

            # Handle negative prompts for classifier-free guidance
            if negative_prompt is not None:
                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt] * effective_batch_size
                elif len(negative_prompt) != effective_batch_size:
                    raise ValueError(f"Length of negative_prompt ({len(negative_prompt)}) must match batch_size ({effective_batch_size})")
                
                uncond_input = self.tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
            else:
                # Use empty prompts for unconditional part of CFG
                uncond_input = self.tokenizer(
                    [""] * effective_batch_size, 
                    padding="max_length", 
                    max_length=self.tokenizer.model_max_length, 
                    return_tensors="pt"
                )
            uncond_input = {k: v.to(device) for k, v in uncond_input.items()}
            uncond_embeddings = self.text_encoder(uncond_input["input_ids"])[0]

            # Concatenate for classifier-free guidance
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Create initial latent noise
        latents_shape = (effective_batch_size, self.unet.in_channels, height // 8, width // 8)
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=device,
            dtype=text_embeddings.dtype,
        )

        # Set timesteps
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        # Scale initial noise
        latents = latents * scheduler.init_noise_sigma

        # Denoising loop
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=not pbar):
            if is_conditional:
                # Conditional generation with classifier-free guidance
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = latent_model_input.to(device)
                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

                # Predict noise residual
                noise_pred = self.unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=text_embeddings
                ).sample

                # Perform classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            else:
                # Unconditional generation - no classifier-free guidance
                latent_model_input = latents.to(device)
                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

                # Predict noise residual
                noise_pred = self.unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=text_embeddings
                ).sample

            # Compute previous sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to images
        # Scale latents back to original scale
        latents = latents.to(device)
        if hasattr(self.vae.config, 'scaling_factor'):
            latents = latents / self.vae.config.scaling_factor
        else:
            latents = latents / 0.18215

        images = self.vae.decode(latents).sample

        # Post-process images
        images = (images / 2 + 0.5).clamp(0, 1)

        if return_pil:
            # Convert to PIL images
            images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).round().astype("uint8")
            from PIL import Image
            pil_images = [Image.fromarray(image) for image in images]
            return pil_images
        else:
            return images
def main():
    # Configuration
    
    # Initialize model
    model = StableDiffusionTrainingModule()
    model.apply_quantization("t")
    
    # Setup logger (optional)
    logger = WandbLogger(project="stable-diffusion-training")
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=40,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed",  # Use mixed precision for memory efficiency
        logger=logger,
        gradient_clip_val=1.0,
        limit_val_batches=50, 
        val_check_interval=500,
    )
    datamodule = LSUNBedroomDataModule(batch_size=16, num_workers=12, image_size=512)
    # Start training
    trainer.fit(model, datamodule)
    


if __name__ == "__main__":
    main()