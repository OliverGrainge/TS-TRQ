import torch
from dotenv import load_dotenv

load_dotenv()

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from typing import Union

from diffusers import (
    UNet2DConditionModel, 
    DDPMScheduler, 
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import torchvision.transforms as transforms
import os
from typing import Optional, List, Tuple
from data import LSUNBedroomDataModule
from diffusers import DiffusionPipeline



def load_stable_diffusion(): 
    cache_dir = os.getenv("HF_TRANSFORMERS_CACHE")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", cache_dir=cache_dir) 
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", cache_dir=cache_dir)
    noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
    return {"vae": vae, "text_encoder": text_encoder, "tokenizer": tokenizer, "unet": unet, "noise_scheduler": noise_scheduler}



class StableDiffusionTrainingModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 500,
        max_train_steps: int = 10000,
        gradient_checkpointing: bool = True,
        train_text_encoder: bool = False,
        resolution: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.max_train_steps = max_train_steps
        self.train_text_encoder = train_text_encoder
        
        # Load the models
        model_dict = load_stable_diffusion()
        self.vae = model_dict["vae"]
        self.text_encoder = model_dict["text_encoder"]
        self.tokenizer = model_dict["tokenizer"]
        self.unet = model_dict["unet"]
        
        # Scheduler for training
        self.noise_scheduler = model_dict["noise_scheduler"]
        
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
            assert latents.shape[1] == self.unet.in_channels
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=latents.device
        ).long()
        
        # Add noise to latents according to noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
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
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        if 'caption' not in batch.keys(): 
            captions = None
        else:
            captions = batch['caption']
        
        
        model_pred, target = self.forward(pixel_values, captions)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
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

def main():
    # Configuration
    
    # Initialize model
    model = StableDiffusionTrainingModule()

    
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
    )
    datamodule = LSUNBedroomDataModule(batch_size=16, num_workers=12, image_size=512)
    # Start training
    trainer.fit(model, datamodule)
    


if __name__ == "__main__":
    main()