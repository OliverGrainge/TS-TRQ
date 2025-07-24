import os

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from quant import linear_layer_names, quantize
from .helper import get_model


class DiTModule(pl.LightningModule):
    def __init__(
        self, model_name="dit-xl-2-256", learning_rate=1e-4, weight_decay=1e-4, quant_type=None, quant_args={}, reg_scale=1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load model components explicitly
        model = get_model(model_name)
        self.vae = model.vae.float()
        self.transformer = model.transformer.float()
        self.scheduler = model.scheduler
        self.pipeline = model  # Keep a reference if you need the pipeline itself

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.reg_scale = reg_scale
        self._quantize_transformer(quant_type, quant_args)


    def _quantize_transformer(self, quant_type: str, quant_args: dict): 
        if quant_type is not None:
            quant_layer_names = self._get_quant_layer_names()
            self.transformer = quantize(self.transformer, layer_names=quant_layer_names, quant_type=quant_type, **quant_args)

    def _get_quant_layer_names(self): 
        layer_names = []
        for name, module in self.transformer.named_modules(): 
            if "ff" in name:
                layer_names.append(name)
        return layer_names

    def forward(self, x, t, y=None):
        output = self.transformer(x, t, class_labels=y).sample
        noise_pred, var_pred = output.chunk(2, dim=1)
        return noise_pred, var_pred

    def encode(self, images):
        dtype = next(self.vae.parameters()).dtype
        images = images.to(dtype=dtype, device=self.device)
        latents = self.vae.encode(images).latent_dist.sample() * 0.18215
        return latents

    def decode(self, latents):
        scaled_latents = latents / 0.18215
        dtype = next(self.vae.parameters()).dtype
        scaled_latents = scaled_latents.to(dtype=dtype, device=self.device)
        images = self.vae.decode(scaled_latents).sample
        return images

    @torch.no_grad()
    def sample(self, batch_size=1, class_labels=None, num_steps=50, device=None, output_type="tensor"):
        generator = torch.manual_seed(45)

        if class_labels is None:
            class_labels = torch.randint(0, 1000, (batch_size,), generator=generator).tolist()

        if device is None:
            device = next(self.parameters()).device

        output = self.pipeline(
            class_labels=class_labels,
            num_inference_steps=num_steps,
            generator=generator,
            guidance_scale=4.0,
        )
        images = output.images

        if output_type == "tensor":
            if isinstance(images, list):
                images = torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in images])
            return images
        elif output_type == "pil":
            if isinstance(images, list):
                return images
            elif isinstance(images, torch.Tensor):
                images = images.cpu()
                images = [Image.fromarray(img.permute(1, 2, 0).numpy()) for img in images]
                return images
            else:
                raise ValueError("Unknown image output type from pipeline. Must either be 'tensor' or 'pil'.")
        else:
            raise ValueError(f"Invalid output type: {output_type}")
        
    def reg_loss(self, reduction="mean"):
        reg_losses = []
        for module in self.transformer.modules():
            reg_fn = getattr(module, "layer_reg_loss", None)
            if callable(reg_fn):
                reg_losses.append(reg_fn())
        if not reg_losses:
            return 0.0
        reg_losses = torch.stack([torch.as_tensor(l) for l in reg_losses])
        if reduction == "mean":
            return reg_losses.mean()
        elif reduction == "sum":
            return reg_losses.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def training_step(self, batch, batch_idx):
        images, class_labels = batch
        latents = self.encode(images)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (images.size(0),), device=images.device
        ).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        noise_pred, var_pred = self(noisy_latents, timesteps, y=class_labels)
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        reg_loss = self.reg_loss()
        print("=="*10)
        print(reg_loss)
        print("=="*10)
        print(loss)
        print("=="*10)

        # Log training metrics
        self.log("train_loss", loss, on_step=True)
        self.log("reg_loss", reg_loss, on_step=True)

        total_loss = loss + reg_loss
        self.log("total_loss", total_loss, on_step=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, class_labels = batch
        latents = self.encode(images)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (images.size(0),), device=images.device
        ).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        noise_pred, var_pred = self(noisy_latents, timesteps, y=class_labels)
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        # Log validation metrics
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.transformer.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,  # Number of epochs
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }



if __name__ == "__main__":
    import os
    os.makedirs("tmp", exist_ok=True)
    dit = DiTModule() 
    images = dit.sample(batch_size=2, output_type="pil")
    for idx, image in enumerate(images):
        image.save(f"tmp/image_{idx}.png")
