import os

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from quant import linear_layer_names, quantize
from .helper import get_model
from diffusers import DDPMScheduler
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch_ema import ExponentialMovingAverage
from PIL import Image
import numpy as np

class DiTModule(pl.LightningModule):
    def __init__(self, model_name="dit-xl-2-256", learning_rate=1e-7,
                 quant_type=None, quant_args=None, reg_scale=0.2, p_uncond=0.1):
        super().__init__()
        if quant_args is None: quant_args = {}
        self.save_hyperparameters()

        model = get_model(model_name)  # your helper
        self.vae = model.vae
        self.transformer = model.transformer
        self.pipeline = model

        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=1e-4, beta_end=2e-2,
            beta_schedule="linear",
            prediction_type="epsilon",
            clip_sample=False
        )

        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        self.learning_rate = learning_rate
        self.reg_scale = reg_scale
        self.p_uncond = p_uncond

        #if quant_type is not None:
        #    self._quantize_transformer(quant_type, quant_args)


    def _quantize_transformer(self, quant_type, quant_args):
        qs = self._get_quant_layer_names()
        self.transformer = quantize(self.transformer, layer_names=qs,
                                    quant_type=quant_type, **quant_args)

    def _get_quant_layer_names(self):
        return [n for n, _ in self.transformer.named_modules() if "ff" in n]

    def forward(self, x, t, y=None):
        out = self.transformer(x, t, class_labels=y).sample
        epsilon, logvar = out.chunk(2, dim=1)
        return epsilon, logvar 

    def encode(self, images):
        images = images.to(dtype=self.vae.dtype, device=self.device)
        return self.vae.encode(images).latent_dist.sample() * self.scaling_factor

    def decode(self, latents):
        latents = (latents / self.scaling_factor).to(dtype=self.vae.dtype, device=self.device)
        return self.vae.decode(latents).sample

    @torch.no_grad()
    def sample(self, batch_size=1, class_labels=None, num_steps=50, device=None, output_type="tensor"):
        generator = torch.manual_seed(45)
        if class_labels is None:
            class_labels = torch.randint(0, 1000, (batch_size,), generator=generator).tolist()
        if device is None:
            device = next(self.parameters()).device

        out = self.pipeline(class_labels=class_labels,
                            num_inference_steps=num_steps,
                            generator=generator,
                            guidance_scale=4.0)
        imgs = out.images
        if output_type == "tensor":
            if isinstance(imgs, list):
                return torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in imgs])
            return imgs
        elif output_type == "pil":
            if isinstance(imgs, list):
                return imgs
            return [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in imgs]
        else:
            raise ValueError(output_type)

    def reg_loss(self, reduction="mean"):
        losses = []
        for m in self.transformer.modules():
            fn = getattr(m, "layer_reg_loss", None)
            if callable(fn):
                losses.append(fn())
        if not losses:
            return torch.tensor(0.0, device=self.device)
        losses = torch.stack([torch.as_tensor(l, device=self.device) for l in losses])
        return losses.mean() if reduction == "mean" else losses.sum()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        
        latents = self.encode(images)
        
        noise = torch.randn_like(latents)

        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.size(0),),
                          device=latents.device, dtype=torch.long)
        
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        # CFG drop
        #drop = torch.rand(latents.size(0), device=latents.device) < self.p_uncond
        cond_labels = labels.clone()
        #cond_labels[drop] = None

        pred, logvar = self(noisy_latents, t, y=cond_labels)
        print(pred.isnan().any())
        loss = F.mse_loss(pred, noise, reduction="mean")
        #print(loss)
        rloss = self.reg_scale * self.reg_loss()
        total = loss

        self.log_dict({"train_loss": loss, "reg_loss": rloss, "total_loss": total},
                      on_step=True, prog_bar=True)
        return total

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        latents = self.encode(images)
        noise = torch.randn_like(latents)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.size(0),),
                          device=latents.device, dtype=torch.long)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        pred, logvar = self(noisy_latents, t, y=labels)
        loss = F.mse_loss(pred, noise, reduction="mean")
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.transformer.parameters(),
                                lr=self.learning_rate,
                                weight_decay=0.0,
                                betas=(0.9, 0.999),
                                eps=1e-8)
        return opt



if __name__ == "__main__":
    import os
    os.makedirs("tmp", exist_ok=True)
    dit = DiTModule() 
    images = dit.sample(batch_size=2, output_type="pil")
    for idx, image in enumerate(images):
        image.save(f"tmp/image_{idx}.png")
