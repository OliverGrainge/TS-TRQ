from dotenv import load_dotenv

load_dotenv()

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
)

import os

def download_sd():
    """
    Download and cache all components for a given Stable Diffusion pipeline type.
    """
    cache_dir = os.getenv("HF_TRANSFORMERS_CACHE", None)
    # Download VAE
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", cache_dir=cache_dir) 
    print("VAE downloaded and cached.")
    # Download Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
    print("Tokenizer downloaded and cached.")
    # Download Text Encoder
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
    print("Text encoder downloaded and cached.")
    # Download UNet
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", cache_dir=cache_dir)
    print("UNet downloaded and cached.")
    # Download Scheduler
    scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000
    )
    print("Noise scheduler downloaded and cached.")



if __name__ == "__main__":
    download_sd()