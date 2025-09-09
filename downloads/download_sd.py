"""
Download Stable Diffusion components from HuggingFace and save to local cache.

Usage:
    python downloads/download_sd.py
"""

import os
from dotenv import load_dotenv
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def download_sd():
    """
    Download and cache all components for Stable Diffusion pipeline.
    """
    load_dotenv()
    
    cache_dir = os.getenv("HF_TRANSFORMERS_CACHE", None)
    
    try:
        # Download VAE
        print("Downloading VAE...")
        vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae", cache_dir=cache_dir
        )
        print("VAE downloaded and cached.")
        
        # Download Tokenizer
        print("Downloading Tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14", cache_dir=cache_dir
        )
        print("Tokenizer downloaded and cached.")
        
        # Download Text Encoder
        print("Downloading Text Encoder...")
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14", cache_dir=cache_dir
        )
        print("Text encoder downloaded and cached.")
        
        # Download UNet
        print("Downloading UNet...")
        unet = UNet2DConditionModel.from_pretrained(
            "nota-ai/bk-sdm-tiny-2m", subfolder="unet", cache_dir=cache_dir
        )
        print("UNet downloaded and cached.")
        
        # Download Scheduler
        print("Downloading Scheduler...")
        scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        print("Noise scheduler downloaded and cached.")
        
        print("Stable Diffusion components download complete.")
        
    except Exception as e:
        print(f"Error downloading Stable Diffusion components: {e}")
        raise


if __name__ == "__main__":
    download_sd()
