"""
Download Stable Diffusion VAE from HuggingFace and save to local cache.

Usage:
    python downloads/download_diff.py
"""

import os
from dotenv import load_dotenv
from diffusers import AutoencoderKL


def download_unet_vae():
    """
    Download and cache Stable Diffusion VAE component.
    """
    load_dotenv()
    
    cache_dir = os.getenv("HF_TRANSFORMERS_CACHE", None)
    
    try:
        # Download VAE
        print("Downloading Stable Diffusion VAE...")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-ema", 
            cache_dir=cache_dir
        )
        print("VAE downloaded and cached.")
        print("Stable Diffusion VAE download complete.")
        
    except Exception as e:
        print(f"Error downloading Stable Diffusion VAE: {e}")
        raise


if __name__ == "__main__":
    download_unet_vae()
