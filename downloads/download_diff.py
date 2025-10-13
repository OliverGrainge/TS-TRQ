"""
Download Stable Diffusion VAE and DDPM models from HuggingFace and save to local cache.

Usage:
    python downloads/download_diff.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

from diffusers import AutoencoderKL, DDPMPipeline


def download_unet_vae():
    """
    Download and cache Stable Diffusion VAE component.
    """

    cache_dir = os.getenv("HF_TRANSFORMERS_CACHE", None)

    try:
        # Download VAE
        print(f"Downloading Stable Diffusion VAE to {cache_dir}")
        print("Downloading Stable Diffusion VAE...")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-ema", cache_dir=cache_dir
        )
        print("VAE downloaded and cached.")
        print("Stable Diffusion VAE download complete.")

    except Exception as e:
        print(f"Error downloading Stable Diffusion VAE: {e}")
        raise


def download_ddpm_models():
    """
    Download and cache DDPM models.
    """

    cache_dir = os.getenv("HF_TRANSFORMERS_CACHE", None)

    model_ids = [
        "google/ddpm-ema-church-256",
        "google/ddpm-ema-bedroom-256",
        "google/ddpm-cifar10-32",
    ]

    for model_id in model_ids:
        try:
            print(f"Downloading {model_id} to {cache_dir}")
            print(f"Downloading {model_id}...")
            pipeline = DDPMPipeline.from_pretrained(model_id, cache_dir=cache_dir)
            print(f"{model_id} downloaded and cached.")

        except Exception as e:
            print(f"Error downloading {model_id}: {e}")
            raise

    print("All DDPM models download complete.")


if __name__ == "__main__":
    download_unet_vae()
    download_ddpm_models()
