from dotenv import load_dotenv

load_dotenv()

import os

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def download_unet_vae():
    """
    Download and cache all components for a given Stable Diffusion pipeline type.
    """
    cache_dir = os.getenv("HF_TRANSFORMERS_CACHE", None)
    # Download VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", cache_dir=cache_dir)
    print("VAE downloaded and cached.")



if __name__ == "__main__":
    download_unet_vae()
