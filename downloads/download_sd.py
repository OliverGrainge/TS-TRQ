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

def download_sd_models(pipe_type="segmind/tiny-sd"):
    print(f"Downloading {pipe_type} components from '{pipe_type}' ...")
    vae = AutoencoderKL.from_pretrained(pipe_type, subfolder="vae")
    print("VAE downloaded and cached.")
    text_encoder = CLIPTextModel.from_pretrained(pipe_type, subfolder="text_encoder")
    print("Text encoder downloaded and cached.")
    tokenizer = CLIPTokenizer.from_pretrained(pipe_type, subfolder="tokenizer")
    print("Tokenizer downloaded and cached.")
    unet = UNet2DConditionModel.from_pretrained(pipe_type, subfolder="unet")
    print("UNet downloaded and cached.")
    noise_scheduler = DDPMScheduler.from_pretrained(pipe_type, subfolder="scheduler")
    print("Noise scheduler downloaded and cached.")
    print(f"All {pipe_type} components downloaded.")



if __name__ == "__main__":
    sdv15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    tinysd = "segmind/tiny-sd"
    download_sd_models(sdv15)
    download_sd_models(tinysd)