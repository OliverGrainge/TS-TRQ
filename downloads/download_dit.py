import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

from diffusers import DiTPipeline
from huggingface_hub import hf_hub_download

# Optionally use a HuggingFace token from the environment
hf_token = os.getenv("HF_TOKEN")

# Optionally set a custom cache dir from the environment
cache_dir = os.getenv("HF_HUB_CACHE", None)

print("Downloading DiT-XL-2-256 model from HuggingFace...")

# Print out where the model will be downloaded before starting
print(f"Model will be downloaded to the custom cache directory: {cache_dir}")

# Download the model weights and config
pipe = DiTPipeline.from_pretrained(
    "facebook/DiT-XL-2-256",
    torch_dtype=None,  # Don't load to GPU or set dtype
    use_auth_token=hf_token,
    cache_dir=cache_dir
)

print(f"Model downloaded to: {pipe.config._name_or_path if hasattr(pipe.config, '_name_or_path') else 'HuggingFace cache directory'}")
print("Download complete.") 