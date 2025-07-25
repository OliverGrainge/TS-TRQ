import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from huggingface_hub import snapshot_download

# Optionally use a HuggingFace token from the environment
hf_token = os.getenv("HF_TOKEN")

# Optionally set a custom cache dir from the environment
cache_dir = os.getenv("HF_HUB_CACHE", None)

print("Downloading ViT-Base-Patch16-224 model from HuggingFace...")

# Print out where the model will be downloaded before starting
print(f"Model will be downloaded to the custom cache directory: {cache_dir}")

# Download all model files as a snapshot
local_dir = snapshot_download(
    repo_id="google/vit-base-patch16-224", cache_dir=cache_dir, token=hf_token
)

print(f"Model downloaded to: {local_dir}")
print("Download complete.")
