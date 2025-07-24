import os
from datasets import load_dataset, DownloadConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optionally use a HuggingFace token from the environment
hf_token = os.getenv("HF_TOKEN")

# Optionally set a custom cache dir from the environment
cache_dir = os.getenv("HF_DATASETS_CACHE", None)

print("Downloading ImageNet (ILSVRC/imagenet-1k) from HuggingFace...")

# NOTE: Do NOT pass use_auth_token to load_dataset, as this causes a TypeError in recent datasets versions.
# Instead, set the HF_TOKEN environment variable, which datasets will use automatically.

# Download both splits to ensure full download
for split in ["train", "validation"]:
    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split=split,
        cache_dir=cache_dir,
        download_config=DownloadConfig(delete_extracted=True)
    )
    print(f"Downloaded split '{split}' to: {ds.cache_files[0]['filename'] if ds.cache_files else 'unknown location'}")

print("Download complete.")
