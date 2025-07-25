import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optionally use a HuggingFace token from the environment
hf_token = os.getenv("HF_TOKEN")

# Optionally set a custom cache dir from the environment
cache_dir = os.getenv("HF_DATASETS_CACHE", None)


from datasets import DownloadConfig, load_dataset
from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE

print("HF_HOME:", HF_HOME)
print("HF_HUB_CACHE:", HF_HUB_CACHE)


# NOTE: Do NOT pass use_auth_token to load_dataset, as this causes a TypeError in recent datasets versions.
# Instead, set the HF_TOKEN environment variable, which datasets will use automatically.

# Enable trust_remote_code for loading datasets with custom code
for split in ["train", "test"]:
    ds = load_dataset(
        "cifar100",
        split=split,
        cache_dir=cache_dir,
        download_config=DownloadConfig(delete_extracted=True),
        trust_remote_code=True,
    )
    print(
        f"Downloaded split '{split}' to: {ds.cache_files[0]['filename'] if ds.cache_files else 'unknown location'}"
    )

print("Download complete.")
