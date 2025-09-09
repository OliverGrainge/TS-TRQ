"""
Download ImageNet-1K dataset from HuggingFace and save to local cache.

Usage:
    python downloads/download_imagenet.py
"""

import os
from dotenv import load_dotenv, find_dotenv
dotenv = find_dotenv()
print(f"Dotenv file: {dotenv}")
load_dotenv(find_dotenv())


from datasets import DownloadConfig, load_dataset
from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE


def download_imagenet():
    """
    Download and cache ImageNet-1K dataset splits.
    """
    # Load environment variables from .env file
    
    
    # Optionally use a HuggingFace token from the environment
    hf_token = os.getenv("HF_TOKEN")
    
    # Optionally set a custom cache dir from the environment
    cache_dir = os.getenv("HF_DATASETS_CACHE", None)
    
    print("HF_HOME:", HF_HOME)
    print("HF_HUB_CACHE:", HF_HUB_CACHE)
    
    # NOTE: Do NOT pass use_auth_token to load_dataset, as this causes a TypeError in recent datasets versions.
    # Instead, set the HF_TOKEN environment variable, which datasets will use automatically.
    
    try:
        for split in ["train", "validation"]:
            print(f"Downloading ImageNet-1K {split} split...")
            ds = load_dataset(
                "timm/imagenet-1k-wds",  # Official repository name
                split=split,
                cache_dir=cache_dir,
                download_config=DownloadConfig(delete_extracted=True),
                token=hf_token,  # Use token parameter instead of trust_remote_code
                streaming=False,  # Set to True if you want streaming mode
            )
            print(
                f"Downloaded split '{split}' to: {ds.cache_files[0]['filename'] if ds.cache_files else 'unknown location'}"
            )
        
        print("ImageNet-1K download complete.")
        print(f"Dataset info: {ds}")
        
    except Exception as e:
        print(f"Error downloading ImageNet-1K: {e}")
        raise


if __name__ == "__main__":
    
    download_imagenet()
