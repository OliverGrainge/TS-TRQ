"""
Download LSUN Bedroom dataset from HuggingFace and save to local cache.

Usage:
    python downloads/download_lsun_bedroom.py
"""

import os
from dotenv import load_dotenv
from datasets import DownloadConfig, load_dataset
from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE


def download_lsun_bedroom():
    """
    Download and cache LSUN Bedroom dataset.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Optionally use a HuggingFace token from the environment
    hf_token = os.getenv("HF_TOKEN")
    
    # Optionally set a custom cache dir from the environment
    cache_dir = os.getenv("HF_DATASETS_CACHE", None)
    
    print("HF_HOME:", HF_HOME)
    print("HF_HUB_CACHE:", HF_HUB_CACHE)
    
    # NOTE: Do NOT pass use_auth_token to load_dataset, as this causes a TypeError in recent datasets versions.
    # Instead, set the HF_TOKEN environment variable, which datasets will use automatically.
    
    try:
        print("Downloading LSUN Bedroom dataset...")
        ds = load_dataset(
            "pcuenq/lsun-bedrooms",
            cache_dir=cache_dir,
            download_config=DownloadConfig(delete_extracted=True),
            trust_remote_code=True,
        )
        
        print("LSUN Bedroom download complete.")
        print(f"Dataset info: {ds}")
        
    except Exception as e:
        print(f"Error downloading LSUN Bedroom: {e}")
        raise


if __name__ == "__main__":
    download_lsun_bedroom()
