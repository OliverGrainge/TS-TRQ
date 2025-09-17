"""
Download FFHQ-256 dataset from HuggingFace and save to local cache.

Usage:
    python downloads/download_ffhq.py
"""

import os
from dotenv import load_dotenv
load_dotenv()
from datasets import DownloadConfig, load_dataset
from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE


def download_ffhq():
    """
    Download and cache FFHQ-256 dataset splits.
    """
    
    # Optionally use a HuggingFace token from the environment
    hf_token = os.getenv("HF_TOKEN")
    
    # Optionally set a custom cache dir from the environment
    cache_dir = os.getenv("HF_DATASETS_CACHE", None)
    
    print("HF_HOME:", HF_HOME)
    print("HF_HUB_CACHE:", HF_HUB_CACHE)
    
    # NOTE: Do NOT pass use_auth_token to load_dataset, as this causes a TypeError in recent datasets versions.
    # Instead, set the HF_TOKEN environment variable, which datasets will use automatically.
    
    try:
        print("Downloading FFHQ-256 dataset...")
        
        # First, load the dataset to see what splits are available
        ds = load_dataset(
            "bitmind/ffhq-256",
            cache_dir=cache_dir,
            download_config=DownloadConfig(delete_extracted=True),
            trust_remote_code=True,
        )
        
        print(f"Available splits: {list(ds.keys())}")
        
        # Download each available split
        for split_name in ds.keys():
            print(f"Processing FFHQ-256 {split_name} split...")
            split_ds = ds[split_name]
            print(f"Split '{split_name}' contains {len(split_ds)} images")
            print(
                f"Downloaded split '{split_name}' to: {split_ds.cache_files[0]['filename'] if split_ds.cache_files else 'unknown location'}"
            )
        
        print("FFHQ-256 download complete.")
        print(f"Dataset info: {ds}")
        
        # Print total dataset size
        total_images = sum(len(ds[split]) for split in ds.keys())
        print(f"Total images downloaded: {total_images}")
        
    except Exception as e:
        print(f"Error downloading FFHQ-256: {e}")
        raise


if __name__ == "__main__":
    download_ffhq()