"""
Download CIFAR-100 dataset from HuggingFace and save to local cache.

Usage:
    python downloads/download_cifar100.py
"""

import os

from datasets import DownloadConfig, load_dataset
from dotenv import load_dotenv
from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE


def download_cifar100():
    """
    Download and cache CIFAR-100 dataset splits.
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
        for split in ["train", "test"]:
            print(f"Downloading CIFAR-100 {split} split...")
            ds = load_dataset(
                "cifar100",
                split=split,
                cache_dir=cache_dir,
                download_config=DownloadConfig(delete_extracted=True),
            )
            print(
                f"Downloaded split '{split}' to: {ds.cache_files[0]['filename'] if ds.cache_files else 'unknown location'}"
            )

        print("CIFAR-100 download complete.")
        print(f"Dataset info: {ds}")

    except Exception as e:
        print(f"Error downloading CIFAR-100: {e}")
        raise


if __name__ == "__main__":
    download_cifar100()
