"""
Download ResNet-18 and ResNet-50 weights from HuggingFace and save to local cache.

Usage:
    python downloads/download_resnet.py
"""

import os

from dotenv import load_dotenv
from transformers import ResNetForImageClassification


def download_resnet(model_name):
    """
    Download and cache a ResNet model.

    Args:
        model_name (str): Name of the ResNet model to download
    """
    try:
        print(f"Downloading {model_name}...")
        cache_dir = os.getenv("HF_TRANSFORMERS_CACHE", None)
        model = ResNetForImageClassification.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        print(f"Downloaded {model_name} and saved to cache.")

    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        raise


def download_all_resnets():
    """
    Download and cache all ResNet models.
    """
    load_dotenv()

    model_names = ["microsoft/resnet-18", "microsoft/resnet-50"]

    for model_name in model_names:
        download_resnet(model_name)

    print("All ResNet models download complete.")


if __name__ == "__main__":
    download_all_resnets()
