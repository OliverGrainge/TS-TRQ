"""
Download ResNet-18 and ResNet-50 weights from HuggingFace and save to local cache.

Usage:
    python downloads/download_resnet.py
"""

from transformers import ResNetForImageClassification

def download_resnet(model_name):
    print(f"Downloading {model_name} ...")
    model = ResNetForImageClassification.from_pretrained(model_name)
    print(f"Downloaded {model_name} and saved to cache.")

if __name__ == "__main__":
    for model_name in ["microsoft/resnet-18", "microsoft/resnet-50"]:
        download_resnet(model_name)