#!/usr/bin/env python3
"""
Test script for diffusion model evaluation with FID computation.

Usage:
    python test.py runs/configs/test/cifar10-unet-diffusion/tsvd-reg[0.5]-rank[32]-A[fp].yaml
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import argparse
from datetime import datetime

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml

# Import clean_fid for FID computation
from cleanfid import fid

# Import project modules
from config_utils import Config
from train import get_module, get_datamodule


def get_device():
    """Get the best available device (GPU if available, otherwise CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Clear GPU cache to start fresh
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    return device


def clear_gpu_cache():
    """Clear GPU cache if CUDA is available"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_tensor_as_image(tensor: torch.Tensor, filepath: str) -> None:
    """Save a tensor as an image file, handling CIFAR-10 transforms properly"""
    # Move tensor to CPU if it's on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Handle CIFAR-10 normalization: images are normalized to [-1, 1] range
    # Convert from [-1, 1] to [0, 1] range
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    if tensor.dim() == 4:  # Batch of images
        tensor = tensor[0]  # Take first image
    
    # Convert from CHW to HWC
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    
    # Convert to numpy and then PIL
    array = tensor.numpy()
    array = (array * 255).astype(np.uint8)
    image = Image.fromarray(array)
    image.save(filepath)


def save_generated_images(model, num_images: int, output_dir: str, device: torch.device, batch_size: int = 32) -> None:
    """Generate and save images using the diffusion model"""
    print(f"Generating {num_images} images...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate images in batches
    num_batches = (num_images + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating images"):
        current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
        
        # Generate images
        generated_images = model.generate(
            batch_size=current_batch_size,
            return_pil=False,  # Return tensors
            num_inference_steps=500,
            use_ema=True,
            pbar=False
        )
        
        # Save each image in the batch
        for i in range(current_batch_size):
            image_idx = batch_idx * batch_size + i
            filepath = os.path.join(output_dir, f"{image_idx}.png")
            save_tensor_as_image(generated_images[i], filepath)
        
        # Clear GPU cache after each batch to manage memory
        if batch_idx % 10 == 0:  # Clear cache every 10 batches
            clear_gpu_cache()


def save_real_images(datamodule, num_images: int, output_dir: str, device: torch.device) -> None:
    """Save real images from the dataloader"""
    print(f"Extracting {num_images} real images from dataloader...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup datamodule
    datamodule.setup()
    dataloader = datamodule.val_dataloader()
    
    image_count = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting real images")):
        if image_count >= num_images:
            break
            
        # Get pixel values from batch and move to device if needed
        pixel_values = batch['pixel_values']
        if device.type == 'cuda' and not pixel_values.is_cuda:
            pixel_values = pixel_values.to(device)
        batch_size = pixel_values.shape[0]
        
        # Save each image in the batch
        for i in range(min(batch_size, num_images - image_count)):
            image_idx = image_count + i
            filepath = os.path.join(output_dir, f"{image_idx}.png")
            save_tensor_as_image(pixel_values[i], filepath)
        
        image_count += batch_size


def compute_fid_score(generated_dir: str, real_dir: str) -> float:
    """Compute FID score between generated and real images"""
    print("Computing FID score...")
    
    fid_score = fid.compute_fid(
        real_dir, 
        generated_dir,
        mode="clean"
    )
    return fid_score


def load_checkpoint(module, checkpoint_path): 
    sd = torch.load(checkpoint_path)
    weights = 'state_dict'
    if "ema_state_dict" in sd.keys():
        weights = "ema_state_dict"
    
    module.load_state_dict(sd[weights], strict=False)
    return module

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test script for diffusion model evaluation with FID computation")
    parser.add_argument("config", help="Path to configuration YAML file")
    parser.add_argument("--num-images", type=int, default=1000, help="Number of images to generate (default: 1000)")
    
    args = parser.parse_args()
    config_path = args.config
    
    # Get device (GPU if available, otherwise CPU)
    device = get_device()
    
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config_dict = load_config(config_path)
    config = Config(config_dict)
    
    # Extract configuration sections
    module_config = getattr(config, "module_config", {})
    datamodule_config = getattr(config, "datamodule_config", {})
    quantization_config = getattr(config, "quantization_config", {})
    
    # Get number of images to generate (CLI argument takes precedence)
    num_images = args.num_images
    print(f"Number of images to generate: {num_images}")
    
    # Load the diffusion model
    print("Loading diffusion model...")
    module = get_module(module_config)
    
    
    # Apply quantization if specified
    quant_type = quantization_config.pop("quant_type", None)
    if quant_type is not None:
        print(f"Applying quantization: {quant_type}")
        module.apply_quantization(quant_type=quant_type, **quantization_config)


    module = load_checkpoint(module, config.checkpoint)
    
    # Move model to device
    print(f"Moving model to device: {device}")
    module = module.to(device)
    
    # Show GPU memory usage if using GPU
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory after model loading - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
    
    # Set model to evaluation mode
    module.eval()
    
    # Load the data module
    print("Loading data module...")
    datamodule = get_datamodule(datamodule_config)
    
    # Create simple output directories
    generated_dir = Path("tmp/generated")
    real_dir = Path("tmp/real")
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    
    
    print(f"Generated images will be saved to: {generated_dir}")
    print(f"Real images will be saved to: {real_dir}")
    
    # Generate images
    save_generated_images(module, num_images, str(generated_dir), device)
    clear_gpu_cache()  # Clear GPU cache after generation
    
    # Save real images
    save_real_images(datamodule, num_images, str(real_dir), device)
    clear_gpu_cache()  # Clear GPU cache after processing real images
    
    # Compute FID score
    print("Computing FID score...")
    fid_score = compute_fid_score(str(generated_dir), str(real_dir))
    
    if fid_score >= 0:
        print(f"\n{'='*50}")
        print(f"FID Score: {fid_score:.4f}")
        print(f"{'='*50}")
    else:
        print("Failed to compute FID score")
    
    print(f"\nImages saved to: {Path.cwd()}")
    print(f"Generated images: {generated_dir.absolute()}")
    print(f"Real images: {real_dir.absolute()}")


if __name__ == "__main__":
    main()
