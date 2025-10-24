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
    """Save a tensor as an image file"""
    # Move tensor to CPU if it's on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Ensure tensor is in [0, 1] range and convert to PIL
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    
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
            use_ema=True,
            pbar=False
        )
        
        # Save each image in the batch
        for i in range(current_batch_size):
            image_idx = batch_idx * batch_size + i
            filepath = os.path.join(output_dir, f"generated_{image_idx:06d}.png")
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
            filepath = os.path.join(output_dir, f"real_{image_idx:06d}.png")
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



def main():
    """Main test function"""
    if len(sys.argv) != 2:
        print("Usage: python test.py <config.yaml>")
        print("Example: python test.py runs/configs/test/cifar10-unet-diffusion/tsvd-reg[0.5]-rank[32]-A[fp].yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
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
    
    # Get number of images to generate
    num_images = getattr(config, "num_images", 1000)
    print(f"Number of images to generate: {num_images}")
    
    # Load the diffusion model
    print("Loading diffusion model...")
    module = get_module(module_config)
    
    # Apply quantization if specified
    quant_type = quantization_config.pop("quant_type", None)
    if quant_type is not None:
        print(f"Applying quantization: {quant_type}")
        module.apply_quantization(quant_type=quant_type, **quantization_config)
    
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
    
    # Create temporary directory for images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        generated_dir = temp_path / "generated"
        real_dir = temp_path / "real"
        
        print(f"Using temporary directory: {temp_path}")
        
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
        
        # Optionally save images to permanent location
        save_permanent = input("\nSave images to permanent location? (y/n): ").lower().strip() == 'y'
        if save_permanent:
            permanent_dir = Path("test_output")
            permanent_dir.mkdir(exist_ok=True)
            
            # Copy generated images
            generated_permanent = permanent_dir / "generated"
            shutil.copytree(generated_dir, generated_permanent, dirs_exist_ok=True)
            print(f"Generated images saved to: {generated_permanent}")
            
            # Copy real images
            real_permanent = permanent_dir / "real"
            shutil.copytree(real_dir, real_permanent, dirs_exist_ok=True)
            print(f"Real images saved to: {real_permanent}")


if __name__ == "__main__":
    main()
