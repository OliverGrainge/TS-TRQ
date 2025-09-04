from typing import List, Optional, Set

import torch
import torch.nn as nn

from .layers import QUANT_LAYERS


def get_all_linear_names(model: nn.Module) -> List[str]:
    """
    Recursively collect all fully-qualified names of nn.Linear layers in the model.

    Args:
        model (nn.Module): The model to search through

    Returns:
        List[str]: List of fully-qualified layer names (e.g., ['0.0', '0.1', '1'])
    """
    linear_names = []

    def _collect_recursive(module: nn.Module, prefix: str = ""):
        """Recursively traverse the model to find linear layers."""
        for name, child in module.named_children():
            # Build the full path name
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                linear_names.append(full_name)
            else:
                # Recursively search in child modules
                _collect_recursive(child, prefix=full_name)

    _collect_recursive(model)
    return linear_names


def get_all_conv2d_names(model: nn.Module) -> List[str]:
    """
    Recursively collect all fully-qualified names of nn.Conv2d layers in the model.

    Args:
        model (nn.Module): The model to search through

    Returns:
        List[str]: List of fully-qualified layer names (e.g., ['0.0', '0.1', '1'])
    """
    conv2d_names = []

    def _collect_recursive(module: nn.Module, prefix: str = ""):
        """Recursively traverse the model to find Conv2d layers."""
        for name, child in module.named_children():
            # Build the full path name
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Conv2d):
                conv2d_names.append(full_name)
            else:
                # Recursively search in child modules
                _collect_recursive(child, prefix=full_name)

    _collect_recursive(model)
    return conv2d_names


def quantize_model(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    ignore_layers: Optional[List[str]] = None,
    quant_type: str = "tlinear",
    **quant_kwargs,
) -> nn.Module:
    """
    Quantize all layers in layer_names with the specified quantization layer type.

    Args:
        model (nn.Module): The model to quantize
        layer_names (List[str], optional): Specific layers to quantize. If None, quantizes all layers of the appropriate type
        ignore_layers (List[str], optional): Layers to skip during quantization
        quant_type (str): Type of quantization layer to use (must be in QUANT_LAYERS)
        **quant_kwargs: Additional arguments passed to the quantization layer constructor

    Returns:
        nn.Module: The modified model with selected layers quantized

    Raises:
        ValueError: If quant_type is not supported
    """
    if quant_type not in QUANT_LAYERS:
        available_types = list(QUANT_LAYERS.keys())
        raise ValueError(
            f"Invalid quant_type: '{quant_type}'. Available types: {available_types}"
        )

    quant_layer_class = QUANT_LAYERS[quant_type]

    # Set default layer names if none provided
    if layer_names is None and "linear" in quant_type:
        layer_names = get_all_linear_names(model)
    elif layer_names is None and "conv2d" in quant_type:
        layer_names = get_all_conv2d_names(model)

    if ignore_layers is None:
        ignore_layers = []

    target_layers: Set[str] = set(layer_names)
    layers_to_ignore: Set[str] = set(ignore_layers)

    def _quantize_recursive(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if full_name in layers_to_ignore:
                continue

            # Quantize if this layer is in our target list
            if full_name in target_layers:
                # Determine which quantization method to use based on quant_type
                if "linear" in quant_type and isinstance(child, nn.Linear):
                    quantized_layer = quant_layer_class.from_linear(
                        child, **quant_kwargs
                    )
                    setattr(module, name, quantized_layer)
                elif "conv2d" in quant_type and isinstance(child, nn.Conv2d):
                    quantized_layer = quant_layer_class.from_conv2d(
                        child, **quant_kwargs
                    )
                    setattr(module, name, quantized_layer)
                else:
                    # If the type does not match, skip quantization for this layer
                    pass
            else:
                _quantize_recursive(child, full_name)

    _quantize_recursive(model)
    return model


# Alias for backward compatibility
quantize = quantize_model


if __name__ == "__main__":
    """
    Example usage and testing of the quantization functionality.
    """
    import copy

    # Create a simple test model with nested structure
    test_model = nn.Sequential(
        nn.Sequential(
            nn.Linear(10, 20, bias=True),
            nn.Linear(20, 10, bias=True),
        ),
        nn.Linear(10, 24, bias=True),
        nn.Linear(24, 10, bias=True),
    )

    print("=" * 50)
    print("MODEL QUANTIZATION DEMONSTRATION")
    print("=" * 50)

    print("\nModel before quantization:")
    print(test_model)

    # Get all linear layer names for reference
    all_linear_names = get_all_linear_names(test_model)
    print(f"\nAll linear layers found: {all_linear_names}")

    # Make a deep copy for comparison
    original_model = copy.deepcopy(test_model)

    # Quantize the model (ignore the first layer in the nested sequential)
    quantized_model = quantize_model(
        test_model,
        quant_type="tlinear",  # Use ternary linear quantization
        ignore_layers=["0.0"],  # Skip the first linear layer
    )

    print("\nModel after quantization:")
    print(quantized_model)

    # Test with sample input
    test_input = torch.randn(10, 10)

    # Get outputs from both models
    with torch.no_grad():
        original_output = original_model(test_input)
        quantized_output = quantized_model(test_input)

    # Compare outputs
    output_diff = (original_output - quantized_output).abs()
    print(f"\nOutput comparison:")
    print(f"Original output shape: {original_output.shape}")
    print(f"Quantized output shape: {quantized_output.shape}")
    print(f"Mean absolute difference: {output_diff.mean().item():.6f}")
    print(f"Max absolute difference: {output_diff.max().item():.6f}")

    print("\n" + "=" * 50)
    print("Quantization completed successfully!")
    print("=" * 50)
