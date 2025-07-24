import torch
import torch.nn as nn

from .layers import QUANT_LAYERS


def linear_layer_names(model: nn.Module) -> list:
    linear_names = []

    def _collect(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                linear_names.append(full_name)
            else:
                _collect(child, prefix=full_name)

    _collect(model)
    return linear_names


def quantize(
    model: nn.Module,
    layer_names: list = None,
    ignore_layers: list = None,
    quant_type: str = "trqlinear",
    **kwargs,
) -> nn.Module:
    """
    Quantizes selected nn.Linear layers in the model using TRQLinear.

    Args:
        model (nn.Module): The model to modify.
        layer_names (list, optional): Fully-qualified names of layers to quantize.
                                      If None, all nn.Linear layers will be quantized.
        ignore_layers (list, optional): Fully-qualified names of layers to skip.
        n_residuals (int): Number of ternary residuals to use.

    Returns:
        nn.Module: Modified model with selected layers quantized.
    """
    assert quant_type in QUANT_LAYERS, f"Invalid quant_type: {quant_type}"
    quant_layer = QUANT_LAYERS[quant_type]

    if layer_names is None:
        layer_names = linear_layer_names(model)

    if ignore_layers is None:
        ignore_layers = []

    # Convert to set for faster lookup
    layer_names = set(layer_names)
    ignore_layers = set(ignore_layers)

    def _quantize(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                if full_name in ignore_layers:
                    continue
                if full_name in layer_names:
                    trq = quant_layer.from_linear(child, **kwargs)
                    setattr(module, name, trq)
            else:
                _quantize(child, full_name)

    _quantize(model)
    return model


if __name__ == "__main__":
    import copy

    model = nn.Sequential(
        nn.Sequential(
            nn.Linear(10, 20, bias=True),
            nn.Linear(20, 10, bias=True),
        ),
        nn.Linear(10, 24, bias=True),
        nn.Linear(24, 10, bias=True),
    )

    print("=" * 40)
    print("Model before quantization:")
    print(model)
    print("=" * 40)

    # Make a deep copy so we can compare before/after quantization
    model_before = copy.deepcopy(model)
    quant_model = quantize(
        model, quant_type="trqlinear", n_residuals=10, ignore_layers=["1"]
    )

    print("Model after quantization:")
    print(quant_model)
    print("=" * 40)

    x = torch.randn(10, 10)

    out_before = model_before(x)
    out_after = quant_model(x)

    # No need to print the tensors
    diff = (out_before - out_after).abs()
    print(f"Mean absolute difference: {diff.mean().item():.6f}")
    print(f"Max absolute difference: {diff.max().item():.6f}")
    print("=" * 40)
