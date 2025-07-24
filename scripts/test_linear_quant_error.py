import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from models import get_model
from quant import quantize
from quant.layers.nbitlinear import NBitLinear
from quant.layers.trqlinear import TRQLinear

# 1. Load the pretrained DiT-XL model using DiTPipeline
print("Loading original model...")
pipe = get_model("dit-xl-2-256")

layer = pipe.transformer.transformer_blocks[18].ff.net[2]

in_features = 8
out_features = 4
batch_size = 3

n_residuals_list = [1, 2, 3, 4, 5, 6]
trq_mean_diffs = []
trq_max_diffs = []
x = torch.randn(batch_size, in_features)
linear = nn.Linear(in_features, out_features, bias=True)

# Test TRQLinear with different numbers of residuals
for n_residuals in n_residuals_list:
    print(f"\nTesting TRQLinear with {n_residuals} residuals:")

    # Create layers
    trq = TRQLinear.from_linear(linear, n_residuals=n_residuals)

    # Forward pass
    out_linear = linear(x)
    out_trq = trq(x)

    # Calculate difference
    diff = (out_linear - out_trq).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    trq_mean_diffs.append(mean_diff)
    trq_max_diffs.append(max_diff)
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Max absolute difference: {max_diff:.6f}")

# Test NBitLinear with 4 bits
print(f"\nTesting NBitLinear with 4 bits:")
nbit_4 = NBitLinear.from_linear(linear, n_bits=4)

# Forward pass
out_linear = linear(x)
out_nbit_4 = nbit_4(x)

# Calculate difference
diff = (out_linear - out_nbit_4).abs()
nbit_4_mean_diff = diff.mean().item()
nbit_4_max_diff = diff.max().item()
print(f"Mean absolute difference: {nbit_4_mean_diff:.6f}")
print(f"Max absolute difference: {nbit_4_max_diff:.6f}")

# Test NBitLinear with 8 bits
print(f"\nTesting NBitLinear with 8 bits:")
nbit_8 = NBitLinear.from_linear(linear, n_bits=8)

# Forward pass
out_nbit_8 = nbit_8(x)

# Calculate difference
diff = (out_linear - out_nbit_8).abs()
nbit_8_mean_diff = diff.mean().item()
nbit_8_max_diff = diff.max().item()
print(f"Mean absolute difference: {nbit_8_mean_diff:.6f}")
print(f"Max absolute difference: {nbit_8_max_diff:.6f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(
    n_residuals_list,
    trq_mean_diffs,
    marker="o",
    label="TRQLinear Mean Absolute Difference",
    linewidth=2,
)
plt.plot(
    n_residuals_list,
    trq_max_diffs,
    marker="s",
    label="TRQLinear Max Absolute Difference",
    linewidth=2,
)
plt.axhline(
    y=nbit_4_mean_diff,
    color="red",
    linestyle="--",
    label=f"NBitLinear (4-bit) Mean: {nbit_4_mean_diff:.6f}",
    linewidth=2,
)
plt.axhline(
    y=nbit_4_max_diff,
    color="orange",
    linestyle="--",
    label=f"NBitLinear (4-bit) Max: {nbit_4_max_diff:.6f}",
    linewidth=2,
)
plt.axhline(
    y=nbit_8_mean_diff,
    color="green",
    linestyle="--",
    label=f"NBitLinear (8-bit) Mean: {nbit_8_mean_diff:.6f}",
    linewidth=2,
)
plt.axhline(
    y=nbit_8_max_diff,
    color="blue",
    linestyle="--",
    label=f"NBitLinear (8-bit) Max: {nbit_8_max_diff:.6f}",
    linewidth=2,
)
plt.xlabel("Number of Residuals")
plt.ylabel("Difference")
plt.title("Quantization Error Comparison: TRQLinear vs NBitLinear (4-bit and 8-bit)")
plt.xscale("log")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/linear_quant_error.png")
