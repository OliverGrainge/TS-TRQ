import torch
import torch.nn as nn


class NBitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, n_bits=8):
        super().__init__(in_features, out_features, bias)
        self.n_bits = n_bits

    @classmethod
    def from_linear(cls, linear: nn.Linear, n_bits=8):
        """Create an NBitLinear layer from an existing nn.Linear layer"""
        nbit_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            n_bits=n_bits,
        )

        # Copy weights and bias
        nbit_linear.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            nbit_linear.bias.data = linear.bias.data.clone()

        return nbit_linear

    def quantize_weights(self, w: nn.Parameter):
        dtype = w.dtype
        """Perform symmetric weight quantization"""
        # Find the maximum absolute value for symmetric quantization
        max_val = torch.max(torch.abs(w))

        # Avoid division by zero
        if max_val == 0:
            return w

        # Define quantization range for symmetric quantization
        # For n_bits: [-2^(n_bits-1), 2^(n_bits-1)-1]
        qmin = -(2 ** (self.n_bits - 1))
        qmax = 2 ** (self.n_bits - 1) - 1

        # Calculate scale factor
        scale = max_val / qmax

        # Quantize: divide by scale and round
        w_quantized = torch.round(w / scale)

        # Clamp to valid quantization range
        w_quantized = torch.clamp(w_quantized, qmin, qmax)

        # Dequantize: multiply by scale to get back to original range
        w_dequantized = w_quantized * scale

        return w_dequantized.to(dtype)

    def forward(self, x: torch.Tensor):
        """Forward pass using quantized weights"""
        # Quantize weights during forward pass (simulation)
        quantized_weight = self.quantize_weights(self.weight)

        # Perform linear transformation with quantized weights
        return nn.functional.linear(x, quantized_weight, self.bias)

    def __repr__(self):
        return f"NBitLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, n_bits={self.n_bits})"


# Example usage:
if __name__ == "__main__":
    # Create a regular linear layer
    linear = nn.Linear(128, 64)

    # Test input
    x = torch.randn(32, 128)
    original_output = linear(x)

    print("Quantization Error Analysis")
    print("=" * 50)
    print(
        f"Original weight range: [{linear.weight.min():.4f}, {linear.weight.max():.4f}]"
    )
    print(f"Original weight std: {linear.weight.std():.4f}")
    print()

    # Test different bit counts
    bit_counts = [1, 2, 3, 4, 6, 8, 16, 32]

    for n_bits in bit_counts:
        # Convert to quantized version
        qlinear = NBitLinear.from_linear(linear, n_bits=n_bits)

        # Get quantized weights
        quantized_weights = qlinear.quantize_weights(qlinear.weight)

        # Calculate weight quantization error
        weight_error = torch.abs(linear.weight - quantized_weights)
        mean_weight_error = weight_error.mean().item()
        max_weight_error = weight_error.max().item()

        # Calculate output error
        quantized_output = qlinear(x)
        output_error = torch.abs(original_output - quantized_output)
        mean_output_error = output_error.mean().item()
        max_output_error = output_error.max().item()

        # Calculate Signal-to-Quantization-Noise Ratio (SQNR)
        signal_power = torch.mean(linear.weight**2)
        noise_power = torch.mean(weight_error**2)
        sqnr_db = (
            10 * torch.log10(signal_power / noise_power)
            if noise_power > 0
            else float("inf")
        )

        print(
            f"{n_bits:2d}-bit | Weight Error: mean={mean_weight_error:.6f}, max={max_weight_error:.6f} | "
            f"Output Error: mean={mean_output_error:.6f}, max={max_output_error:.6f} | "
            f"SQNR: {sqnr_db:.2f} dB"
        )

    print("\nNote: Lower error values indicate better quantization quality")
    print("SQNR (Signal-to-Quantization-Noise Ratio): Higher values are better")
