import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- TERNARY QUANT WITH STE ----------
def ternary_quantize(w: torch.Tensor, thresh_ratio: float = 0.75):
    abs_w = w.abs()
    # Per-channel (per output channel/row) quantization
    delta = thresh_ratio * abs_w.mean(dim=1, keepdim=True)  # Shape: (out_features, 1)
    mask = (abs_w > delta).to(w.dtype)
    q_nograd = torch.sign(w) * mask
    nonzeros = mask.sum(dim=1, keepdim=True)  # Shape: (out_features, 1)
    alpha = (abs_w * mask).sum(dim=1, keepdim=True) / nonzeros.clamp(
        min=1
    )  # Shape: (out_features, 1)
    return q_nograd, alpha, mask, delta


def ste_hard_replace(x_cont: torch.Tensor, x_disc_nograd: torch.Tensor):
    return x_cont + (x_disc_nograd - x_cont).detach()


class TLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        thresh_ratio: float = 0.75,
    ):
        super().__init__(in_features, out_features, bias=bias)
        # Make alpha per-channel (one scalar per output channel)
        self.alpha = nn.Parameter(torch.ones(out_features, 1))
        self.thresh_ratio = thresh_ratio
        #self.norm = nn.RMSNorm(normalized_shape=in_features)

    @classmethod
    def from_linear(
        cls,
        lin: nn.Linear,
    ):
        mod = cls(
            lin.in_features,
            lin.out_features,
            bias=(lin.bias is not None),
        )
        with torch.no_grad():
            mod.weight.copy_(lin.weight)
            if lin.bias is not None:
                mod.bias.copy_(lin.bias)
        mod._init_weights()
        return mod

    @torch.no_grad()
    def _init_weights(self):
        # Ensure weight is finite before quantization
        if not torch.isfinite(self.weight).all():
            print(f"Warning: Non-finite weights detected in layer initialization")
            self.weight.data = torch.where(
                torch.isfinite(self.weight), self.weight, torch.zeros_like(self.weight)
            )

        _, alpha, _, _ = ternary_quantize(self.weight, self.thresh_ratio)
        self.alpha.data = alpha.detach()

    def forward(self, x):
        #x = self.norm(x)
        q_nograd, _, _, _ = ternary_quantize(self.weight, self.thresh_ratio)
        q = ste_hard_replace(self.weight, q_nograd)
        w_q = self.alpha * q  # Use learnable alpha

        # Check for NaN in quantized weights
        if not torch.isfinite(w_q).all():
            print(f"Warning: Non-finite values in quantized weights w_q")
            w_q = torch.where(torch.isfinite(w_q), w_q, torch.zeros_like(w_q))

        # Final check for NaN in combined weights
        if not torch.isfinite(w_q).all():
            print(f"Warning: Non-finite values in final weights w_hat")
            w_q = torch.where(torch.isfinite(w_q), w_q, torch.zeros_like(w_q))

        # Ensure weight and bias share the input dtype to avoid dtype mismatch (e.g., FP16 vs FP32)
        w_q = w_q.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_q, bias)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


if __name__ == "__main__":
    # Example: Create a regular linear layer
    regular_linear = nn.Linear(in_features=10, out_features=5, bias=True)

    # Initialize weights with some values
    with torch.no_grad():
        regular_linear.weight.data = torch.randn(5, 10) * 0.1
        regular_linear.bias.data = torch.randn(5) * 0.1

    print("Original Linear Layer:")
    print(f"Weight shape: {regular_linear.weight.shape}")
    print(f"Bias shape: {regular_linear.bias.shape}")
    print(
        f"Weight stats - Mean: {regular_linear.weight.mean():.4f}, Std: {regular_linear.weight.std():.4f}"
    )

    # Create TLinear from existing linear layer
    tlinear = TLinear.from_linear(regular_linear)
    print(f"\nTLinear Layer (initialized from existing linear):")
    print(f"Weight shape: {tlinear.weight.shape}")
    print(f"Alpha shape: {tlinear.alpha.shape}")
    print(f"Threshold ratio: {tlinear.thresh_ratio}")

    # Create some input data
    batch_size = 4
    x = torch.randn(batch_size, 10)
    print(f"\nInput shape: {x.shape}")

    # Forward pass through both layers
    with torch.no_grad():
        regular_output = regular_linear(x)
        tlinear_output = tlinear(x)

    print(f"Regular linear output shape: {regular_output.shape}")
    print(f"TLinear output shape: {tlinear_output.shape}")
    print(
        f"Regular output stats - Mean: {regular_output.mean():.4f}, Std: {regular_output.std():.4f}"
    )
    print(
        f"TLinear output stats - Mean: {tlinear_output.mean():.4f}, Std: {tlinear_output.std():.4f}"
    )

    # Show quantization effect
    print(f"\nQuantization effect:")
    print(
        f"Original weight sparsity: {(regular_linear.weight == 0).float().mean():.2%}"
    )

    # Get quantized weights
    with torch.no_grad():
        q_nograd, alpha, mask, delta = ternary_quantize(
            tlinear.weight, tlinear.thresh_ratio
        )
        q = ste_hard_replace(tlinear.weight, q_nograd)
        w_q = alpha * q

    print(f"Quantized weight sparsity: {(w_q == 0).float().mean():.2%}")
    print(f"Alpha values: {alpha.squeeze().tolist()}")
    print(f"Delta values: {delta.squeeze().tolist()}")

    # Test with different threshold ratios
    print(f"\nTesting different threshold ratios:")
    for thresh in [0.5, 0.75, 0.9]:
        tlinear_test = TLinear(10, 5, bias=True, thresh_ratio=thresh)
        tlinear_test.weight.data = regular_linear.weight.data.clone()
        if regular_linear.bias is not None:
            tlinear_test.bias.data = regular_linear.bias.data.clone()

        with torch.no_grad():
            tlinear_test._init_weights()
            output = tlinear_test(x)
            q_nograd, _, mask, _ = ternary_quantize(tlinear_test.weight, thresh)
            sparsity = (q_nograd == 0).float().mean()

        print(
            f"  Threshold {thresh}: Sparsity {sparsity:.2%}, Output std {output.std():.4f}"
        )
