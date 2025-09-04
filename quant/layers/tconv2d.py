import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- TERNARY QUANT WITH STE ----------
def ternary_quantize_conv(w: torch.Tensor, thresh_ratio: float = 0.75):
    """
    Ternary quantization for convolutional weights.

    Args:
        w: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
        thresh_ratio: Threshold ratio for quantization

    Returns:
        q_nograd: Quantized weights {-1, 0, 1}
        alpha: Per-channel scaling factors
        mask: Binary mask indicating non-zero quantized weights
        delta: Per-channel thresholds
    """
    abs_w = w.abs()
    # Per-channel (per output channel) quantization - compute stats over spatial and input channel dims
    delta = thresh_ratio * abs_w.mean(
        dim=(1, 2, 3), keepdim=True
    )  # Shape: (out_channels, 1, 1, 1)
    mask = (abs_w > delta).to(w.dtype)
    q_nograd = torch.sign(w) * mask
    nonzeros = mask.sum(dim=(1, 2, 3), keepdim=True)  # Shape: (out_channels, 1, 1, 1)
    alpha = (abs_w * mask).sum(dim=(1, 2, 3), keepdim=True) / nonzeros.clamp(
        min=1
    )  # Shape: (out_channels, 1, 1, 1)
    return q_nograd, alpha, mask, delta


def ste_hard_replace(x_cont: torch.Tensor, x_disc_nograd: torch.Tensor):
    """Straight-through estimator with hard replacement."""
    return x_cont + (x_disc_nograd - x_cont).detach()


class ConvRMSNorm(nn.Module):
    """RMSNorm over channels at each spatial location.
    Input: (N, C, H, W) -> Output: (N, C, H, W)
    """

    def __init__(
        self, channels: int, eps: float = 1e-6, elementwise_affine: bool = True
    ):
        super().__init__()
        self.norm = nn.RMSNorm(channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)  # normalized over last dim (C)
        # back to (N, C, H, W)
        return x.permute(0, 3, 1, 2)


class TConv2d(nn.Conv2d):
    """Ternary quantized 2D convolutional layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        thresh_ratio: float = 0.75,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        # Make alpha per-channel (one scalar per output channel)
        self.alpha = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        self.thresh_ratio = thresh_ratio
        self.norm = ConvRMSNorm(in_channels)

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, thresh_ratio: float = 0.75):
        """Create TConv2d from existing Conv2d layer."""
        mod = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
            padding_mode=conv.padding_mode,
            thresh_ratio=thresh_ratio,
        )
        with torch.no_grad():
            mod.weight.copy_(conv.weight)
            if conv.bias is not None:
                mod.bias.copy_(conv.bias)
        mod._init_weights()
        return mod

    @torch.no_grad()
    def _init_weights(self):
        """Initialize alpha parameters based on initial quantization."""
        # Ensure weight is finite before quantization
        if not torch.isfinite(self.weight).all():
            print(f"Warning: Non-finite weights detected in layer initialization")
            self.weight.data = torch.where(
                torch.isfinite(self.weight), self.weight, torch.zeros_like(self.weight)
            )

        _, alpha, _, _ = ternary_quantize_conv(self.weight, self.thresh_ratio)
        self.alpha.data = alpha.detach()

    def forward(self, x):
        x = self.norm(x)
        # Quantize weights
        q_nograd, _, _, _ = ternary_quantize_conv(self.weight, self.thresh_ratio)
        q = ste_hard_replace(self.weight, q_nograd)
        w_q = self.alpha * q  # Use learnable alpha

        # Check for NaN in quantized weights
        if not torch.isfinite(w_q).all():
            print(f"Warning: Non-finite values in quantized weights w_q")
            w_q = torch.where(torch.isfinite(w_q), w_q, torch.zeros_like(w_q))

        # Ensure weight and bias share the input dtype to avoid dtype mismatch
        w_q = w_q.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None

        # Use the appropriate conv function based on groups
        if self.groups == 1:
            return F.conv2d(
                x, w_q, bias, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            return F.conv2d(
                x, w_q, bias, self.stride, self.padding, self.dilation, self.groups
            )

    def __repr__(self):
        return (
            f"TConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}, thresh_ratio={self.thresh_ratio})"
        )


# ---------- UTILITY FUNCTIONS ----------
def replace_conv2d_with_ternary(module, thresh_ratio=0.75, skip_names=None):
    """
    Recursively replace Conv2d layers with TConv2d layers in a module.

    Args:
        module: PyTorch module to modify
        thresh_ratio: Threshold ratio for ternary quantization
        skip_names: Set of layer names to skip replacement
    """
    if skip_names is None:
        skip_names = set()

    for name, child in module.named_children():
        if name in skip_names:
            continue

        if isinstance(child, nn.Conv2d) and not isinstance(child, TConv2d):
            # Replace Conv2d with TConv2d
            tconv = TConv2d.from_conv2d(child, thresh_ratio)
            setattr(module, name, tconv)
        else:
            # Recursively apply to child modules
            replace_conv2d_with_ternary(child, thresh_ratio, skip_names)


if __name__ == "__main__":
    print("=== Testing TConv2d Implementation ===")

    # Example: Create a regular conv layer
    regular_conv = nn.Conv2d(
        in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True
    )

    # Initialize weights with some values
    with torch.no_grad():
        regular_conv.weight.data = torch.randn(16, 3, 3, 3) * 0.1
        regular_conv.bias.data = torch.randn(16) * 0.1

    print("Original Conv2d Layer:")
    print(f"Weight shape: {regular_conv.weight.shape}")
    print(f"Bias shape: {regular_conv.bias.shape}")
    print(
        f"Weight stats - Mean: {regular_conv.weight.mean():.4f}, Std: {regular_conv.weight.std():.4f}"
    )

    # Create TConv2d from existing conv layer
    tconv = TConv2d.from_conv2d(regular_conv)
    print(f"\nTConv2d Layer (initialized from existing conv):")
    print(f"Weight shape: {tconv.weight.shape}")
    print(f"Alpha shape: {tconv.alpha.shape}")
    print(f"Threshold ratio: {tconv.thresh_ratio}")

    # Create some input data (batch_size=2, channels=3, height=32, width=32)
    x = torch.randn(2, 3, 32, 32)
    print(f"\nInput shape: {x.shape}")

    # Forward pass through both layers
    with torch.no_grad():
        regular_output = regular_conv(x)
        tconv_output = tconv(x)

    print(f"Regular conv output shape: {regular_output.shape}")
    print(f"TConv2d output shape: {tconv_output.shape}")
    print(
        f"Regular output stats - Mean: {regular_output.mean():.4f}, Std: {regular_output.std():.4f}"
    )
    print(
        f"TConv2d output stats - Mean: {tconv_output.mean():.4f}, Std: {tconv_output.std():.4f}"
    )

    # Show quantization effect
    print(f"\nQuantization effect:")
    print(f"Original weight sparsity: {(regular_conv.weight == 0).float().mean():.2%}")

    # Get quantized weights
    with torch.no_grad():
        q_nograd, alpha, mask, delta = ternary_quantize_conv(
            tconv.weight, tconv.thresh_ratio
        )
        q = ste_hard_replace(tconv.weight, q_nograd)
        w_q = alpha * q

    print(f"Quantized weight sparsity: {(w_q == 0).float().mean():.2%}")
    print(f"Alpha values (first 5): {alpha.squeeze()[:5].tolist()}")
    print(f"Delta values (first 5): {delta.squeeze()[:5].tolist()}")

    # Test with different threshold ratios
    print(f"\nTesting different threshold ratios:")
    for thresh in [0.5, 0.75, 0.9]:
        tconv_test = TConv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=True, thresh_ratio=thresh
        )
        tconv_test.weight.data = regular_conv.weight.data.clone()
        if regular_conv.bias is not None:
            tconv_test.bias.data = regular_conv.bias.data.clone()

        with torch.no_grad():
            tconv_test._init_weights()
            output = tconv_test(x)
            q_nograd, _, mask, _ = ternary_quantize_conv(tconv_test.weight, thresh)
            sparsity = (q_nograd == 0).float().mean()

        print(
            f"  Threshold {thresh}: Sparsity {sparsity:.2%}, Output std {output.std():.4f}"
        )

    # Test utility function for replacing layers in a simple model
    print(f"\n=== Testing Layer Replacement Utility ===")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
            self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
            self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x

    model = SimpleModel()
    print("Before replacement:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, TConv2d)):
            print(f"  {name}: {type(module).__name__}")

    # Replace Conv2d layers with TConv2d
    replace_conv2d_with_ternary(model, thresh_ratio=0.75)

    print("After replacement:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, TConv2d)):
            print(f"  {name}: {type(module).__name__}")

    # Test the model
    with torch.no_grad():
        output = model(x)
        print(f"Model output shape: {output.shape}")
        print(
            f"Model output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}"
        )
