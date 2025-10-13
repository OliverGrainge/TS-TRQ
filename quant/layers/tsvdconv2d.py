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


class TSVDConv2d(nn.Conv2d):
    """Ternary quantized 2D convolutional layer with SVD error correction."""

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
        rank=8,
        thresh_ratio=0.75,
        reg_scale=0.5,
        reg_type="l1",
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
        self.thresh_ratio = thresh_ratio
        self.rank = rank
        self.reg_scale = reg_scale
        self.reg_type = reg_type

        # Calculate flattened input dimension for SVD
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size_tuple = kernel_size
        self.flattened_in_dim = in_channels * kernel_size[0] * kernel_size[1]

        # Initialize SVD matrices as buffers
        self.register_buffer("L", torch.zeros(out_channels, rank))
        self.register_buffer("R", torch.zeros(rank, self.flattened_in_dim))

        # Make alpha per-channel (one scalar per output channel)
        self.alpha = nn.Parameter(torch.ones(out_channels, 1, 1, 1))

        # Layer-wise scalars for LR approximation
        self.lr_scalars = nn.Parameter(torch.ones(out_channels, 1, 1, 1))

        # RMSNorm (rms taken over input channels)
        # self.norm = ConvRMSNorm(in_channels)

    @classmethod
    def from_conv2d(
        cls, conv: nn.Conv2d, rank=8, thresh_ratio=0.75, reg_scale=0.5, reg_type="l1"
    ):
        """Create TSVDConv2d from existing Conv2d layer."""
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
            rank=rank,
            thresh_ratio=thresh_ratio,
            reg_scale=reg_scale,
            reg_type=reg_type,
        )
        with torch.no_grad():
            mod.weight.copy_(conv.weight)
            if conv.bias is not None:
                mod.bias.copy_(conv.bias)
        mod._init_weights(rank=rank)
        return mod

    @torch.no_grad()
    def _init_weights(self, rank: int = None):
        """Initialize SVD decomposition and alpha parameters."""
        if rank is not None:
            self.rank = rank

        # Ensure weight is finite before quantization
        if not torch.isfinite(self.weight).all():
            print(f"Warning: Non-finite weights detected in layer initialization")
            self.weight.data = torch.where(
                torch.isfinite(self.weight), self.weight, torch.zeros_like(self.weight)
            )

        # Get ternary quantization
        q, alpha, _, _ = ternary_quantize_conv(self.weight, self.thresh_ratio)

        # Compute error matrix
        E = self.weight - alpha * q

        # Reshape for SVD: (out_channels, in_channels * kernel_h * kernel_w)
        original_shape = E.shape
        E_flat = E.view(original_shape[0], -1)  # (out_channels, flattened_features)

        # Check for NaN/Inf in the error matrix before SVD
        if not torch.isfinite(E_flat).all():
            print(f"Warning: Non-finite values in error matrix E")
            E_flat = torch.where(
                torch.isfinite(E_flat), E_flat, torch.zeros_like(E_flat)
            )

        # Perform SVD
        U, S, Vh = torch.linalg.svd(E_flat, full_matrices=False)

        r = min(self.rank, S.numel())
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]

        L = U_r * S_r.unsqueeze(0)  # (out_channels, rank)
        R = Vh_r  # (rank, flattened_features)

        # Update buffers
        self.L = L.detach().half()
        self.R = R.detach().half()

        # Initialize per-channel alpha - alpha already has shape (out_channels, 1, 1, 1)
        self.alpha.data = alpha.detach()
        self.lr_scalars.data.fill_(1.0)

    def forward(self, x):
        # x = self.norm(x)

        # Get ternary quantization
        q_nograd, _, _, _ = ternary_quantize_conv(self.weight, self.thresh_ratio)
        q = ste_hard_replace(self.weight, q_nograd)
        w_q = self.alpha * q  # Use learnable alpha

        # Check for NaN in quantized weights
        if not torch.isfinite(w_q).all():
            print(f"Warning: Non-finite values in quantized weights w_q")
            w_q = torch.where(torch.isfinite(w_q), w_q, torch.zeros_like(w_q))

        # Ensure low-rank buffers are on the correct device / dtype before using them
        L = self.L.to(device=x.device, dtype=x.dtype)
        R = self.R.to(device=x.device, dtype=x.dtype)

        if self.rank > 0 and L.numel() != 0 and R.numel() != 0:
            # Compute low-rank correction
            E_lr_flat = (
                self.lr_scalars.view(-1, 1).to(x.dtype) * L
            ) @ R  # (out_channels, flattened_features)

            # Reshape back to conv weight shape
            E_lr = E_lr_flat.view(self.weight.shape)

            # Check for NaN in low-rank correction
            if not torch.isfinite(E_lr).all():
                print(f"Warning: Non-finite values in low-rank correction E_lr")
                E_lr = torch.where(torch.isfinite(E_lr), E_lr, torch.zeros_like(E_lr))

            w_hat = w_q + E_lr
        else:
            w_hat = w_q

        # Final check for NaN in combined weights
        if not torch.isfinite(w_hat).all():
            print(f"Warning: Non-finite values in final weights w_hat")
            w_hat = torch.where(torch.isfinite(w_hat), w_hat, torch.zeros_like(w_hat))

        # Ensure weight and bias share the input dtype to avoid dtype mismatch
        w_hat = w_hat.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None

        return F.conv2d(
            x, w_hat, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def layer_reg_loss(self):
        """Compute regularization loss for low-rank scalars."""
        if self.reg_type == "l1":
            return self.reg_scale * self.lr_scalars.abs().mean()
        elif self.reg_type == "l2":
            return self.reg_scale * self.lr_scalars.pow(2).mean()
        else:
            raise ValueError(f"Invalid regularization type: {self.reg_type}")

    def reset_lowrank(self, rank: int):
        """Reset the low-rank approximation with a new rank."""
        self._init_weights(rank=rank)

    def __repr__(self):
        return (
            f"TSVDConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}, "
            f"rank={self.rank}, thresh_ratio={self.thresh_ratio})"
        )


# ---------- UTILITY FUNCTIONS ----------
def replace_conv2d_with_ternary_svd(
    module, rank=8, thresh_ratio=0.75, reg_scale=0.5, reg_type="l1", skip_names=None
):
    """
    Recursively replace Conv2d layers with TSVDConv2d layers in a module.

    Args:
        module: PyTorch module to modify
        rank: Rank for SVD approximation
        thresh_ratio: Threshold ratio for ternary quantization
        reg_scale: Regularization scale
        reg_type: Regularization type ("l1" or "l2")
        skip_names: Set of layer names to skip replacement
    """
    if skip_names is None:
        skip_names = set()

    for name, child in module.named_children():
        if name in skip_names:
            continue

        if isinstance(child, nn.Conv2d) and not isinstance(child, TSVDConv2d):
            # Replace Conv2d with TSVDConv2d
            tconv = TSVDConv2d.from_conv2d(
                child,
                rank=rank,
                thresh_ratio=thresh_ratio,
                reg_scale=reg_scale,
                reg_type=reg_type,
            )
            setattr(module, name, tconv)
        else:
            # Recursively apply to child modules
            replace_conv2d_with_ternary_svd(
                child, rank, thresh_ratio, reg_scale, reg_type, skip_names
            )


def compute_total_reg_loss(module):
    """Compute total regularization loss from all TSVDConv2d layers in a module."""
    total_loss = 0.0
    for m in module.modules():
        if isinstance(m, TSVDConv2d):
            total_loss += m.layer_reg_loss()
    return total_loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    print("=== Testing TSVDConv2d Implementation ===")
    torch.manual_seed(0)

    # Create a regular conv layer
    conv = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)

    # Initialize with some values
    with torch.no_grad():
        conv.weight.data = torch.randn(256, 128, 3, 3)
        conv.bias.data = torch.randn(256) * 0.1

    print(f"Original Conv2d weight shape: {conv.weight.shape}")

    # Create TSVDConv2d
    tsvd_conv = TSVDConv2d.from_conv2d(conv, rank=8)

    print(f"TSVDConv2d L shape: {tsvd_conv.L.shape}")
    print(f"TSVDConv2d R shape: {tsvd_conv.R.shape}")
    print(f"TSVDConv2d alpha shape: {tsvd_conv.alpha.shape}")
    print(f"TSVDConv2d lr_scalars shape: {tsvd_conv.lr_scalars.shape}")

    # Test with some input
    x = torch.randn(2, 128, 32, 32)
    print(f"\nInput shape: {x.shape}")

    # Forward passes
    with torch.no_grad():
        y_ref = conv(x)
        y_tsvd = tsvd_conv(x)

    print(f"Original output shape: {y_ref.shape}")
    print(f"TSVDConv2d output shape: {y_tsvd.shape}")
    print(f"Original output stats - Mean: {y_ref.mean():.4f}, Std: {y_ref.std():.4f}")
    print(
        f"TSVDConv2d output stats - Mean: {y_tsvd.mean():.4f}, Std: {y_tsvd.std():.4f}"
    )

    # Test error vs rank
    max_possible_rank = min(conv.weight.shape[0], np.prod(conv.weight.shape[1:]))
    max_rank = min(20, max_possible_rank)  # Limit for practical testing
    errors = []

    print(f"\n=== Testing Error vs Rank (max_rank={max_rank}) ===")
    print(f"{'Rank':>6} | {'Mean Absolute Error':>20} | {'Reg Loss':>12}")
    print("-" * 45)

    for r in range(0, max_rank + 1):
        tsvd_conv.reset_lowrank(rank=r)
        with torch.no_grad():
            y_hat = tsvd_conv(x)
            err = (y_ref - y_hat).abs().mean().item()
            reg_loss = tsvd_conv.layer_reg_loss().item()
        errors.append(err)
        print(f"{r:6d} | {err:20.6e} | {reg_loss:12.6f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, max_rank + 1), errors, marker="o")
    plt.xlabel("Low-rank correction rank")
    plt.ylabel("Mean absolute error vs original")
    plt.title("Conv2d Quantization error vs. low-rank correction rank")
    plt.grid(True)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

    # Test regularization
    print(f"\n=== Testing Regularization ===")
    print(f"L1 regularization loss: {tsvd_conv.layer_reg_loss():.6f}")

    # Test different regularization types
    tsvd_conv_l2 = TSVDConv2d.from_conv2d(conv, rank=8, reg_type="l2")
    print(f"L2 regularization loss: {tsvd_conv_l2.layer_reg_loss():.6f}")

    # Test utility function for replacing layers in a model
    print(f"\n=== Testing Model Replacement Utility ===")

    class SimpleConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
            self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
            self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)
            return x

    model = SimpleConvModel()
    print("Before replacement:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, TSVDConv2d)):
            print(f"  {name}: {type(module).__name__}")

    # Replace Conv2d layers with TSVDConv2d
    replace_conv2d_with_ternary_svd(model, rank=8, thresh_ratio=0.75)

    print("After replacement:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, TSVDConv2d)):
            print(f"  {name}: {type(module).__name__}")

    # Test the model
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(x)
        total_reg = compute_total_reg_loss(model)

    print(f"Model output shape: {output.shape}")
    print(f"Model output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
    print(f"Total regularization loss: {total_reg:.6f}")

    # Test different threshold ratios
    print(f"\n=== Testing Different Threshold Ratios ===")
    for thresh in [0.5, 0.75, 0.9]:
        test_conv = TSVDConv2d.from_conv2d(conv, rank=8, thresh_ratio=thresh)
        with torch.no_grad():
            output = test_conv(x)
            # Get sparsity
            q_nograd, _, _, _ = ternary_quantize_conv(test_conv.weight, thresh)
            sparsity = (q_nograd == 0).float().mean()
            error = (y_ref - output).abs().mean()

        print(
            f"  Threshold {thresh}: Sparsity {sparsity:.2%}, Error {error:.6f}, Output std {output.std():.4f}"
        )
