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


class TSVDLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        rank=8,
        thresh_ratio=0.75,
        reg_scale=0.5,
        reg_type="l1",
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.thresh_ratio = thresh_ratio
        self.rank = rank
        self.reg_scale = reg_scale
        self.reg_type = reg_type

        self.register_buffer("L", torch.zeros(out_features, rank))
        self.register_buffer("R", torch.zeros(rank, in_features))
        # Make alpha per-channel (one scalar per output channel)
        self.alpha = nn.Parameter(torch.ones(out_features, 1))

        # Layer-wise scalars for LR approximation
        self.lr_scalars = nn.Parameter(torch.ones(out_features, 1))

    # self.norm = nn.RMSNorm(normalized_shape=in_features)

    @classmethod
    def from_linear(
        cls, lin: nn.Linear, rank=8, thresh_ratio=0.75, reg_scale=0.5, reg_type="l1"
    ):
        mod = cls(
            lin.in_features,
            lin.out_features,
            bias=(lin.bias is not None),
            rank=rank,
            thresh_ratio=thresh_ratio,
            reg_scale=reg_scale,
            reg_type=reg_type,
        )
        with torch.no_grad():
            mod.weight.copy_(lin.weight)
            if lin.bias is not None:
                mod.bias.copy_(lin.bias)
        mod._init_weights(rank=rank)
        return mod

    @torch.no_grad()
    def _init_weights(self, rank: int = None):
        if rank is not None:
            self.rank = rank

        # Ensure weight is finite before quantization
        if not torch.isfinite(self.weight).all():
            print(f"Warning: Non-finite weights detected in layer initialization")
            self.weight.data = torch.where(
                torch.isfinite(self.weight), self.weight, torch.zeros_like(self.weight)
            )

        q, alpha, _, _ = ternary_quantize(self.weight, self.thresh_ratio)
        E = self.weight - alpha * q

        # Check for NaN/Inf in the error matrix before SVD
        if not torch.isfinite(E).all():
            print(f"Warning: Non-finite values in error matrix E")
            E = torch.where(torch.isfinite(E), E, torch.zeros_like(E))

        U, S, Vh = torch.linalg.svd(E, full_matrices=False)

        # Check SVD results for NaN/Inf
        if not (
            torch.isfinite(U).all()
            and torch.isfinite(S).all()
            and torch.isfinite(Vh).all()
        ):
            print(f"Warning: SVD produced non-finite values, using zero initialization")

            r = min(self.rank, self.weight.shape[0], self.weight.shape[1])
            L = torch.zeros(
                self.weight.shape[0],
                r,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            R = torch.zeros(
                r,
                self.weight.shape[1],
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
        else:
            r = min(self.rank, S.numel())
            U_r = U[:, :r]
            S_r = S[:r]
            Vh_r = Vh[:r, :]

            L = U_r * S_r.unsqueeze(0)
            R = Vh_r

        self.L = L.detach().half()
        self.R = R.detach().half()
        # Initialize per-channel alpha - alpha already has shape (out_features, 1)
        self.alpha.data = alpha.detach()
        self.lr_scalars.data.fill_(1.0)

    def forward(self, x):
        # x = self.norm(x)
        q_nograd, _, _, _ = ternary_quantize(self.weight, self.thresh_ratio)
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
            E_lr = (self.lr_scalars.to(x.dtype) * L) @ R
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

        # Ensure weight and bias share the input dtype to avoid dtype mismatch (e.g., FP16 vs FP32)
        w_hat = w_hat.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_hat, bias)

    def layer_reg_loss(self):
        if self.reg_type == "l1":
            return self.reg_scale * self.lr_scalars.abs().mean()
        elif self.reg_type == "l2":
            return self.reg_scale * self.lr_scalars.pow(2).mean()
        else:
            raise ValueError(f"Invalid regularization type: {self.reg_type}")

    def reset_lowrank(self, rank: int):
        """Reset the low-rank approximation with a new rank"""
        self._init_weights(rank=rank)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, rank={self.rank}, thresh_ratio={self.thresh_ratio}"
        )


if __name__ == "__main__":

    def replace_linear_with_ternary_svd(
        module, rank=8, thresh_ratio=0.75, reg_scale=0.5, reg_type="l1", skip_names=None
    ):
        """
        Recursively replace Linear layers with TSVDLinear layers in a module.

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

            if isinstance(child, nn.Linear) and not isinstance(child, TSVDLinear):
                # Replace Linear with TSVDLinear
                tlinear = TSVDLinear.from_linear(
                    child,
                    rank=rank,
                    thresh_ratio=thresh_ratio,
                    reg_scale=reg_scale,
                    reg_type=reg_type,
                )
                setattr(module, name, tlinear)
            else:
                # Recursively apply to child modules
                replace_linear_with_ternary_svd(
                    child, rank, thresh_ratio, reg_scale, reg_type, skip_names
                )


def compute_total_reg_loss(module):
    """Compute total regularization loss from all TSVDLinear layers in a module."""
    total_loss = 0.0
    for m in module.modules():
        if isinstance(m, TSVDLinear):
            total_loss += m.layer_reg_loss()
    return total_loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=== Testing TSVDLinear Implementation ===")
    torch.manual_seed(0)

    # Create a regular linear layer
    linear = nn.Linear(128, 256, bias=True)

    # Initialize with some values
    with torch.no_grad():
        linear.weight.data = torch.randn(256, 128) * 0.1
        linear.bias.data = torch.randn(256) * 0.1

    print(f"Original Linear weight shape: {linear.weight.shape}")

    # Create TSVDLinear
    tsvd_linear = TSVDLinear.from_linear(linear, rank=8)

    print(f"TSVDLinear L shape: {tsvd_linear.L.shape}")
    print(f"TSVDLinear R shape: {tsvd_linear.R.shape}")
    print(f"TSVDLinear alpha shape: {tsvd_linear.alpha.shape}")
    print(f"TSVDLinear lr_scalars shape: {tsvd_linear.lr_scalars.shape}")

    # Test with some input
    x = torch.randn(16, 128)
    print(f"\nInput shape: {x.shape}")

    # Forward passes
    with torch.no_grad():
        y_ref = linear(x)
        y_tsvd = tsvd_linear(x)

    print(f"Original output shape: {y_ref.shape}")
    print(f"TSVDLinear output shape: {y_tsvd.shape}")
    print(f"Original output stats - Mean: {y_ref.mean():.4f}, Std: {y_ref.std():.4f}")
    print(
        f"TSVDLinear output stats - Mean: {y_tsvd.mean():.4f}, Std: {y_tsvd.std():.4f}"
    )

    # Test error vs rank
    max_possible_rank = min(linear.weight.shape[0], linear.weight.shape[1])
    max_rank = min(50, max_possible_rank)  # Limit for practical testing
    errors = []

    print(f"\n=== Testing Error vs Rank (max_rank={max_rank}) ===")
    print(f"{'Rank':>6} | {'Mean Absolute Error':>20} | {'Reg Loss':>12}")
    print("-" * 45)

    for r in range(0, max_rank + 1):
        tsvd_linear.reset_lowrank(rank=r)
        with torch.no_grad():
            y_hat = tsvd_linear(x)
            err = (y_ref - y_hat).abs().mean().item()
            reg_loss = tsvd_linear.layer_reg_loss().item()
        errors.append(err)
        print(f"{r:6d} | {err:20.6e} | {reg_loss:12.6f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, max_rank + 1), errors, marker="o")
    plt.xlabel("Low-rank correction rank")
    plt.ylabel("Mean absolute error vs original")
    plt.title("Linear Quantization error vs. low-rank correction rank")
    plt.grid(True)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

    # Test regularization
    print(f"\n=== Testing Regularization ===")
    print(f"L1 regularization loss: {tsvd_linear.layer_reg_loss():.6f}")

    # Test different regularization types
    tsvd_linear_l2 = TSVDLinear.from_linear(linear, rank=8, reg_type="l2")
    print(f"L2 regularization loss: {tsvd_linear_l2.layer_reg_loss():.6f}")

    # Test different regularization scales
    print(f"\nTesting different regularization scales:")
    for scale in [0.1, 0.5, 1.0, 2.0]:
        tsvd_test = TSVDLinear.from_linear(
            linear, rank=8, reg_scale=scale, reg_type="l1"
        )
        reg_loss = tsvd_test.layer_reg_loss().item()
        print(f"  Scale {scale}: L1 reg loss {reg_loss:.6f}")

    # Test utility function for replacing layers in a model
    print(f"\n=== Testing Model Replacement Utility ===")

    class SimpleLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    model = SimpleLinearModel()
    print("Before replacement:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, TSVDLinear)):
            print(f"  {name}: {type(module).__name__}")

    # Replace Linear layers with TSVDLinear
    replace_linear_with_ternary_svd(model, rank=8, thresh_ratio=0.75)

    print("After replacement:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, TSVDLinear)):
            print(f"  {name}: {type(module).__name__}")

    # Test the model
    with torch.no_grad():
        output = model(x)
        total_reg = compute_total_reg_loss(model)

    print(f"Model output shape: {output.shape}")
    print(f"Model output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
    print(f"Total regularization loss: {total_reg:.6f}")

    # Test different threshold ratios
    print(f"\n=== Testing Different Threshold Ratios ===")
    for thresh in [0.5, 0.75, 0.9]:
        test_linear = TSVDLinear.from_linear(linear, rank=8, thresh_ratio=thresh)
        with torch.no_grad():
            output = test_linear(x)
            # Get sparsity
            q_nograd, _, _, _ = ternary_quantize(test_linear.weight, thresh)
            sparsity = (q_nograd == 0).float().mean()
            error = (y_ref - output).abs().mean()

        print(
            f"  Threshold {thresh}: Sparsity {sparsity:.2%}, Error {error:.6f}, Output std {output.std():.4f}"
        )

    # Test SVD quality analysis
    print(f"\n=== SVD Quality Analysis ===")
    tsvd_linear.reset_lowrank(rank=16)

    with torch.no_grad():
        # Get the original error matrix
        q, alpha, _, _ = ternary_quantize(tsvd_linear.weight, tsvd_linear.thresh_ratio)
        E = tsvd_linear.weight - alpha * q

        # Compute SVD
        U, S, Vh = torch.linalg.svd(E, full_matrices=False)

        # Show singular value decay
        print(f"Top 10 singular values: {S[:10].tolist()}")
        print(f"Singular value ratio (S[9]/S[0]): {S[9]/S[0]:.6f}")

        # Show reconstruction quality for different ranks
        print(f"\nReconstruction quality:")
        for r in [1, 2, 4, 8, 16]:
            if r <= len(S):
                U_r = U[:, :r]
                S_r = S[:r]
                Vh_r = Vh[:r, :]
                E_approx = U_r @ torch.diag(S_r) @ Vh_r
                recon_error = (E - E_approx).pow(2).mean().sqrt()
                print(f"  Rank {r}: Reconstruction RMSE {recon_error:.6f}")

    # Test memory and parameter analysis
    print(f"\n=== Memory and Parameter Analysis ===")

    # Original linear parameters
    orig_params = linear.weight.numel() + (
        linear.bias.numel() if linear.bias is not None else 0
    )

    # TSVDLinear parameters
    tsvd_params = (
        tsvd_linear.weight.numel()
        + (tsvd_linear.bias.numel() if tsvd_linear.bias is not None else 0)
        + tsvd_linear.alpha.numel()
        + tsvd_linear.lr_scalars.numel()
    )

    # SVD buffer sizes (not trainable)
    svd_buffers = tsvd_linear.L.numel() + tsvd_linear.R.numel()

    print(f"Original Linear parameters: {orig_params:,}")
    print(f"TSVDLinear trainable parameters: {tsvd_params:,}")
    print(f"TSVDLinear SVD buffers: {svd_buffers:,}")
    print(f"Total TSVDLinear memory: {tsvd_params + svd_buffers:,}")
    print(f"Parameter ratio: {tsvd_params / orig_params:.3f}x")

    # Test gradients flow
    print(f"\n=== Testing Gradient Flow ===")
    tsvd_linear.train()
    x_grad = torch.randn(4, 128, requires_grad=True)
    y_pred = tsvd_linear(x_grad)
    loss = y_pred.sum()
    loss.backward()

    print(f"Input gradient norm: {x_grad.grad.norm():.6f}")
    print(f"Weight gradient norm: {tsvd_linear.weight.grad.norm():.6f}")
    print(f"Alpha gradient norm: {tsvd_linear.alpha.grad.norm():.6f}")
    print(f"LR scalars gradient norm: {tsvd_linear.lr_scalars.grad.norm():.6f}")

    # Test with different input batch sizes
    print(f"\n=== Testing Different Batch Sizes ===")
    for batch_size in [1, 4, 16, 64]:
        x_test = torch.randn(batch_size, 128)
        with torch.no_grad():
            y_test = tsvd_linear(x_test)
        print(
            f"  Batch size {batch_size}: Output shape {y_test.shape}, Mean {y_test.mean():.4f}"
        )

    print(f"\n=== Test Complete ===")
