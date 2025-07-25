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
    alpha = (abs_w * mask).sum(dim=1, keepdim=True) / nonzeros.clamp(min=1)  # Shape: (out_features, 1)
    return q_nograd, alpha, mask, delta


def ste_hard_replace(x_cont: torch.Tensor, x_disc_nograd: torch.Tensor):
    return x_cont + (x_disc_nograd - x_cont).detach()


class TSVDLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, rank=8, thresh_ratio=0.75):
        super().__init__(in_features, out_features, bias=bias)
        self.thresh_ratio = thresh_ratio
        self.rank = rank

        self.register_buffer("L", torch.zeros(out_features, rank))
        self.register_buffer("R", torch.zeros(rank, in_features))
        # Make alpha per-channel (one scalar per output channel)
        self.alpha = nn.Parameter(torch.ones(out_features, 1))

        # Layer-wise scalars for LR approximation
        self.lr_scalars = nn.Parameter(torch.ones(out_features, 1))

    @classmethod
    def from_linear(cls, lin: nn.Linear, rank=8, thresh_ratio=0.75):
        mod = cls(
            lin.in_features,
            lin.out_features,
            bias=(lin.bias is not None),
            rank=rank,
            thresh_ratio=thresh_ratio,
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

        self.L = L.detach()
        self.R = R.detach()
        # Initialize per-channel alpha - alpha already has shape (out_features, 1)
        self.alpha.data = alpha.detach()
        self.lr_scalars.data.fill_(1.0)

    def forward(self, x):
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
        return self.lr_scalars.abs().mean()

    def reset_lowrank(self, rank: int):
        """Reset the low-rank approximation with a new rank"""
        self._init_weights(rank=rank)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, rank={self.rank}, thresh_ratio={self.thresh_ratio}"
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    lin = nn.Linear(128, 256, bias=True)
    tq = TSVDLinear.from_linear(lin, rank=8)  # set max rank
    x = torch.randn(16, 128)

    max_rank = min(lin.weight.shape)
    errors = []

    print(f"{'Rank':>6} | {'Mean Absolute Error':>20}")
    print("-" * 30)
    for r in range(0, max_rank + 1):
        tq.reset_lowrank(rank=r)
        y_hat = tq(x)
        y_ref = lin(x)
        err = (y_ref - y_hat).abs().mean().item()
        errors.append(err)
        print(f"{r:6d} | {err:20.6e}")

    plt.figure()
    plt.plot(range(0, max_rank + 1), errors, marker="o")
    plt.xlabel("Low-rank correction rank")
    plt.ylabel("Mean absolute error vs float")
    plt.title("Quantization error vs. low-rank correction rank")
    plt.grid(True)
    plt.show()
