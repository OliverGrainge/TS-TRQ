import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- TERNARY QUANT WITH STE ----------
def ternary_quantize(w: torch.Tensor, thresh_ratio: float = 0.75):
    """
    Return (q, alpha, mask):
      q     ∈ {-1,0,1}^{...}  (no grad)
      alpha scalar (per-tensor) scale
      mask  ∈ {0,1}^{...}     which entries survived thresholding
    """
    abs_w  = w.abs()
    delta  = thresh_ratio * abs_w.mean()
    mask   = (abs_w > delta).to(w.dtype)
    q_nograd = torch.sign(w) * mask  # {-1,0,1}
    nonzeros = mask.sum()
    alpha    = (abs_w * mask).sum() / nonzeros.clamp(min=1)

    return q_nograd, alpha, mask, delta

def ste_hard_replace(x_cont: torch.Tensor, x_disc_nograd: torch.Tensor):
    """
    Straight-through: forward uses x_disc_nograd, backward uses gradient wrt x_cont.
    """
    return x_cont + (x_disc_nograd - x_cont).detach()




class TernaryQATLinear(nn.Linear):
    """
    y = x @ (alpha * Q + L @ R)^T + b
      - Q ∈ {-1,0,1}^{out,in}  (computed each forward, STE over latent float weights)
      - alpha ∈ ℝ (per-tensor here; could go per-channel)
      - L ∈ ℝ^{out,r}, R ∈ ℝ^{r,in}; both are *frozen* (no grad) and updated only when you call reset_lowrank().
    """
    def __init__(self, in_features, out_features, bias=True, rank=8, thresh_ratio=0.75):
        super().__init__(in_features, out_features, bias=bias)
        self.thresh_ratio = thresh_ratio
        self.rank         = rank

        # We re-purpose `self.weight` (float latent weights) and quantize it in forward.
        # Low-rank factors: buffers => no grads
        self.register_buffer("L", torch.zeros(out_features, rank))
        self.register_buffer("R", torch.zeros(rank, in_features))
        self.register_buffer("alpha", torch.tensor(1.0))  # store last alpha for logging

    @classmethod
    def from_linear(cls, lin: nn.Linear, rank=8, thresh_ratio=0.75):
        mod = cls(lin.in_features, lin.out_features, bias=(lin.bias is not None),
                  rank=rank, thresh_ratio=thresh_ratio)
        with torch.no_grad():
            mod.weight.copy_(lin.weight)
            if lin.bias is not None:
                mod.bias.copy_(lin.bias)
        # initial low-rank approximation is zeros (no correction yet)
        return mod

    @torch.no_grad()
    def reset_lowrank(self, rank: int = None):
        """
        Recompute (L, R) from current quantization error using truncated SVD.
        """
        if rank is not None:
            self.rank = rank

        # Compute current ternary approx (no grad)
        q, alpha, _, _ = ternary_quantize(self.weight, self.thresh_ratio)
        E = self.weight - alpha * q  # quantization error

        # SVD
        U, S, Vh = torch.linalg.svd(E, full_matrices=False)
        r = min(self.rank, S.numel())
        U_r  = U[:, :r]
        S_r  = S[:r]
        Vh_r = Vh[:r, :]

        # Compose L and R (LoRA: L @ R ~ U S V^T)
        L = U_r * S_r.unsqueeze(0)         # (out, r)
        R = Vh_r                           # (r, in)

        # Store as buffers (frozen)
        self.L = L.detach()
        self.R = R.detach()

        # Also stash alpha just for logging/inspection
        self.alpha = alpha.detach()

    def forward(self, x):
        # 1) ternarize with STE
        q_nograd, alpha, _, _ = ternary_quantize(self.weight, self.thresh_ratio)
        q = ste_hard_replace(self.weight, q_nograd)  # forward uses q_nograd, grad flows through self.weight
        w_q = alpha * q

        # 2) add frozen low-rank correction
        if self.rank > 0 and self.L.numel() != 0 and self.R.numel() != 0:
            E_lr = self.L @ self.R
            w_hat = w_q + E_lr
        else:
            w_hat = w_q

        return F.linear(x, w_hat, self.bias)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, rank={self.rank}, thresh_ratio={self.thresh_ratio}")
    

import matplotlib.pyplot as plt

if __name__ == "__main__":
    torch.manual_seed(0)
    lin = nn.Linear(128, 256, bias=True)
    tq = TernaryQATLinear.from_linear(lin, rank=8)  # set max rank
    x = torch.randn(16, 128)

    max_rank = min(lin.weight.shape)
    errors = []

    for r in range(0, max_rank + 1):
        tq.reset_lowrank(rank=r)
        y_hat = tq(x)
        y_ref = lin(x)
        err = (y_ref - y_hat).abs().mean().item()
        errors.append(err)

    plt.figure()
    plt.plot(range(0, max_rank + 1), errors, marker="o")
    plt.xlabel("Low-rank correction rank")
    plt.ylabel("Mean absolute error vs float")
    plt.title("Quantization error vs. low-rank correction rank")
    plt.grid(True)
    plt.savefig("tsvdlinear.png")