import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- helpers ----------
def ternary_quantize(w: torch.Tensor, thresh_ratio: float = 0.75):
    abs_w = w.abs()
    delta = thresh_ratio * abs_w.mean(dim=1, keepdim=True)           # (out,1)
    mask  = (abs_w > delta).to(w.dtype)
    q_ng  = torch.sign(w) * mask                                     # ternary
    nonzeros = mask.sum(dim=1, keepdim=True)
    alpha = (abs_w * mask).sum(dim=1, keepdim=True) / nonzeros.clamp(min=1)
    return q_ng, alpha, mask, delta


def ste_hard_replace(x_cont: torch.Tensor, x_disc_nograd: torch.Tensor):
    return x_cont + (x_disc_nograd - x_cont).detach()


# ---------- TSVDLinear ----------
class TSVDTLinear(nn.Linear):
    r"""
    Ternary + low-rank residual.
    *   L & R are trainable.
    *   lr_scalars fixed at 1.0 (buffer, no grad).
    *   Optional L1/L2 regularisation on L‖ + ‖R‖   or   ‖LR‖.
    """

    def __init__(
        self, in_features, out_features,
        bias=True, rank=8, thresh_ratio=0.75,
        reg_scale=1e-4, reg_type="l2", reg_target="E_lr"  # "factor" | "E_lr"
    ):
        super().__init__(in_features, out_features, bias=bias)

        self.thresh_ratio = thresh_ratio
        self.rank         = rank

        # ----- trainable low-rank factors -----
        self.L = nn.Parameter(torch.empty(out_features, rank))
        self.R = nn.Parameter(torch.empty(rank, in_features))

        # ----- fixed scalar gate (all-ones) -----
        self.register_buffer("lr_scalars", torch.ones(out_features, 1))

        # ----- misc learnables -----
        self.alpha = nn.Parameter(torch.ones(out_features, 1))

        # ----- regularisation config -----
        self.reg_scale  = float(reg_scale)
        self.reg_type   = reg_type.lower()   # "l1" | "l2"
        self.reg_target = reg_target.lower() # "factor" | "e_lr"

        # populate parameters via SVD
        self._init_weights(rank)

    # ---- convenience constructor ----
    @classmethod
    def from_linear(
        cls, lin: nn.Linear, rank=8, thresh_ratio=0.75,
        reg_scale=1e-4, reg_type="l2", reg_target="factor"
    ):
        mod = cls(lin.in_features, lin.out_features,
                  bias=(lin.bias is not None),
                  rank=rank, thresh_ratio=thresh_ratio,
                  reg_scale=reg_scale, reg_type=reg_type,
                  reg_target=reg_target)
        with torch.no_grad():
            mod.weight.copy_(lin.weight)
            if lin.bias is not None:
                mod.bias.copy_(lin.bias)
        # re-init low-rank pieces for the transferred weights
        mod._init_weights(rank)
        return mod

    # ---- SVD initialisation ----
    @torch.no_grad()
    def _init_weights(self, rank=None):
        if rank is not None:
            self.rank = rank

        # 1. ternary decomposition
        q_ng, alpha, *_ = ternary_quantize(self.weight, self.thresh_ratio)
        E = self.weight - alpha * q_ng

        # 2. truncated SVD of the residual
        U, S, Vh = torch.linalg.svd(E, full_matrices=False)
        r = min(self.rank, S.numel())
        L0 = U[:, :r] * S[:r].unsqueeze(0)   # (out,r)
        R0 = Vh[:r, :]                       # (r,in)

        # 3. copy into parameters (keep requires_grad=True)
        self.L.data.copy_(L0)
        self.R.data.copy_(R0)
        self.alpha.data.copy_(alpha)

    # ---- forward ----
    def forward(self, x: torch.Tensor):
        q_ng, _, _, _ = ternary_quantize(self.weight, self.thresh_ratio)
        q = ste_hard_replace(self.weight, q_ng)          # STE
        w_q = self.alpha * q                             # ternary branch

        # low-rank residual (lr_scalars fixed at 1.0)
        E_lr = self.L @ self.R                           # (out,in)
        w_hat = w_q + E_lr

        return F.linear(x, w_hat.to(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype))

    # ---- regulariser ----
    def layer_reg_loss(self):
        p = 1 if self.reg_type == "l1" else 2

        if self.reg_target == "factor":
            loss = (self.L.norm(p=p) / self.L.numel()
                    + self.R.norm(p=p) / self.R.numel())
        elif self.reg_target == "e_lr":
            loss = (self.L @ self.R).norm(p=p) / (self.L.size(0) * self.R.size(1))
        else:
            raise ValueError("reg_target must be 'factor' or 'E_lr'")

        return self.reg_scale * loss

    # ---- utility ----
    def reset_lowrank(self, rank: int):
        """Re-initialise L & R with a new rank (SVD again)."""
        self._init_weights(rank)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, rank={self.rank}, "
                f"thresh_ratio={self.thresh_ratio}, "
                f"reg={self.reg_type}@{self.reg_scale}, target={self.reg_target}")
    

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
