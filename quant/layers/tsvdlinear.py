import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- TERNARY QUANT WITH STE ----------
def ternary_quantize(w: torch.Tensor, thresh_ratio: float = 0.75):
    abs_w  = w.abs()
    delta  = thresh_ratio * abs_w.mean()
    mask   = (abs_w > delta).to(w.dtype)
    q_nograd = torch.sign(w) * mask
    nonzeros = mask.sum()
    alpha    = (abs_w * mask).sum() / nonzeros.clamp(min=1)

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
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Make alpha learnable

        # Layer-wise scalars for LR approximation
        self.lr_scalars = nn.Parameter(torch.ones(out_features, 1))

    @classmethod
    def from_linear(cls, lin: nn.Linear, rank=8, thresh_ratio=0.75):
        mod = cls(lin.in_features, lin.out_features, bias=(lin.bias is not None),
                  rank=rank, thresh_ratio=thresh_ratio)
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

        q, alpha, _, _ = ternary_quantize(self.weight, self.thresh_ratio)
        E = self.weight - alpha * q

        U, S, Vh = torch.linalg.svd(E, full_matrices=False)
        r = min(self.rank, S.numel())
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]

        L = U_r * S_r.unsqueeze(0)
        R = Vh_r

        self.L = L.detach()
        self.R = R.detach()
        self.alpha.data = alpha.detach()  # Initialize learnable alpha
        self.lr_scalars.data.fill_(1.0)

    def forward(self, x):
        q_nograd, _, _, _ = ternary_quantize(self.weight, self.thresh_ratio)
        q = ste_hard_replace(self.weight, q_nograd)
        w_q = self.alpha * q  # Use learnable alpha

        if self.rank > 0 and self.L.numel() != 0 and self.R.numel() != 0:
            E_lr = (self.lr_scalars * self.L) @ self.R
            w_hat = w_q + E_lr
        else:
            w_hat = w_q

        return F.linear(x, w_hat, self.bias)

    def layer_reg_loss(self):
        return self.lr_scalars.abs().mean()

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, rank={self.rank}, thresh_ratio={self.thresh_ratio}")
    


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