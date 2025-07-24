import torch
import torch.nn as nn

@torch.no_grad()
def weight_quant(w, thresh_ratio=0.75):
    dtype = w.dtype
    abs_w = w.float().abs()
    delta = thresh_ratio * abs_w.mean()        # Δ = 0.75·E|w|
    mask  = (abs_w > delta)       # {0,1}
    q     = torch.sign(w) * mask               # {-1,0,1}

    nonzeros = mask.sum()
    if nonzeros == 0:                          # all weights were below Δ
        return q, w.new_tensor(0.)

    alpha = (abs_w * mask).sum() / nonzeros    # <— use |w|, not w
    return q.to(dtype), alpha.to(dtype)




class TRQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, n_residuals=2):
        super().__init__(in_features, out_features, bias)
        self.n_residuals = n_residuals
        self.quantized_weights = None
        self.scales = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, n_residuals=2):
        trq = cls(linear.in_features, linear.out_features, bias=(linear.bias is not None), n_residuals=n_residuals)
        trq.weight.data = linear.weight.data.clone()
        if trq.bias is not None and linear.bias is not None:
            trq.bias.data = linear.bias.data.clone()
        return trq

    def quantize_weights(self, w):
        r = w.clone()
        qweight_list = []
        scale_list = []

        for i in range(self.n_residuals):
            t, s = weight_quant(r)
            qweight_list.append(t)
            scale_list.append(s)
            r = r - (t * s)
        return qweight_list, scale_list

    def forward(self, x):
        
        qweights, scales = self.quantize_weights(self.weight)

        out = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)
        for qw, s in zip(qweights, scales):
            out += torch.nn.functional.linear(x, qw) * s

        if self.bias is not None:
            out += self.bias

        return out
    
        
    def __repr__(self):
        return f"TRQLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, n_residuals={self.n_residuals})"


if __name__ == "__main__":
    # Test to verify the implementation works
    in_features = 8
    out_features = 4
    batch_size = 3
    
    # Create original linear layer
    linear = nn.Linear(in_features, out_features, bias=True)
    x = torch.randn(batch_size, in_features)
    # Test with different numbers of residuals
    for n_residuals in [1, 2, 5, 10, 20]:
        print(f"\nTesting with {n_residuals} residuals:")

        # Create TRQ layer - FIXED: properly assign the returned value
        trq = TRQLinear.from_linear(linear, n_residuals=n_residuals)

        # Forward pass
        
        out_linear = linear(x)
        out_trq = trq(x)

        # Calculate difference
        diff = (out_linear - out_trq).abs().mean().item()
        max_diff = (out_linear - out_trq).abs().max().item()
        
        print(f"Mean absolute difference: {diff:.6f}")
        print(f"Max absolute difference: {max_diff:.6f}")
        
        # Check quantization effectiveness
        original_norm = linear.weight.norm().item()
        residual_norm = (linear.weight - sum(qw * s for qw, s in zip(*trq.quantize_weights(linear.weight)))).norm().item()
        print(f"Residual norm / Original norm: {residual_norm / original_norm:.6f}")