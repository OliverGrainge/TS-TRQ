from .nbitlinear import NBitLinear
from .trqlinear import TRQLinear

__all__ = ["TRQLinear", "NBitLinear"]


QUANT_LAYERS = {
    "trqlinear": TRQLinear,
    "nbitlinear": NBitLinear,
}
