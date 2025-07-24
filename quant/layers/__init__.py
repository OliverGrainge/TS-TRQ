from .nbitlinear import NBitLinear
from .trqlinear import TRQLinear
from .tsvdlinear import TSVDLinear

__all__ = ["TRQLinear", "NBitLinear", "TSVDLinear"]


QUANT_LAYERS = {
    "trqlinear": TRQLinear,
    "nbitlinear": NBitLinear,
    "tsvdlinear": TSVDLinear,
}
