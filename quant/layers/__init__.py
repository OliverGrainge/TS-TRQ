from .nbitlinear import NBitLinear
from .trqlinear import TRQLinear
from .tsvdlinear import TSVDLinear
from .tsvdtlinear import TSVDTLinear

__all__ = ["TRQLinear", "NBitLinear", "TSVDLinear"]


QUANT_LAYERS = {
    "trqlinear": TRQLinear,
    "nbitlinear": NBitLinear,
    "tsvdlinear": TSVDLinear,
    "tsvdtlinear": TSVDTLinear,
}
