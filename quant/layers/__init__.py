from .tsvdlinear import TSVDLinear
from .tlinear import TLinear
from .tconv2d import TConv2d    
from .tsvdconv2d import TSVDConv2d

__all__ = ["TLinear", "TConv2d", "TSVDLinear", "TSVDConv2d"]


QUANT_LAYERS = {
    "tsvdlinear": TSVDLinear,
    "tlinear": TLinear,
    "tconv2d": TConv2d,
    "tsvdconv2d": TSVDConv2d,
}
