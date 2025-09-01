from .tsvdlinear import TSVDLinear

__all__ = ["TRQLinear", "NBitLinear", "TSVDLinear"]


QUANT_LAYERS = {
    "tsvdlinear": TSVDLinear,
}
