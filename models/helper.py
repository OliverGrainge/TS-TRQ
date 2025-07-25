import os
from typing import Optional

import torch


def get_model(
    model_name: str,
    dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,
    local_path: Optional[str] = None,
    force_safe: bool = False,
):
    """
    Load a DiT model pipeline.

    Args:
        model_name: currently only "dit-xl-2-256".
        dtype: torch dtype to load weights in.
        device: e.g. "cuda", "cpu", or "cuda:0". Auto if None.
        local_path: path to an already-downloaded HF snapshot (optional).
        force_safe: if True, require safetensors (allow_pickle=False). If False, let .bin load.
    """
    if model_name.lower() != "dit-xl-2-256":
        raise ValueError(f"Model {model_name} not found")

    from diffusers import DiTPipeline, DPMSolverMultistepScheduler

    repo_or_path = local_path or "facebook/DiT-XL-2-256"
    kwargs = dict(
        torch_dtype=dtype,
        cache_dir=os.getenv("HF_HUB_CACHE"),
        local_files_only=bool(local_path),  # only look locally if you gave a path
    )

    if force_safe:
        # make it blow up if only .bin exists (good for CI / security)
        kwargs["allow_pickle"] = False
    else:
        # quiet the warning and accept .bin
        kwargs["use_safetensors"] = False

    pipe = DiTPipeline.from_pretrained(repo_or_path, **kwargs)

    # Replace scheduler (your original code)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Move to device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    return pipe
