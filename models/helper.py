import torch


def get_model(model_name: str, dtype=torch.float16):
    if model_name.lower() == "dit-xl-2-256":
        from diffusers import DiTPipeline, DPMSolverMultistepScheduler

        pipe = DiTPipeline.from_pretrained(
            "facebook/DiT-XL-2-256", torch_dtype=torch.float16
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        return pipe
    else:
        raise ValueError(f"Model {model_name} not found")
