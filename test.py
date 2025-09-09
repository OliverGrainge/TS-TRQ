from dotenv import load_dotenv

load_dotenv()
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "segmind/tiny-sd", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")
print(pipeline)
