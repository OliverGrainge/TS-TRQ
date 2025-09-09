import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
import pytorch_lightning as pl
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from pytorch_lightning.loggers import WandbLogger
from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
    LMSDiscreteScheduler,
)
from data import ImageNetDataModule
from quant import quantize_model, get_all_conv2d_names, get_all_linear_names  

load_dotenv()

from models import DiffusionModule 


model = DiffusionModule()
img = model.generate(height=128, width=128)
print(img)
img[0].save("tmp/test_image.png")