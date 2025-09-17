from dotenv import load_dotenv

load_dotenv() 
import torch 
from models import LatentDiffusionModule, PixelDiffusionModule

from quant.layers import TSVDLinear
import torch.nn as nn 

layer = nn.Linear(256, 128, bias=True)
tsvd_layer = TSVDLinear.from_linear(layer, rank=8)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256, bias=True)
        # The following line is incorrect:
        # self.fc2 = TSVDLinear.from_linear(self.fc1, rank=8)
        # This tries to create a TSVDLinear from fc1, but fc1 is 128->256, while fc2 should be 256->256.
        # Instead, you should create a new nn.Linear(256, 256) and pass that to TSVDLinear.from_linear.
        fc2 = nn.Linear(256, 256, bias=True)
        self.fc2 = TSVDLinear.from_linear(fc2, rank=8)
        self.fc3 = nn.Linear(256, 128, bias=True)
        self.fc4 = nn.Linear(128, 64, bias=True)
        self.fc5 = nn.Linear(64, 10, bias=True)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


model = Model()
x = torch.randn(16, 128)
model.eval()
y = model(x)
print(y.shape)
