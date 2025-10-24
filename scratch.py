
from dotenv import load_dotenv 
load_dotenv()
from models import DiffusionModule

model = DiffusionModule(model_id="google/ddpm-cifar10-32")
print(model)