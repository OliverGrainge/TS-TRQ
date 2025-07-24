from models import DiTModule 
from quant import quantize, linear_layer_names
import torch

def main(): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = DiTModule() 
    model = model.to(device)

    linear_names = linear_layer_names(model.transformer)

    model.transformer = quantize(model.transformer, layer_names=linear_names, quant_type="tsvdlinear", rank=192)
    model.transformer = model.transformer.to(device)

    img = model.sample(batch_size=1, class_labels=[10], num_steps=50, device=device, output_type="pil")

    # model.sample returns a list of PIL images if output_type="pil"
    if isinstance(img, list):
        img[0].save("tmp/test_image.png")
    else:
        img.save("tmp/test_image.png")


if __name__ == "__main__":
    main()  
