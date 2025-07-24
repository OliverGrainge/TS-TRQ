import copy

import clip
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from models import get_model
from quant import quantize

# ---- 1. Load original model ----
print("Loading original DiT-XL model...")
pipe = get_model("dit-xl-2-256")

clip_model, preprocess_clip = clip.load(
    "ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"
)
device = clip_model.visual.conv1.weight.device
clip_model.eval()

# ---- 2. Setup generation config ----
imagenet_classes = {
    207: "Golden Retriever",
    208: "Labrador Retriever",
    281: "Tabby Cat",
    285: "Egyptian Cat",
    151: "Chihuahua",
}
# Use multiple categories for more robust evaluation
class_labels_list = [207, 208, 281, 285, 151]  # Multiple categories
generator = torch.manual_seed(45)
generation_kwargs = {
    "num_inference_steps": 50,
    "generator": generator,
    "guidance_scale": 4.0,
}


# ---- 3. Image error metric ----
def compute_clip_similarity(image1, image2):
    image1_tensor = preprocess_clip(image1).unsqueeze(0).to(device)
    image2_tensor = preprocess_clip(image2).unsqueeze(0).to(device)
    with torch.no_grad():
        emb1 = clip_model.encode_image(image1_tensor).float()
        emb2 = clip_model.encode_image(image2_tensor).float()
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)
        similarity = (emb1 @ emb2.T).item()
    return similarity


# ---- 4. Helper to extract linear layers to quantize ----
def get_layers(model):
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "ff" in name and "proj" in name:
            layer_names.append(name)
    return layer_names


# ---- 5. Helper to quantize, generate, and compare ----
def evaluate_quantization(pipe, quant_type, param_name, param_value, n_samples=3):
    pipe_q = copy.deepcopy(pipe)
    layers = get_layers(pipe_q.transformer)
    kwargs = {"quant_type": quant_type, "layer_names": layers, param_name: param_value}
    pipe_q.transformer = quantize(pipe_q.transformer, **kwargs)

    # Generate multiple samples and average CLIP scores
    total_similarity = 0.0
    for i in range(n_samples):
        # Use different class for each sample
        class_label = class_labels_list[i % len(class_labels_list)]
        sample_kwargs = {**generation_kwargs, "class_labels": [class_label]}

        output = pipe_q(**sample_kwargs)
        image_q = output.images[0]

        # Get corresponding original image
        output_original = pipe(**sample_kwargs)
        image_original = output_original.images[0]

        similarity = compute_clip_similarity(image_original, image_q)
        total_similarity += similarity

    return total_similarity / n_samples


# ---- 6. Run evaluations ----
bitwidths = [2, 4, 6, 8]
residual_counts = [1, 2, 3, 4, 5, 6]

clip_scores_nbit = []
clip_scores_trq = []

print("\nEvaluating nbitlinear quantization:")
for n_bits in bitwidths:
    clip_score = evaluate_quantization(pipe, "nbitlinear", "n_bits", n_bits)
    clip_scores_nbit.append(clip_score)
    print(f"  n_bits = {n_bits}: CLIP Score = {clip_score:.6f}")

print("\nEvaluating trqlinear quantization:")
for n_res in residual_counts:
    clip_score = evaluate_quantization(pipe, "trqlinear", "n_residuals", n_res)
    clip_scores_trq.append(clip_score)
    print(f"  n_residuals = {n_res}: CLIP Score = {clip_score:.6f}")

# Convert trqlinear residuals to equivalent bitwidth (n residuals = n+2 bits)
trq_bitwidths = [n_res * 2 for n_res in residual_counts]

# ---- 7. Plot results ----
fig, ax = plt.subplots(figsize=(10, 6))

# Plot both methods on the same x-axis
ax.plot(
    bitwidths,
    clip_scores_nbit,
    marker="o",
    label="nbitlinear",
    linewidth=2,
    color="blue",
)
ax.plot(
    trq_bitwidths,
    clip_scores_trq,
    marker="s",
    label="trqlinear",
    linewidth=2,
    color="red",
)

ax.set_xlabel("n_bits")
ax.set_ylabel("CLIP Similarity to Original Image")
ax.set_title(
    "CLIP Similarity Comparison: nbitlinear vs trqlinear\n(Averaged across multiple categories)"
)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/image_quant_error.png")
plt.show()

print("\n✅ Plot saved as 'plots/image_quant_error.png'")
print(
    f"📊 Evaluation averaged across {len(class_labels_list)} categories with 3 samples each"
)
