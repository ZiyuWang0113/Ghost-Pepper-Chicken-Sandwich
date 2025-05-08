import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import cv2

# CONFIG
MODEL_PATH = "resnet18_ffpp.pth"
IMAGE_FOLDER = "sample"
OUTPUT_FOLDER = "out"
TARGET_LAYER = "layer4"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load Model
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

target_layer = dict([*model.named_modules()])[TARGET_LAYER]

# Score-CAM
def scorecam(model, target_layer, input_tensor, pred_class):
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    hook = target_layer.register_forward_hook(forward_hook)
    _ = model(input_tensor)
    hook.remove()

    activation_maps = activations[0].squeeze(0)  # shape: (C, H, W)
    cam = torch.zeros_like(activation_maps[0])   # shape: (H, W)

    for i in range(activation_maps.shape[0]):
        act = activation_maps[i]
        act_resized = torch.nn.functional.interpolate(
            act.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze()

        # Normalize
        act_norm = (act_resized - act_resized.min()) / (act_resized.max() - act_resized.min() + 1e-8)

        # Mask input
        masked_input = input_tensor.clone()
        for c in range(3):
            masked_input[0, c] *= act_norm

        with torch.no_grad():
            score = model(masked_input)[0, pred_class].item()

        cam += act * score  # still at (H, W)

    # Normalize & upscale to 224x224
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_up = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()

    return cam_up

# === Main loop for each image ===
for file in os.listdir(IMAGE_FOLDER):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(IMAGE_FOLDER, file)
        raw_img = Image.open(img_path).convert("RGB")
        input_tensor = transform(raw_img).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)
            pred_class = logits.argmax(dim=1).item()
            pred_prob = torch.softmax(logits, dim=1)[0][pred_class].item()

        cam_map = scorecam(model, target_layer, input_tensor, pred_class)
        raw_np = np.array(raw_img.resize((224, 224))) / 255.0

        overlay = raw_np.copy()
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * raw_np + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)

        # Plot and save
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(raw_np)
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(overlay)
        axs[1].set_title(f"Score-CAM ({'Real' if pred_class else 'Fake'}, {pred_prob:.2f})")
        axs[1].axis("off")

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(file)[0]}_scorecam.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Saved Score-CAM to {save_path}")
