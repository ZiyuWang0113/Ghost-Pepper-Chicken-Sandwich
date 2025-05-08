import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from captum.attr import IntegratedGradients

MODEL_PATH = "resnet18_ffpp.pth"
IMAGE_FOLDER = "sample"
OUTPUT_FOLDER = "out"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


ig = IntegratedGradients(model)
for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(IMAGE_FOLDER, filename)
        raw_img = Image.open(img_path).convert("RGB")
        input_tensor = transform(raw_img).unsqueeze(0)
        input_tensor.requires_grad = True

        with torch.no_grad():
            pred_class = model(input_tensor).argmax(dim=1).item()
            prob = torch.nn.functional.softmax(model(input_tensor), dim=1)[0][pred_class].item()
            print(f"[INFO] {filename} -> {'Fake' if pred_class==0 else 'Real'} ({prob:.2f})")

        attr, delta = ig.attribute(input_tensor, target=pred_class, return_convergence_delta=True)
        attr = attr.squeeze().detach().numpy()
        attr = np.transpose(attr, (1, 2, 0))
        attr_gray = np.mean(attr, axis=2)
        attr_norm = (attr_gray - attr_gray.min()) / (attr_gray.max() - attr_gray.min() + 1e-8)

        # === Save plot ===
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(raw_img)
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(raw_img)
        axs[1].imshow(attr_norm, cmap="hot", alpha=0.5)
        axs[1].set_title("Integrated Gradients")
        axs[1].axis("off")
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_ig.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Saved visualization to {save_path}")
