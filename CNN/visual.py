import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image

# Config
MODEL_PATH = "resnet18_ffpp.pth"
IMAGE_FOLDER = "./sample"
TARGET_LAYER = "layer4"

# Load Model
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_layer = getattr(model, TARGET_LAYER)
cam = GradCAM(model=model, target_layers=[target_layer])

# Visualize
def visualize(img_path, save_path=None):
    # Load and preprocess image
    raw_img = cv2.imread(img_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(raw_img, (256, 256))
    
    input_tensor = transform(Image.fromarray(img_resized)).unsqueeze(0).to(device)

    # Prediction
    outputs = model(input_tensor)
    pred_class = outputs.argmax().item()
    pred_score = torch.softmax(outputs, dim=1)[0][pred_class].item()

    # GradCAM (must use context manager)
    with cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]

    # Overlay GradCAM
    visualization = show_cam_on_image(img_resized / 255.0, grayscale_cam, use_rgb=True)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_resized)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(grayscale_cam, cmap='jet')
    axs[1].set_title(f"Prediction: {'Fake' if pred_class==0 else 'Real'} ({pred_score:.2f})\nFeature Importance Map")
    axs[1].axis('off')

    axs[2].imshow(visualization)
    axs[2].set_title("Importance Overlay")
    axs[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # === Saving output ===
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(IMAGE_FOLDER, f"{base_filename}_gradcam.png")  # Save to sample/ folder
    plt.savefig(save_path)
    print(f"[INFO] Saved GradCAM visualization to {save_path}")

    plt.close()

if __name__ == "__main__":
    for img_file in os.listdir(IMAGE_FOLDER):
        if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(IMAGE_FOLDER, img_file)
            visualize(img_path)
