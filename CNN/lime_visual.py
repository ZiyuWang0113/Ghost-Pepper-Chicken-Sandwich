import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from torchvision import models, transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries

# === Configuration ===
MODEL_PATH = "resnet18_ffpp.pth"
IMAGE_FOLDER = "./sample"
OUTPUT_FOLDER = "./output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Load Model ===
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# === Image Transform ===
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_tensor = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    normalize,
])

transform_raw = transforms.Compose([
    transforms.Resize((256, 256)),
])

# === Prediction Function for LIME ===
def batch_predict(images):
    model.eval()
    batch = torch.stack([transform_tensor(Image.fromarray(img)) for img in images], dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()

# === Visualization ===
def visualize_lime(img_path):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(transform_raw(img))

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_np, batch_predict, top_labels=1, hide_color=0, num_samples=1000)

    temp, mask = explanation.get_image_and_mask(label=explanation.top_labels[0], positive_only=True, hide_rest=False)
    result = mark_boundaries(temp / 255.0, mask)

    # Save the figure
    filename = os.path.basename(img_path)
    base = os.path.splitext(filename)[0]
    save_path = os.path.join(OUTPUT_FOLDER, f"{base}_lime.png")

    plt.imshow(result)
    plt.title(f"LIME - {base}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Saved LIME visualization to {save_path}")
    plt.close()

# === Run All ===
if __name__ == "__main__":
    for img_file in os.listdir(IMAGE_FOLDER):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(IMAGE_FOLDER, img_file)
            visualize_lime(img_path)
