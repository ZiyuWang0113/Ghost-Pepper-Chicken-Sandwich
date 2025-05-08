import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import models, transforms
from matplotlib import gridspec

# === Configuration ===
MODEL_PATH = "resnet18_ffpp.pth"
IMAGE_FOLDER = "./sample"
OUTPUT_FOLDER = "./output"
R2_LOG_PATH = os.path.join(OUTPUT_FOLDER, "lime_r2_scores.txt")
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

# === Transforms ===
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

transform_raw = transforms.Compose([
    transforms.Resize((224, 224)),
])

# === Prediction Function ===
def batch_predict(images):
    batch = torch.stack([transform_tensor(Image.fromarray(img)) for img in images], dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()

# === LIME Visualization Function ===
def visualize_lime(img_path, top_regions=5):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(transform_raw(img))

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np, batch_predict, top_labels=1, hide_color=0, num_samples=1000
    )

    label = explanation.top_labels[0]
    probs = batch_predict([img_np])[0]
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]
    is_correct = (label == pred_class)
    segments = explanation.segments
    weights = explanation.local_exp[label]
    r2 = explanation.score

    # Plot LIME
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, top_regions + 1)

    # Superpixel view
    ax = fig.add_subplot(gs[0])
    ax.imshow(mark_boundaries(img_np, segments))
    ax.set_title("All Superpixels")
    ax.axis("off")

    # Top region overlays
    top_segments = sorted(weights, key=lambda x: abs(x[1]), reverse=True)[:top_regions]
    for i, (seg_id, weight) in enumerate(top_segments):
        mask = segments == seg_id
        highlighted = img_np.copy().astype(float)
        overlay = np.zeros_like(highlighted)
        if weight > 0:
            overlay[mask] = [0, 255, 0]  # green
        else:
            overlay[mask] = [255, 0, 0]  # red
        alpha = 0.5
        composite = np.clip((1 - alpha) * highlighted + alpha * overlay, 0, 255).astype(np.uint8)

        ax = fig.add_subplot(gs[i + 1])
        ax.imshow(mark_boundaries(composite, mask.astype(int)))
        ax.set_title(f"Region {seg_id}\nImportance: {weight:+.4f}")
        ax.axis("off")

    base = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(OUTPUT_FOLDER, f"{base}_lime_regions.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved enhanced LIME visualization to {save_path}")

    avg_weight = np.mean([abs(w) for (_, w) in explanation.local_exp[label]])
    with open(R2_LOG_PATH, "a") as f:
        f.write(f"{base}: R^2 Score = {r2:.4f}, Confidence = {confidence:.4f}, "
                f"Correct = {is_correct}, AvgInfluence = {avg_weight:.4f}\n")

# === Run All ===
if __name__ == "__main__":
    with open(R2_LOG_PATH, "w") as f:
        f.write("LIME R^2 Scores per Image:\n")

    for img_file in os.listdir(IMAGE_FOLDER):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(IMAGE_FOLDER, img_file)
            visualize_lime(img_path)
