import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# Configuration
DATA_ROOT = "./data"
BATCH_SIZE = 256
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
SAVE_MODEL_PATH = "resnet18_ffpp.pth"
LOG_PATH = "train_log.txt"

# Logging utility
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)  # Optional: remove if you don’t want stdout

# Clear previous log
with open(LOG_PATH, "w") as f:
    f.write("[INFO] Starting new training log...\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"[INFO] Using device: {device}")

# Normalization for ResNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    normalize,
])

# Only normalization for val/test
transform_eval = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    normalize,
])

# Data Loaders
train_dir = os.path.join(DATA_ROOT, "Train")
val_dir   = os.path.join(DATA_ROOT, "Validation")
test_dir  = os.path.join(DATA_ROOT, "Test")

train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset   = datasets.ImageFolder(val_dir, transform=transform_eval)
test_dataset  = datasets.ImageFolder(test_dir, transform=transform_eval)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model setup
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 2)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4)

# Training loop
best_val_acc = 0.0
log("[INFO] Starting training...")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / len(train_dataset)

    # Validation
    model.eval()
    val_correct = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_preds.extend(preds.cpu().tolist())
            val_labels.extend(labels.cpu().tolist())

    val_acc = val_correct / len(val_dataset)
    log(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        log(f"[INFO] New best model saved with val acc: {best_val_acc:.4f}")

    scheduler.step()

# Final Evaluation on Test Set
log("[INFO] Evaluating best model on test set...")
model.load_state_dict(torch.load(SAVE_MODEL_PATH))
model.eval()

misclassified = {0: [], 1: []}  # 0 = fake, 1 = real
test_preds, test_labels = [], []

# Need access to image paths
test_loader_with_paths = DataLoader(test_dataset, batch_size=BATCH_SIZE)
image_paths = [s[0] for s in test_dataset.samples]

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader_with_paths):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu()
        labels = labels.cpu()

        for j in range(len(preds)):
            pred, label = preds[j].item(), labels[j].item()
            test_preds.append(pred)
            test_labels.append(label)

            if pred != label and len(misclassified[label]) < 10:
                img_index = i * BATCH_SIZE + j
                if img_index < len(image_paths):
                    misclassified[label].append(image_paths[img_index])

# Save misclassified examples to a file
MISCLASSIFIED_PATH = "misclassified.txt"
with open(MISCLASSIFIED_PATH, "w") as f:
    f.write("Misclassified as FAKE (should be REAL):\n")
    for path in misclassified[1]:
        f.write(f"{path}\n")
    f.write("\nMisclassified as REAL (should be FAKE):\n")
    for path in misclassified[0]:
        f.write(f"{path}\n")
log(f"[INFO] Saved misclassified image list to {MISCLASSIFIED_PATH}")

# Report results
log(str(confusion_matrix(test_labels, test_preds)))
log(classification_report(test_labels, test_preds, target_names=["Deepfake (0)", "Original (1)"]))