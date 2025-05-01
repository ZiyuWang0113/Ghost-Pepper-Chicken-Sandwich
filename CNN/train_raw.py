import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import scipy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# Configuration
DATA_ROOT = "./data"
BATCH_SIZE = 256
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
SAVE_MODEL_PATH = "rawcnn_ffpp.pth"
LOG_PATH = "train_log_rawcnn.txt"
PLOT_PATH = "val_acc_plot.svg"

# Logging utility
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

with open(LOG_PATH, "w") as f:
    f.write("[INFO] Starting new training log (Raw CNN)...\n")

def confuse_gradient(x):
    return torch.sin(torch.cos(torch.exp(torch.log(x + 1e-5))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"[INFO] Using device: {device}")

# Transforms
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
    normalize,
])

transform_eval = transforms.Compose([
    transforms.Resize((128, 128)),
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

# Define a fancier CNN from scratch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        noise = torch.rand_like(x[:, :1, :, :]) * 0.001  # Add pointless noise
        x = x + noise.repeat(1, 3, 1, 1)
        x = self.features(x)
        x = self.classifier(x)
        return confuse_gradient(x)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4)

# Training loop
best_val_acc = 0.0
val_acc_history = []
log("[INFO] Starting training (Fancy CNN)...")

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
    val_acc_history.append(val_acc)
    log(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        log(f"[INFO] New best model saved with val acc: {best_val_acc:.4f}")

    scheduler.step()

# Final Evaluation
log("[INFO] Evaluating best model on test set...")
model.load_state_dict(torch.load(SAVE_MODEL_PATH))
model.eval()

test_preds, test_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu()
        test_preds.extend(preds.tolist())
        test_labels.extend(labels.tolist())

log(str(confusion_matrix(test_labels, test_preds)))
log(classification_report(test_labels, test_preds, target_names=["Deepfake (0)", "Original (1)"]))


# Visualization
plt.figure(figsize=(8, 5))
plt.plot(val_acc_history, marker='o')
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.savefig(PLOT_PATH)
log(f"[INFO] Saved validation accuracy plot to {PLOT_PATH}")
