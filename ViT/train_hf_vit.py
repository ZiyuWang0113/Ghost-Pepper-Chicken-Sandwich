import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import random
import argparse
import time

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Hugging Face ViT model for deepfake detection')
parser.add_argument('--data_root', type=str, default='/oscar/scratch/rgao44/Dataset', 
                    help='Root directory for dataset')
parser.add_argument('--output_dir', type=str, default='./output', 
                    help='Directory to save outputs')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=20, 
                    help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=1e-5, 
                    help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, 
                    help='Weight decay for optimizer')
parser.add_argument('--num_workers', type=int, default=4, 
                    help='Number of worker threads for dataloader')
parser.add_argument('--checkpoint', type=str, default=None, 
                    help='Path to checkpoint to resume from')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Dataset class adapted for Hugging Face
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='Train', image_processor=None, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.image_processor = image_processor
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Class 0: Fake, Class 1: Real
        class_mapping = {'Fake': 0, 'Real': 1}
        
        for class_name, class_idx in class_mapping.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)
        
        print(f"Loaded {len(self.image_paths)} images for {split} split")
        print(f"Class distribution - Fake: {self.labels.count(0)}, Real: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            # Apply traditional augmentations first if in training mode
            if self.transform:
                image = self.transform(image)
            
            # Apply Hugging Face image processor
            if self.image_processor:
                inputs = self.image_processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].squeeze(0)
            else:
                # Fallback if no processor provided
                pixel_values = transforms.ToTensor()(image)
            
            return {'pixel_values': pixel_values, 'labels': label}
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder
            if self.image_processor:
                placeholder = Image.new('RGB', (224, 224), color='black')
                inputs = self.image_processor(images=placeholder, return_tensors="pt")
                pixel_values = inputs['pixel_values'].squeeze(0)
            else:
                pixel_values = torch.zeros(3, 224, 224)
            return {'pixel_values': pixel_values, 'labels': self.labels[idx]}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the pretrained model and processor
    model_name = 'ashish-001/deepfake-detection-using-ViT'
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    # Custom transforms for training
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ])
    
    # No transforms for validation (processor will handle resizing)
    val_transforms = None
    
    # Create datasets with updated directory structure
    train_dataset = DeepfakeDataset(args.data_root, 'Train', image_processor=image_processor, transform=train_transforms)
    val_dataset = DeepfakeDataset(args.data_root, 'Validation', image_processor=image_processor, transform=val_transforms)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size // max(1, torch.cuda.device_count()),
        per_device_eval_batch_size=args.batch_size // max(1, torch.cuda.device_count()),
        warmup_steps=500,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=args.num_workers,
        learning_rate=args.learning_rate,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Resume from checkpoint if provided
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Resuming from checkpoint: {args.checkpoint}")
        trainer.train(resume_from_checkpoint=args.checkpoint)
    else:
        # Train from scratch
        trainer.train()
    
    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, 'final_model'))
    
    # Evaluate on validation set
    eval_results = trainer.evaluate()
    
    # Print evaluation results
    print("Evaluation Results:")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, 'eval_results.txt'), 'w') as f:
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")
    
    # Generate confusion matrix
    predictions = trainer.predict(val_dataset)
    y_true = predictions.label_ids
    y_pred = predictions.predictions.argmax(-1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Also evaluate on test set
    print("Evaluating on test set...")
    test_dataset = DeepfakeDataset(args.data_root, 'Test', image_processor=image_processor, transform=val_transforms)
    test_results = trainer.predict(test_dataset)
    
    # Calculate test metrics
    y_true_test = test_results.label_ids
    y_pred_test = test_results.predictions.argmax(-1)
    
    accuracy_test = accuracy_score(y_true_test, y_pred_test)
    precision_test = precision_score(y_true_test, y_pred_test, average='weighted')
    recall_test = recall_score(y_true_test, y_pred_test, average='weighted', zero_division=0)
    f1_test = f1_score(y_true_test, y_pred_test, average='weighted')
    
    print("Test Results:")
    print(f"Accuracy: {accuracy_test:.4f}")
    print(f"Precision: {precision_test:.4f}")
    print(f"Recall: {recall_test:.4f}")
    print(f"F1 Score: {f1_test:.4f}")
    
    # Save test results
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy_test}\n")
        f.write(f"Precision: {precision_test}\n")
        f.write(f"Recall: {recall_test}\n")
        f.write(f"F1 Score: {f1_test}\n")
    
    # Generate test confusion matrix
    cm_test = confusion_matrix(y_true_test, y_pred_test)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Set Confusion Matrix')
    plt.savefig(os.path.join(args.output_dir, 'test_confusion_matrix.png'))
    plt.close()
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()