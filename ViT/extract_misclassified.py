import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import shutil
from tqdm import tqdm
import argparse
import numpy as np
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

set_seed()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract misclassified images from trained model')
parser.add_argument('--data_root', type=str, default='./Dataset', 
                    help='Root directory for dataset')
parser.add_argument('--model_path', type=str, default='./final_model', 
                    help='Path to trained model directory')
parser.add_argument('--output_dir', type=str, default='./misclassified', 
                    help='Directory to save misclassified images')
parser.add_argument('--batch_size', type=int, default=16, 
                    help='Batch size for inference')
parser.add_argument('--num_workers', type=int, default=4, 
                    help='Number of worker threads for dataloader')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'false_negatives'), exist_ok=True)  # Fake classified as Real
os.makedirs(os.path.join(args.output_dir, 'false_positives'), exist_ok=True)  # Real classified as Fake

# Dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='Test', image_processor=None, transform=None):
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
            
            # Apply transform if available
            if self.transform:
                image = self.transform(image)
            
            # Apply image processor
            if self.image_processor:
                inputs = self.image_processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].squeeze(0)
            else:
                # Fallback
                from torchvision import transforms
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

def main():
    # Configure device for Apple Silicon (M4)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load the trained model and processor
    print(f"Loading model from {args.model_path}...")
    image_processor = AutoImageProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    # Create combined dataset (we're not separating validation and test)
    # First check if data directories exist
    test_dir = os.path.join(args.data_root, 'Test')
    val_dir = os.path.join(args.data_root, 'Validation')
    
    datasets = []
    if os.path.exists(test_dir):
        test_dataset = DeepfakeDataset(args.data_root, 'Test', image_processor=image_processor)
        datasets.append(test_dataset)
    
    if os.path.exists(val_dir):
        val_dataset = DeepfakeDataset(args.data_root, 'Validation', image_processor=image_processor)
        datasets.append(val_dataset)
    
    if not datasets:
        raise ValueError(f"No Test or Validation directories found in {args.data_root}")
    
    # Extract misclassified images from each dataset
    total_false_positives = 0
    total_false_negatives = 0
    
    for dataset in datasets:
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Finding misclassified images")):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(pixel_values=pixel_values)
                predictions = outputs.logits.argmax(dim=-1)
                
                # Find misclassified images
                misclassified_mask = (predictions != labels)
                misclassified_indices = misclassified_mask.nonzero(as_tuple=True)[0]
                
                for idx in misclassified_indices:
                    global_idx = batch_idx * dataloader.batch_size + idx.item()
                    img_path = dataset.image_paths[global_idx]
                    true_label = labels[idx].item()
                    pred_label = predictions[idx].item()
                    
                    # Calculate confidence score
                    confidence = torch.softmax(outputs.logits[idx], dim=0)[pred_label].item()
                    
                    # Determine the type of error and destination directory
                    if true_label == 0 and pred_label == 1:  
                        # False Negative: Fake classified as Real
                        dest_dir = os.path.join(args.output_dir, 'false_negatives')
                        total_false_negatives += 1
                    else:  
                        # False Positive: Real classified as Fake
                        dest_dir = os.path.join(args.output_dir, 'false_positives')
                        total_false_positives += 1
                    
                    # Create filename with metadata
                    original_class = "Fake" if true_label == 0 else "Real"
                    predicted_class = "Fake" if pred_label == 0 else "Real"
                    filename = f"{original_class}_as_{predicted_class}_conf_{confidence:.4f}_{os.path.basename(img_path)}"
                    dest_path = os.path.join(dest_dir, filename)
                    
                    # Copy the file
                    try:
                        shutil.copy(img_path, dest_path)
                    except Exception as e:
                        print(f"Error copying {img_path}: {e}")
    
    print("\nMisclassification Summary:")
    print(f"False Positives (Real classified as Fake): {total_false_positives}")
    print(f"False Negatives (Fake classified as Real): {total_false_negatives}")
    print(f"Total misclassified images: {total_false_positives + total_false_negatives}")
    print(f"All misclassified images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()