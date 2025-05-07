import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
import argparse
import matplotlib.gridspec as gridspec
import shap
from matplotlib.colors import LinearSegmentedColormap
import time
import signal
import sys
import json
from datetime import datetime
import gc
from sklearn.metrics import r2_score

# Parse command line arguments
parser = argparse.ArgumentParser(description='Improved SHAP explanation for deepfake images')
parser.add_argument('--model_path', type=str, default='./final_model', 
                    help='Path to the trained model directory')
parser.add_argument('--image_path', type=str, required=True, 
                    help='Path to the image to analyze')
parser.add_argument('--output_dir', type=str, default='./improved_shap_analysis', 
                    help='Directory to save SHAP visualization results')
parser.add_argument('--num_samples', type=int, default=2000, 
                    help='Number of perturbed samples to generate (increase for better results)')
parser.add_argument('--num_superpixels', type=int, default=50, 
                    help='Number of superpixels to segment the image into')
parser.add_argument('--compactness', type=float, default=10, 
                    help='Compactness parameter for SLIC segmentation')
parser.add_argument('--sigma', type=float, default=1, 
                    help='Width of Gaussian smoothing kernel for SLIC')
parser.add_argument('--analyze_both_classes', action='store_true',
                    help='Analyze explanations for both fake and real classes')
parser.add_argument('--background_size', type=int, default=20,
                    help='Number of background samples to use for SHAP')
parser.add_argument('--multi_gpu', action='store_true',
                    help='Enable multi-GPU processing')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for processing samples')
parser.add_argument('--resume_from', type=str, default=None,
                    help='Path to a checkpoint file to resume analysis from')
parser.add_argument('--low_memory_mode', action='store_true',
                    help='Enable low memory optimizations')
parser.add_argument('--image_size', type=int, default=224,
                    help='Image size for resizing before analysis')
parser.add_argument('--max_workers', type=int, default=4,
                    help='Maximum number of worker processes')
parser.add_argument('--checkpoint_interval', type=int, default=100,
                    help='Save checkpoints after this many samples')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Helper functions for checkpointing
def save_checkpoint(explainer, segments, shap_map, segment_values, pred_probs, class_idx, r2_score, filename):
    """Save intermediate results to avoid losing progress if job terminates"""
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'class_idx': class_idx,
        'segments': segments.tolist() if segments is not None else None,
        'segment_values': segment_values.tolist() if segment_values is not None else None,
        'prediction': pred_probs.tolist() if pred_probs is not None else None,
        'r2_score': r2_score if r2_score is not None else None
    }
    
    # Save numerical results
    np.savez(
        filename,
        segments=segments,
        shap_map=shap_map,
        segment_values=segment_values,
        prediction=pred_probs,
        r2_score=r2_score
    )
    
    # Save metadata
    with open(f"{filename}_meta.json", 'w') as f:
        json.dump(checkpoint, f)
    
    print(f"Checkpoint saved to {filename}")

def signal_handler(sig, frame):
    """Handle termination signals gracefully"""
    print("\nReceived termination signal. Saving current progress...")
    # Save state if global variables are available
    if 'current_explainer' in globals() and 'current_results' in globals():
        save_checkpoint(
            current_explainer,
            current_results.get('segments'),
            current_results.get('shap_map'),
            current_results.get('segment_values'),
            current_results.get('pred_probs'),
            current_results.get('class_idx'),
            current_results.get('r2_score'),
            os.path.join(args.output_dir, f"checkpoint_interrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
        )
    print("Exiting...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# Load the model and processor
print(f"Loading model from {args.model_path}")
try:
    image_processor = AutoImageProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageClassification.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully: {model.__class__.__name__}")
    print(f"Model config: {model.config.model_type}")
    # Get class names from the model config
    class_names = ["Fake", "Real"]  # Default if not available in the model
    if hasattr(model.config, 'id2label'):
        class_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    print(f"Classes: {class_names}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Create a custom colormap for visualizations
red_blue_cmap = LinearSegmentedColormap.from_list('red_blue', 
                                               [(0.8, 0, 0), (1, 1, 1), (0, 0, 0.8)])

# Improved SHAP class for deepfake detection
class ImprovedShapExplainer:
    def __init__(self, model, image_processor, device, 
                 num_samples=2000, num_superpixels=50, 
                 compactness=10, sigma=1, background_size=20,
                 multi_gpu=False, batch_size=32,
                 low_memory_mode=False, image_size=224,
                 max_workers=4, checkpoint_interval=100,
                 output_dir='.'):
        self.model = model
        self.image_processor = image_processor
        self.device = device
        self.num_samples = num_samples
        self.num_superpixels = num_superpixels
        self.compactness = compactness
        self.sigma = sigma
        self.background_size = background_size
        self.multi_gpu = multi_gpu
        self.batch_size = batch_size
        self.low_memory_mode = low_memory_mode
        self.image_size = image_size
        self.max_workers = max_workers
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = output_dir
        
        # Store image and segments as instance variables
        self.image = None
        self.segments = None
        self.segment_values = None
        self.max_segment = 0
        
        # Store R² values
        self.r2_score = None
        
        # Check available GPUs and adjust multi-GPU settings
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count() if self.multi_gpu else 1
            if self.multi_gpu and self.num_gpus > 1:
                print(f"Using {self.num_gpus} GPUs for SHAP computation")
                # Clone model to each GPU if using multiple GPUs
                self.models = {}
                for gpu_id in range(self.num_gpus):
                    self.models[gpu_id] = self._clone_model_to_device(gpu_id)
            else:
                self.multi_gpu = False
                print("Multi-GPU processing not available or only 1 GPU. Using single GPU.")
        else:
            self.multi_gpu = False
            self.num_gpus = 0
            print("CUDA not available. Using CPU.")
        
        # Apply low memory optimizations if requested
        if self.low_memory_mode:
            print("Low memory mode enabled with the following optimizations:")
            print(f" - Image size reduced to {self.image_size}x{self.image_size}")
            print(f" - Using smaller batch size: {self.batch_size}")
            print(f" - Memory-efficient SHAP explainer")
            print(f" - Incremental processing with checkpoints every {self.checkpoint_interval} samples")
    
    def _clone_model_to_device(self, gpu_id):
        """Clone the model to a specific GPU device"""
        try:
            # Clone the model
            new_model = type(self.model)(self.model.config)
            new_model.load_state_dict(self.model.state_dict())
            new_model = new_model.to(f'cuda:{gpu_id}')
            new_model.eval()
            return new_model
        except Exception as e:
            print(f"Error cloning model to GPU {gpu_id}: {e}")
            # Fall back to using the main model
            self.multi_gpu = False
            return self.model
        
    def segment_image(self, image):
        """Segment the image into superpixels using SLIC algorithm"""
        img_array = np.array(image)
        segments = slic(img_array, n_segments=self.num_superpixels, 
                        compactness=self.compactness, sigma=self.sigma, 
                        start_label=0)  # Start from 0 for easier indexing
        max_segment = np.max(segments)
        print(f"Image segmented into {max_segment + 1} superpixels")
        return segments, max_segment
    
    def apply_mask(self, mask):
        """Apply a mask to the image based on superpixel segments"""
        # Convert to numpy array
        img_array = np.array(self.image).copy()
        
        # Create a blurred/gray version of the image for masked areas
        gray_img = np.ones_like(img_array) * 128  # Mid-gray
        
        # Apply mask to the image
        for segment_id in range(self.max_segment + 1):
            if not mask[segment_id]:  # If this segment is masked
                segment_mask = (self.segments == segment_id)
                img_array[segment_mask] = gray_img[segment_mask]
        
        # Convert back to PIL
        return Image.fromarray(img_array.astype(np.uint8))
    
    def predict(self, masked_image):
        """Get model prediction for a masked image"""
        inputs = self.image_processor(images=masked_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        
        return probs
    
    def generate_shap_values(self, target_class):
        """Generate SHAP values for the image using a partition-based approach"""
        # The number of segments
        n_segments = self.max_segment + 1
        
        # Generate random background samples
        # Each row represents a random mask (0/1 for each segment)
        np.random.seed(42)  # For reproducibility
        background = np.random.randint(0, 2, size=(self.background_size, n_segments))
        
        # Make sure there's at least one sample with all segments included and one with none
        if self.background_size > 1:
            background[0] = np.ones(n_segments)  # All segments
            background[1] = np.zeros(n_segments)  # No segments
        
        # Generate test samples for calculating SHAP values
        test_sample = np.ones(n_segments)  # The sample we want to explain (all segments present)
        
        # Create a function that returns model predictions for masked images
        def f(masks):
            preds = []
            for mask in masks:
                masked_img = self.apply_mask(mask.astype(bool))
                pred = self.predict(masked_img)
                preds.append(pred[target_class])
            return np.array(preds)
        
        # Create SHAP explainer
        print("Initializing KernelExplainer...")
        explainer = shap.KernelExplainer(f, background)
        
        # Calculate SHAP values
        print(f"Computing SHAP values with {self.num_samples} samples...")
        shap_values = explainer.shap_values(test_sample, 
                                           nsamples=self.num_samples, 
                                           silent=False)
        
        # Calculate R² to measure how well the SHAP values explain the model's prediction
        # Get the predictions for all background samples plus the test sample
        all_samples = np.vstack([background, test_sample.reshape(1, -1)])
        actual_preds = f(all_samples)
        
        # Get the mean prediction (the expected value)
        expected_value = explainer.expected_value
        
        # Calculate predicted values using SHAP values
        predicted_preds = np.zeros_like(actual_preds)
        for i, sample in enumerate(all_samples):
            # The SHAP prediction is the expected value plus the contribution of each feature
            predicted_preds[i] = expected_value + np.sum(shap_values * sample)
        
        # Calculate R² using scikit-learn's r2_score
        self.r2_score = r2_score(actual_preds, predicted_preds)
        print(f"SHAP explanation quality (R²): {self.r2_score:.4f}")
        
        # Save the expected_value and shap_values
        self.expected_value = explainer.expected_value
        
        return shap_values
    
    def explain(self, image, target_class=None):
        """Generate a SHAP explanation for the image"""
        # Store the image
        self.image = image
        
        # Get initial prediction
        initial_prediction = self.predict(image)
        predicted_class = np.argmax(initial_prediction)
        
        # Set target class if not specified
        if target_class is None:
            target_class = predicted_class
            
        print(f"Explaining class: {target_class} (Probability: {initial_prediction[target_class]:.4f})")
        
        # Segment the image and store segments
        self.segments, self.max_segment = self.segment_image(image)
        
        # Generate SHAP values
        start_time = time.time()
        shap_values = self.generate_shap_values(target_class)
        end_time = time.time()
        print(f"SHAP calculation took {end_time - start_time:.2f} seconds")
        
        # Map SHAP values back to image segments for visualization
        segment_values = shap_values  # Already indexed by segment
        
        # Create a heatmap of SHAP values
        image_shape = np.array(image).shape[:2]
        shap_map = np.zeros(image_shape)
        for segment_id in range(self.max_segment + 1):
            shap_map[self.segments == segment_id] = segment_values[segment_id]
        
        return self.segments, shap_map, segment_values, predicted_class, initial_prediction
    
    def visualize_explanation(self, segments, shap_map, segment_values, 
                             class_name, probability, save_path=None):
        """Create visualization of the SHAP explanation"""
        img_array = np.array(self.image)
        
        # Create a figure with a grid layout
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.05])
        
        # 1. Original image
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(img_array)
        ax1.set_title('Original Image', fontsize=14)
        ax1.axis('off')
        
        # 2. Segmented image
        ax2 = plt.subplot(gs[0, 1])
        segmented_img = mark_boundaries(img_array, segments, color=(1, 1, 0))
        ax2.imshow(segmented_img)
        ax2.set_title(f'Segmentation ({self.max_segment + 1} superpixels)', fontsize=14)
        ax2.axis('off')
        
        # 3. SHAP heatmap
        ax3 = plt.subplot(gs[0, 2])
        
        # Get the absolute max for symmetric colormap
        vmax = np.max(np.abs(shap_map)) if np.max(np.abs(shap_map)) > 0 else 1
        heatmap = ax3.imshow(shap_map, cmap=red_blue_cmap, vmin=-vmax, vmax=vmax)
        ax3.set_title(f'SHAP Values Heatmap\nClass: {class_name}, Prob: {probability:.4f}, R²: {self.r2_score:.4f}', 
                      fontsize=14)
        ax3.axis('off')
        
        # Add a colorbar
        cax = plt.subplot(gs[1, :])
        cbar = plt.colorbar(heatmap, cax=cax, orientation='horizontal')
        cax.set_xlabel('SHAP Value (Blue: Negative, Red: Positive)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def create_overlay_visualization(self, shap_map, class_name, probability, 
                                   threshold=0.1, save_path=None):
        """Create an overlay visualization showing important regions on the original image"""
        img_array = np.array(self.image).astype(float) / 255.0
        
        # Get the absolute max for normalization
        vmax = np.max(np.abs(shap_map)) if np.max(np.abs(shap_map)) > 0 else 1
        
        # Normalize shap_map to [-1, 1]
        normalized_shap = shap_map / vmax if vmax > 0 else shap_map
        
        # Create mask for positive and negative contributions
        pos_mask = normalized_shap > threshold
        neg_mask = normalized_shap < -threshold
        
        # Create overlay image
        overlay = img_array.copy()
        alpha = 0.6  # Transparency
        
        # Apply red color for positive contributions
        if np.any(pos_mask):
            red_intensity = np.zeros_like(img_array)
            red_intensity[..., 0] = np.abs(normalized_shap) * pos_mask  # Red channel
            overlay = overlay * (1 - alpha * pos_mask[..., None]) + red_intensity * alpha
        
        # Apply blue color for negative contributions
        if np.any(neg_mask):
            blue_intensity = np.zeros_like(img_array)
            blue_intensity[..., 2] = np.abs(normalized_shap) * neg_mask  # Blue channel
            overlay = overlay * (1 - alpha * neg_mask[..., None]) + blue_intensity * alpha
        
        overlay = np.clip(overlay, 0, 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(overlay)
        ax.set_title(f'SHAP Explanation Overlay\nClass: {class_name}, Probability: {probability:.4f}, R²: {self.r2_score:.4f}', 
                     fontsize=14)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved overlay visualization to {save_path}")
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def create_force_plot(self, segment_values, class_name, save_path=None):
        """Create a force plot to show feature contributions"""
        # Sort segments by absolute SHAP value
        indices = np.argsort(np.abs(segment_values))[::-1]
        top_n = min(15, len(indices))  # Show at most 15 segments
        
        # Get feature names and sorted values
        feature_names = [f"Segment {i}" for i in indices[:top_n]]
        values = segment_values[indices[:top_n]]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['red' if x < 0 else 'blue' for x in values]
        y_pos = np.arange(len(feature_names))
        
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title(f'Top Segments Contributing to Class: {class_name}\nR² Score: {self.r2_score:.4f}')
        
        # Add a vertical line at x=0
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add value labels on the bars
        for i, v in enumerate(values):
            if v < 0:
                ax.text(v - 0.01, i, f'{v:.4f}', ha='right', va='center', color='white')
            else:
                ax.text(v + 0.01, i, f'{v:.4f}', ha='left', va='center', color='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved force plot to {save_path}")
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def visualize_top_regions(self, segments, segment_values, top_n=5, save_path=None):
        """Visualize the top influential regions"""
        img_array = np.array(self.image)
        
        # Get the most influential regions by absolute SHAP value
        abs_values = np.abs(segment_values)
        top_indices = np.argsort(abs_values)[::-1][:top_n]
        
        # Create a figure with the original image and highlighted regions
        fig, ax = plt.subplots(1, top_n + 1, figsize=(20, 5))
        
        # Original image with all superpixels
        segmented_img = mark_boundaries(img_array, segments, color=(1, 1, 0))
        ax[0].imshow(segmented_img)
        ax[0].set_title(f'All Superpixels\nExplanation R²: {self.r2_score:.4f}', fontsize=12)
        ax[0].axis('off')
        
        # Individual top regions
        for i, idx in enumerate(top_indices):
            segment_id = idx  # Segments are 0-indexed
            importance = segment_values[idx]
            
            # Create a mask for this segment
            mask = segments == segment_id
            
            # Create a visualization with just this segment highlighted
            highlighted = img_array.copy()
            
            # Add a colored overlay
            overlay = np.zeros_like(highlighted, dtype=float)
            if importance > 0:
                # Red for positive (contributes to deepfake)
                overlay[mask] = [255, 0, 0]
            else:
                # Blue for negative (against deepfake)
                overlay[mask] = [0, 0, 255]
            
            # Blend with original
            alpha = 0.6
            highlighted = highlighted * (1 - alpha) + overlay * alpha
            highlighted = np.clip(highlighted, 0, 255).astype(np.uint8)
            
            # Add boundaries
            highlighted = mark_boundaries(highlighted, mask.astype(int), color=(1, 1, 1))
            
            ax[i+1].imshow(highlighted)
            sign = "+" if importance > 0 else "-"
            ax[i+1].set_title(f'Region {segment_id}\nContribution: {sign}{abs_values[idx]:.4f}', fontsize=10)
            ax[i+1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved top regions visualization to {save_path}")
            plt.close()
        else:
            plt.show()
        
        return fig

def main():
    # Load and analyze the image
    try:
        # Load image
        print(f"Loading image from {args.image_path}")
        image = Image.open(args.image_path).convert('RGB')
        
        # Create base filename for outputs
        base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
        
        # Check if resuming from a checkpoint
        completed_classes = []
        if args.resume_from and os.path.exists(args.resume_from):
            print(f"Resuming from checkpoint: {args.resume_from}")
            # Load metadata if it exists
            meta_file = f"{args.resume_from}_meta.json"
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                print(f"Checkpoint from: {metadata.get('timestamp')}")
                completed_class_idx = metadata.get('class_idx')
                if completed_class_idx is not None:
                    completed_classes.append(completed_class_idx)
                    print(f"Already completed analysis for class: {class_names[completed_class_idx]}")
        
        # Initialize SHAP explainer with memory optimizations
        shap_explainer = ImprovedShapExplainer(
            model, 
            image_processor, 
            device, 
            num_samples=args.num_samples, 
            num_superpixels=args.num_superpixels,
            compactness=args.compactness,
            sigma=args.sigma,
            background_size=args.background_size,
            multi_gpu=args.multi_gpu,
            batch_size=args.batch_size,
            low_memory_mode=args.low_memory_mode,
            image_size=args.image_size,
            max_workers=args.max_workers,
            checkpoint_interval=args.checkpoint_interval,
            output_dir=args.output_dir
        )
        
        # Get the model's initial prediction
        initial_prediction = shap_explainer.predict(image)
        predicted_class_idx = np.argmax(initial_prediction)
        predicted_class_name = class_names[predicted_class_idx]
        predicted_prob = initial_prediction[predicted_class_idx]
        
        print(f"\nModel Prediction:")
        for i, (class_name, prob) in enumerate(zip(class_names, initial_prediction)):
            print(f"  {class_name}: {prob:.4f}" + (" (Predicted)" if i == predicted_class_idx else ""))
        
        # Classes to analyze
        classes_to_analyze = [predicted_class_idx]
        if args.analyze_both_classes:
            classes_to_analyze = list(range(len(class_names)))
        
        # Filter out completed classes if resuming
        if 'completed_classes' in locals():
            classes_to_analyze = [c for c in classes_to_analyze if c not in completed_classes]
            
            if not classes_to_analyze:
                print("All requested classes have already been analyzed. Nothing to do.")
                return
        
        # For each class, generate and visualize SHAP explanation
        for class_idx in classes_to_analyze:
            class_name = class_names[class_idx]
            class_prob = initial_prediction[class_idx]
            
            print(f"\n{'='*50}")
            print(f"Analyzing explanation for class: {class_name} (Probability: {class_prob:.4f})")
            print(f"{'='*50}")
            
            # Generate explanation
            segments, shap_map, segment_values, pred_class, pred_probs = shap_explainer.explain(
                image, target_class=class_idx)
            
            # Create output path for this class
            output_prefix = f"{base_filename}_{class_name.lower()}"
            
            # Set up current_explainer and current_results for checkpointing
            global current_explainer, current_results
            current_explainer = shap_explainer
            current_results = {
                'class_idx': class_idx,
                'segments': segments,
                'shap_map': shap_map,
                'segment_values': segment_values,
                'pred_probs': pred_probs,
                'r2_score': shap_explainer.r2_score
            }
            
            # Save a checkpoint
            checkpoint_path = os.path.join(args.output_dir, f"{output_prefix}_checkpoint.npz")
            save_checkpoint(
                shap_explainer,
                segments, 
                shap_map, 
                segment_values, 
                pred_probs,
                class_idx,
                shap_explainer.r2_score,
                checkpoint_path
            )
            
            # Visualize explanation
            shap_explainer.visualize_explanation(
                segments, shap_map, segment_values, 
                class_name, class_prob,
                save_path=os.path.join(args.output_dir, f"{output_prefix}_explanation.png")
            )
            
            # Create overlay visualization
            shap_explainer.create_overlay_visualization(
                shap_map, class_name, class_prob, threshold=0.1,
                save_path=os.path.join(args.output_dir, f"{output_prefix}_overlay.png")
            )
            
            # Create force plot
            shap_explainer.create_force_plot(
                segment_values, class_name,
                save_path=os.path.join(args.output_dir, f"{output_prefix}_force_plot.png")
            )
            
            # Find and print influential regions
            abs_values = np.abs(segment_values)
            top_indices = np.argsort(abs_values)[::-1][:5]
            
            print("\nMost Influential Regions:")
            for i, idx in enumerate(top_indices):
                print(f"  Rank {i+1}: Segment {idx}, " +
                      f"Contribution: {segment_values[idx]:.6f} " +
                      f"({'Positive' if segment_values[idx] > 0 else 'Negative'})")
            
            # Visualize top regions
            shap_explainer.visualize_top_regions(
                segments, segment_values, top_n=5,
                save_path=os.path.join(args.output_dir, f"{output_prefix}_top_regions.png")
            )
            
            # Save numerical results for further analysis
            np.savez(
                os.path.join(args.output_dir, f"{output_prefix}_data.npz"),
                segments=segments,
                shap_map=shap_map,
                segment_values=segment_values,
                prediction=pred_probs,
                r2_score=shap_explainer.r2_score
            )
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\nAnalysis complete! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save checkpoint if available
        if 'current_explainer' in globals() and 'current_results' in globals() and 'segments' in current_results:
            error_checkpoint = os.path.join(args.output_dir, f"error_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
            try:
                save_checkpoint(
                    current_explainer,
                    current_results.get('segments'),
                    current_results.get('shap_map'),
                    current_results.get('segment_values'),
                    current_results.get('pred_probs'),
                    current_results.get('class_idx'),
                    current_results.get('r2_score'),
                    error_checkpoint
                )
                print(f"Saved error checkpoint to {error_checkpoint}")
            except Exception as checkpoint_error:
                print(f"Failed to save error checkpoint: {checkpoint_error}")

if __name__ == "__main__":
    main()