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
from sklearn.linear_model import Ridge
import argparse
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

# Parse command line arguments
parser = argparse.ArgumentParser(description='LIME explanation for a single deepfake image')
parser.add_argument('--model_path', type=str,default='./final_model', 
                    help='Path to the trained model directory')
parser.add_argument('--image_path', type=str, required=True, 
                    help='Path to the image to analyze')
parser.add_argument('--output_dir', type=str, default='./lime_analysis', 
                    help='Directory to save LIME visualization results')
parser.add_argument('--num_samples', type=int, default=1000, 
                    help='Number of perturbed samples to generate')
parser.add_argument('--num_superpixels', type=int, default=50, 
                    help='Number of superpixels to segment the image into')
parser.add_argument('--compactness', type=float, default=10, 
                    help='Compactness parameter for SLIC segmentation')
parser.add_argument('--sigma', type=float, default=1, 
                    help='Width of Gaussian smoothing kernel for SLIC')
parser.add_argument('--analyze_both_classes', action='store_true',
                    help='Analyze explanations for both fake and real classes')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

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

# LIME class for deepfake detection
class LimeExplainer:
    def __init__(self, model, image_processor, device, 
                 num_samples=1000, num_superpixels=50, 
                 compactness=10, sigma=1):
        self.model = model
        self.image_processor = image_processor
        self.device = device
        self.num_samples = num_samples
        self.num_superpixels = num_superpixels
        self.compactness = compactness
        self.sigma = sigma
    
    def segment_image(self, image):
        """Segment the image into superpixels"""
        img_array = np.array(image)
        segments = slic(img_array, n_segments=self.num_superpixels, 
                        compactness=self.compactness, sigma=self.sigma, 
                        start_label=1)
        print(f"Image segmented into {np.max(segments)} superpixels")
        return segments
    
    def perturb_image(self, image, segments, perturb_mask):
        """Create a perturbed version of the image by hiding some segments"""
        # Convert to numpy array for manipulation
        img_array = np.array(image).copy()
        
        # Create a grayscale version for replacement (mean across channels)
        gray_img = np.mean(img_array, axis=2, keepdims=True).repeat(3, axis=2)
        
        # For segments where perturb_mask is False, replace with gray
        for segment_id in range(1, np.max(segments) + 1):
            if not perturb_mask[segment_id-1]:  # -1 because segments start at 1
                img_array[segments == segment_id] = gray_img[segments == segment_id]
        
        # Convert back to PIL image
        perturbed_img = Image.fromarray(img_array.astype(np.uint8))
        return perturbed_img
    
    def get_model_prediction(self, image):
        """Get the model's prediction for a single image"""
        # Process image for the model
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            
        return probabilities
    
    def explain(self, image, target_class=None):
        """Generate a LIME explanation for the image"""
        # Get initial prediction to use as target class if not specified
        initial_prediction = self.get_model_prediction(image)
        predicted_class = np.argmax(initial_prediction)
        if target_class is None:
            target_class = predicted_class
            
        print(f"Explaining class: {target_class} (Probability: {initial_prediction[target_class]:.4f})")
        
        # Segment the image
        segments = self.segment_image(image)
        num_segments = np.max(segments)
        
        # Generate perturbed samples
        perturbed_data = []
        predictions = []
        
        for i in tqdm(range(self.num_samples), desc=f"Generating samples for class {target_class}"):
            # Randomly turn segments on or off
            perturb_mask = np.random.randint(0, 2, num_segments, dtype=bool)
            
            # Generate the perturbed image
            perturbed_img = self.perturb_image(image, segments, perturb_mask)
            
            # Get model prediction for the perturbed image
            prediction = self.get_model_prediction(perturbed_img)
            
            # Store the sample and prediction
            perturbed_data.append(perturb_mask)
            predictions.append(prediction[target_class])
            
            # Save a few perturbed images as examples
            if i < 5 and i % 50 == 0:
                perturbation_path = os.path.join(args.output_dir, f"perturbation_class{target_class}_sample{i}.png")
                perturbed_img.save(perturbation_path)
        
        # Convert to numpy arrays
        perturbed_data = np.array(perturbed_data)
        predictions = np.array(predictions)
        
        # Train a ridge regression model to explain the predictions
        explainer = Ridge(alpha=1.0)
        explainer.fit(perturbed_data, predictions)
        
        # Get feature importance scores (coefficients of the linear model)
        feature_importance = explainer.coef_
        
        # Create a heatmap of feature importances
        segments_importance = np.zeros(image.size[::-1], dtype=np.float32)
        for segment_id in range(1, num_segments + 1):
            segments_importance[segments == segment_id] = feature_importance[segment_id - 1]
        
        # Calculate R² of the linear model to measure explanation quality
        from sklearn.metrics import r2_score
        y_pred = explainer.predict(perturbed_data)
        r2 = r2_score(predictions, y_pred)
        print(f"Explanation quality (R²): {r2:.4f}")
        
        return segments, segments_importance, feature_importance, r2, predicted_class, initial_prediction
    
    def visualize_explanation(self, image, segments, segments_importance, 
                             class_label, probability, r2, save_path=None):
        """Create a detailed visualization of the LIME explanation"""
        img_array = np.array(image)
        
        # Normalize importance scores to [-1, 1] range for visualization
        if np.max(np.abs(segments_importance)) > 0:
            segments_importance = segments_importance / np.max(np.abs(segments_importance))
        
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
        ax2.imshow(mark_boundaries(img_array, segments))
        ax2.set_title(f'Segmentation ({np.max(segments)} superpixels)', fontsize=14)
        ax2.axis('off')
        
        # 3. LIME explanation (heatmap)
        ax3 = plt.subplot(gs[0, 2])
        
        # Create a custom colormap for the heatmap
        cmap = plt.cm.RdYlGn  # Red (negative) to green (positive)
        heatmap = ax3.imshow(segments_importance, cmap=cmap, vmin=-1, vmax=1)
        ax3.set_title(f'Feature Importance Heatmap\nClass: {class_label}, Prob: {probability:.4f}, R²: {r2:.4f}', 
                      fontsize=14)
        ax3.axis('off')
        
        # Add a colorbar
        cax = plt.subplot(gs[1, :])
        plt.colorbar(heatmap, cax=cax, orientation='horizontal')
        cax.set_xlabel('Feature Importance (Red: Negative, Green: Positive)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def create_overlay_visualization(self, image, segments, segments_importance, 
                                    class_label, probability, threshold=0.0, save_path=None):
        """Create an overlay visualization showing important regions on the original image"""
        img_array = np.array(image).astype(float) / 255.0
        
        # Normalize importance scores
        if np.max(np.abs(segments_importance)) > 0:
            segments_importance = segments_importance / np.max(np.abs(segments_importance))
        
        # Create mask for positive and negative contributions
        pos_mask = segments_importance > threshold
        neg_mask = segments_importance < -threshold
        
        # Create overlay image
        overlay = img_array.copy()
        alpha = 0.5  # Transparency
        
        # Apply green color for positive contributions
        if np.any(pos_mask):
            green_intensity = np.zeros_like(img_array)
            green_intensity[..., 1] = np.abs(segments_importance) * pos_mask  # Green channel
            overlay = overlay * (1 - alpha * pos_mask[..., None]) + green_intensity * alpha
        
        # Apply red color for negative contributions
        if np.any(neg_mask):
            red_intensity = np.zeros_like(img_array)
            red_intensity[..., 0] = np.abs(segments_importance) * neg_mask  # Red channel
            overlay = overlay * (1 - alpha * neg_mask[..., None]) + red_intensity * alpha
        
        overlay = np.clip(overlay, 0, 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(overlay)
        ax.set_title(f'LIME Explanation Overlay\nClass: {class_label}, Probability: {probability:.4f}', 
                     fontsize=14)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved overlay visualization to {save_path}")
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def find_most_influential_regions(self, segments, feature_importance, top_n=5):
        """Find the most influential superpixels for the prediction"""
        # Get absolute importance for ranking
        abs_importance = np.abs(feature_importance)
        
        # Get indices of top influential superpixels
        top_indices = np.argsort(abs_importance)[::-1][:top_n]
        
        # Get corresponding importance values
        top_importance = feature_importance[top_indices]
        
        # Get sign (positive or negative influence)
        top_signs = np.sign(top_importance)
        
        # Create a list of superpixel information
        influential_regions = []
        for i, idx in enumerate(top_indices):
            # Segment ID is index + 1 because SLIC starts at 1
            segment_id = idx + 1
            importance = top_importance[i]
            sign = "Positive" if top_signs[i] > 0 else "Negative"
            influential_regions.append({
                "rank": i + 1,
                "segment_id": segment_id,
                "importance": importance,
                "influence": sign
            })
        
        return influential_regions
    
    def visualize_top_regions(self, image, segments, feature_importance, top_n=5, save_path=None):
        """Visualize the top influential regions"""
        img_array = np.array(image)
        
        # Get the most influential regions
        abs_importance = np.abs(feature_importance)
        top_indices = np.argsort(abs_importance)[::-1][:top_n]
        
        # Create a figure with the original image and highlighted regions
        fig, ax = plt.subplots(1, top_n + 1, figsize=(20, 5))
        
        # Original image with all superpixels
        ax[0].imshow(mark_boundaries(img_array, segments))
        ax[0].set_title('All Superpixels', fontsize=12)
        ax[0].axis('off')
        
        # Individual top regions
        for i, idx in enumerate(top_indices):
            segment_id = idx + 1  # SLIC starts at 1
            importance = feature_importance[idx]
            
            # Create a mask for this segment
            mask = segments == segment_id
            
            # Create a visualization with just this segment highlighted
            highlighted = img_array.copy()
            
            # Add a colored overlay
            overlay = np.zeros_like(highlighted, dtype=float)
            if importance > 0:
                # Green for positive
                overlay[mask] = [0, 255, 0]
            else:
                # Red for negative
                overlay[mask] = [255, 0, 0]
            
            # Blend with original
            alpha = 0.5
            highlighted = highlighted * (1 - alpha) + overlay * alpha
            highlighted = np.clip(highlighted, 0, 255).astype(np.uint8)
            
            # Add boundaries
            highlighted = mark_boundaries(highlighted, mask.astype(int), color=(1, 1, 1))
            
            ax[i+1].imshow(highlighted)
            sign = "+" if importance > 0 else "-"
            ax[i+1].set_title(f'Region {segment_id}\nImportance: {sign}{abs_importance[idx]:.4f}', fontsize=10)
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
        
        # Initialize LIME explainer
        lime_explainer = LimeExplainer(
            model, 
            image_processor, 
            device, 
            num_samples=args.num_samples, 
            num_superpixels=args.num_superpixels,
            compactness=args.compactness,
            sigma=args.sigma
        )
        
        # Get the model's initial prediction
        prediction = lime_explainer.get_model_prediction(image)
        predicted_class_idx = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_idx]
        predicted_prob = prediction[predicted_class_idx]
        
        print(f"\nModel Prediction:")
        for i, (class_name, prob) in enumerate(zip(class_names, prediction)):
            print(f"  {class_name}: {prob:.4f}" + (" (Predicted)" if i == predicted_class_idx else ""))
        
        # Classes to analyze
        classes_to_analyze = [predicted_class_idx]
        if args.analyze_both_classes:
            classes_to_analyze = list(range(len(class_names)))
        
        # For each class, generate and visualize LIME explanation
        for class_idx in classes_to_analyze:
            class_name = class_names[class_idx]
            class_prob = prediction[class_idx]
            
            print(f"\n{'='*50}")
            print(f"Analyzing explanation for class: {class_name} (Probability: {class_prob:.4f})")
            print(f"{'='*50}")
            
            # Generate explanation
            segments, segments_importance, feature_importance, r2, pred_class, pred_probs = lime_explainer.explain(
                image, target_class=class_idx)
            
            # Create output path for this class
            output_prefix = f"{base_filename}_{class_name.lower()}"
            
            # Visualize explanation
            lime_explainer.visualize_explanation(
                image, segments, segments_importance, 
                class_name, class_prob, r2,
                save_path=os.path.join(args.output_dir, f"{output_prefix}_explanation.png")
            )
            
            # Create overlay visualization
            lime_explainer.create_overlay_visualization(
                image, segments, segments_importance,
                class_name, class_prob, threshold=0.0,
                save_path=os.path.join(args.output_dir, f"{output_prefix}_overlay.png")
            )
            
            # Find and print influential regions
            influential_regions = lime_explainer.find_most_influential_regions(
                segments, feature_importance, top_n=5)
            
            print("\nMost Influential Regions:")
            for region in influential_regions:
                print(f"  Rank {region['rank']}: Segment {region['segment_id']}, " +
                      f"Importance: {region['importance']:.4f} ({region['influence']})")
            
            # Visualize top regions
            lime_explainer.visualize_top_regions(
                image, segments, feature_importance, top_n=5,
                save_path=os.path.join(args.output_dir, f"{output_prefix}_top_regions.png")
            )
            
            # Save numerical results for further analysis
            np.savez(
                os.path.join(args.output_dir, f"{output_prefix}_data.npz"),
                segments=segments,
                segments_importance=segments_importance,
                feature_importance=feature_importance,
                r2=r2,
                prediction=prediction
            )
        
        print(f"\nAnalysis complete! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()