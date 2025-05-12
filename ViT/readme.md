## Repository Structure

- `train_hf_vit.py`: Python script for training a Hugging Face Vision Transformer model for deepfake detection
- `train_hf_vit.sh`: SLURM batch script for running the ViT training on a GPU cluster (OSCAR)
- `extract_misclassified.py`: Utility for identifying and saving misclassified images for error analysis
- `lime_single_explainer.py`: Implementation of LIME (Local Interpretable Model-agnostic Explanations) for analyzing individual images
- `shap_single_explainer.py`: Implementation of SHAP (SHapley Additive exPlanations) for analyzing feature contributions
- Configuration files:
  - `config.json`: Model configuration for Vision Transformer
  - `preprocessor_config.json`: Image preprocessing configuration

## Setup and Requirements

### Dependencies

- PyTorch with CUDA support
- Transformers (Hugging Face)
- scikit-learn, scikit-image
- SHAP
- Matplotlib, Seaborn
- PIL, OpenCV
- NumPy, Pandas

### Dataset Structure

The code expects data in the following directory structure:
```
Dataset/
├── Train/
│   ├── Fake/
│   └── Real/
├── Validation/
│   ├── Fake/
│   └── Real/
└── Test/
    ├── Fake/
    └── Real/
```

## Usage Instructions

### Training a Vision Transformer Model

To train a ViT model on a local machine:
```bash
python train_hf_vit.py --data_root /path/to/Dataset --output_dir ./output --batch_size 64 --num_epochs 20
```

For cluster training (SLURM):
```bash
sbatch train_hf_vit.sh
```

The weights of the ViT models are too large to upload to github, please email rui_gao@brown.edu for weights inquiry.

### Extracting Misclassified Images

To identify where the model fails and extract misclassified images:
```bash
python extract_misclassified.py --data_root /path/to/Dataset --model_path ./final_model --output_dir ./misclassified
```

### Explainability Analysis

#### LIME Analysis on a Single Image
```bash
python lime_single_explainer.py --model_path ./final_model --image_path /path/to/image.jpg --output_dir ./lime_analysis
```

#### SHAP Analysis on a Single Image
```bash
python shap_single_explainer.py --model_path ./final_model --image_path /path/to/image.jpg --output_dir ./shap_analysis --num_samples 2000
```

## Visualization Outputs

The explainability scripts generate several visualization types:

- Segmentation maps showing superpixels used in analysis
- Heatmaps showing feature importance
- Overlay visualizations highlighting influential image regions
- Top-N most influential regions for the classification decision
- Force plots showing feature contributions (SHAP)

## Example Workflow

1. Train the deepfake detection model:
   ```
   python train_hf_vit.py --data_root ./Dataset
   ```

2. Extract misclassified images:
   ```
   python extract_misclassified.py --model_path ./final_model
   ```

3. Analyze a misclassified fake image using LIME:
   ```
   python lime_single_explainer.py --model_path ./final_model --image_path ./misclassified/false_negatives/Fake_as_Real_example.jpg
   ```

4. Compare with SHAP analysis of the same image:
   ```
   python shap_single_explainer.py --model_path ./final_model --image_path ./misclassified/false_negatives/Fake_as_Real_example.jpg
   ```

5. Examine and compare the results to understand which visual features influence the model's decision.

## Performance Metrics

The training code tracks multiple metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion matrices

Results are saved as text files and visualizations in the output directory.

## Advanced Usage Options

### LIME Options
- `--num_superpixels`: Number of segments to divide the image into (default: 50)
- `--num_samples`: Number of perturbed samples to generate (default: 1000)
- `--analyze_both_classes`: Analyze explanations for both fake and real classes

### SHAP Options
- `--background_size`: Number of background samples to use (default: 20)
- `--compactness`: Compactness parameter for SLIC segmentation (default: 10)

## Citation

If you find this code useful for your research, please consider citing our work.

## Acknowledgments

This project utilizes:
- Hugging Face Transformers
- LIME and SHAP implementation frameworks
- PyTorch
- scikit-image and scikit-learn