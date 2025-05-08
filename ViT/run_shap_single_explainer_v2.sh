#!/bin/bash
#SBATCH --job-name=shap_deepfake_explainer  # Job name
#SBATCH --partition=gpu                     # Partition: Using the GPU partition
#SBATCH --nodes=1                           # Number of nodes: Requesting 1 node
#SBATCH --ntasks-per-node=1                 # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8                   # CPU cores per task
#SBATCH --mem=48G                           # Maximum memory allocation (48GB)
#SBATCH --gres=gpu:2                        # Using 2 GPUs
#SBATCH --time=48:00:00                     # Maximum time allocation (48 hours)
#SBATCH --output=slurm_outputs/shap_explainer_%j.out  # Standard output file
#SBATCH --error=slurm_outputs/shap_explainer_%j.err   # Standard error file

# --- Safety and Setup ---
set -e  # Exit immediately if a command exits with a non-zero status.
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "GPU configuration: $(nvidia-smi -L 2>/dev/null || echo 'nvidia-smi not available')"

# Create output directory for slurm logs if it doesn't exist
mkdir -p slurm_outputs
echo "Created slurm_outputs directory (if it didn't exist)."

# --- Environment Setup ---
echo "Loading modules..."
module load python/3.9.16s-x3wdtvt

echo "Setting up virtual environment..."
# Define Venv directory in your scratch space (same as training script for consistency)
VENV_DIR="/oscar/scratch/rgao44/venv/deepfake_hf_vit_env"

# Create parent directory for venv if it doesn't exist
mkdir -p "$(dirname "$VENV_DIR")"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR/bin" ]; then
    echo "Creating virtual environment in $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "Which python: $(which python)"

# --- Dependency Installation ---
echo "Installing/updating Python packages..."
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face transformers and other required packages
pip install --upgrade transformers accelerate
pip install --upgrade matplotlib pandas scikit-learn scikit-image seaborn tqdm

# Install SHAP with minimal dependencies
pip install --upgrade shap psutil

echo "Finished package installation."

# --- Working Directory ---
echo "Changing to submission directory: $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"
echo "Current directory: $(pwd)"

# --- Environment Variables for Performance ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to $OMP_NUM_THREADS"

# For Hugging Face optimization
export HF_HOME="/oscar/scratch/rgao44/hf_cache"
mkdir -p "$HF_HOME"
echo "HF_HOME set to $HF_HOME"

# Disable use_fast warning
export TRANSFORMERS_NO_USE_FAST_PROCESSOR=0

# --- Define Paths and Arguments ---
# Replace with your actual training job ID
MODEL_PATH="/oscar/scratch/rgao44/output/hf_vit_training_11193199/final_model"

# Replace with image path - can be a single image or multiple images using $1
if [ -z "$1" ]; then
    # Default image if none provided as argument
    echo "No image path provided as argument, using default"
    IMAGE_PATH="/path/to/default/image.jpg"
else
    IMAGE_PATH="$1"
    echo "Using image: $IMAGE_PATH"
fi

OUTPUT_DIR="/oscar/scratch/rgao44/shap_analysis/${SLURM_JOB_ID}"
SCRIPT_NAME="shap_single_explainer_oscar_v2.py"

# Create the specific output directory for this run
mkdir -p "$OUTPUT_DIR"
echo "Output directory for this run: $OUTPUT_DIR"

# --- Execute SHAP Explainer Script with memory-aware settings ---
echo "Starting Python SHAP explainer script: $SCRIPT_NAME with optimized settings"

python "$SCRIPT_NAME" \
    --model_path "$MODEL_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 2000 \
    --num_superpixels 50 \
    --compactness 10 \
    --sigma 1 \
    --background_size 15 \
    --analyze_both_classes \
    --multi_gpu \
    --batch_size 8 \
    --max_workers 4 \
    --low_memory_mode \
    --image_size 224 \
    --checkpoint_interval 100

# Check exit status
if [ $? -eq 0 ]; then
    echo "SHAP explainer script finished successfully at $(date)."
else
    echo "SHAP explainer script encountered an error. Check the logs for details."
fi

# Clean up
echo "Cleaning up..."
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()
"

echo "Job completed."