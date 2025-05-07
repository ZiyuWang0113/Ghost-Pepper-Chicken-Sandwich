#!/bin/bash
#SBATCH --job-name=hf_vit_deepfake_detection # Job name for Hugging Face ViT
#SBATCH --partition=gpu                   # Partition: Using the GPU partition
#SBATCH --nodes=1                         # Number of nodes: Requesting 1 node
#SBATCH --ntasks-per-node=1               # Number of tasks (processes) per node
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --mem=32G                         # Memory per node (increased for larger dataset)
#SBATCH --gres=gpu:2                      # Number of GPUs
#SBATCH --time=48:00:00                   # Time limit (increased for larger dataset)
#SBATCH --output=slurm_outputs/hf_vit_deepfake_%j.out # Standard output file
#SBATCH --error=slurm_outputs/hf_vit_deepfake_%j.err  # Standard error file

# --- Safety and Setup ---
set -e # Exit immediately if a command exits with a non-zero status.
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Create output directory for slurm logs if it doesn't exist
mkdir -p slurm_outputs
echo "Created slurm_outputs directory (if it didn't exist)."

# --- Environment Setup ---
echo "Loading modules..."
module load python/3.9.16s-x3wdtvt

echo "Setting up virtual environment..."
# Define Venv directory in your scratch space
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
pip install --upgrade transformers accelerate datasets
pip install --upgrade matplotlib pandas scikit-learn seaborn tqdm tensorboard

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
DATA_DIR="/oscar/scratch/rgao44/Dataset"
OUTPUT_DIR="/oscar/scratch/rgao44/output/hf_vit_training_${SLURM_JOB_ID}"
SCRIPT_NAME="train_hf_vit.py"

# Create the specific output directory for this run
mkdir -p "$OUTPUT_DIR"
echo "Output directory for this run: $OUTPUT_DIR"

# --- Execute Training Script ---
echo "Starting Python training script: $SCRIPT_NAME"

python "$SCRIPT_NAME" \
    --data_root "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 64 \
    --num_epochs 20 \
    --learning_rate 1e-5 \
    --weight_decay 1e-3 \
    --num_workers "$SLURM_CPUS_PER_TASK"

echo "Training script finished at $(date)."
echo "Job completed."