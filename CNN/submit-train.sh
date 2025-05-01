#!/bin/bash
#SBATCH -p gpu                  # Use the GPU partition
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH -n 4                    # Request 4 CPU cores
#SBATCH --mem=16G               # Request 16 GB memory
#SBATCH -t 3:30:00              # Time limit (hh:mm:ss)
#SBATCH -J wzy_train_augdrop    # Descriptive job name
#SBATCH -o train_output.log     # standard output
#SBATCH -e train_error.log      # standard error

# Load modules
module purge
module load cuda/11.8

# Activate your Python environment
source ~/1430/bin/activate

# Go to submission directory
cd $SLURM_SUBMIT_DIR

# Run the updated training script
python train.py
