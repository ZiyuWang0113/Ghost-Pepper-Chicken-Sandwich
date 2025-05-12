#SBATCH --job-name=effnet_kag_deepfake
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --output=effnet_kag_%j.out

cd $SLURM_SUBMIT_DIR

module load python/3.11.0s-ixrhc3q

source venv_effnet/bin/activate

python train_effnet.py \
  --data Dataset \
  --model efficientnet_b3 \
  --size 300 \
  --batch 64 \
  --epochs 50 \
  --out checkpoints_kag \
  --workers 8 \
  --lr 3e-4
