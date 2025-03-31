#!/bin/bash
#SBATCH --job-name=generate_ganformer
#SBATCH --output=generate_ganformer_%j.out
#SBATCH --error=generate_ganformer_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=ma618

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"

# Activate your environment 
source ~/venv/gansformer/bin/activate

# Navigate to the project directory
cd $HOME/ConditionalGansformer

# Choose the latest checkpoint or specify one
CHECKPOINT=$(ls -t results/cifar100_ganformer/network-snapshot-*.pkl | head -n 1)
if [ -z "$CHECKPOINT" ]; then
    echo "No checkpoint found to generate images from!"
    exit 1
fi
echo "Generating images using checkpoint: $CHECKPOINT"

# Create output directory
IMAGES_DIR="generated_images/cifar100"
mkdir -p $IMAGES_DIR

# Generate images
echo "Generating images from trained GANsformer..."
python gansformer/generate.py \
  --model=$CHECKPOINT \
  --gpus=0 \
  --output-dir=$IMAGES_DIR \
  --images-num=100 \
  --truncation-psi=0.7 \
  --ratio=1.0

echo "Images generated at $IMAGES_DIR"
echo "Completed at $(date)" 