#!/bin/bash
#SBATCH --job-name=ganformer_cifar100
#SBATCH --output=ganformer_cifar100_%j.out
#SBATCH --error=ganformer_cifar100_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=ma618

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"


# Activate your environment 
source ~/venv/gansformer/bin/activate

# Navigate to the project directory (modify with your actual path)
cd $HOME/ConditionalGansformer

# Create results directory
mkdir -p results/cifar100_ganformer

module load cuda12.0

# Print CUDA devices and PyTorch version
echo "CUDA devices available:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('PyTorch version:', torch.__version__)"

# Run the training
echo "Starting GANsformer training on CIFAR-100..."
python gansformer/run_network.py \
  --data-dir=datasets \
  --dataset=cifar100 \
  --resolution=32 \
  --components-num=8 \
  --latent-size=128 \
  --batch-size=32 \
  --batch-gpu=32 \
  --gamma=15 \
  --total-kimg=15000 \
  --mirror-augment \
  --autotune \
  --g-lr=0.0025 \
  --d-lr=0.0025 \
  --result-dir=results/cifar100_ganformer \
  --train \
  --gpus=0 \
  --expname=cifar100_ganformer --transformer 

echo "Training completed at $(date)" 