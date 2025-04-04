#!/bin/bash
#SBATCH --job-name=resume_ganformer
#SBATCH --output=resume_ganformer_%j.out
#SBATCH --error=resume_ganformer_%j.err
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

# Navigate to the project directory
cd $HOME/ConditionalGansformer

module load cuda12.0

# Print CUDA devices information
echo "CUDA devices available:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('PyTorch version:', torch.__version__)"

# Choose the latest checkpoint
CHECKPOINT='/home/ma618/ConditionalGansformer/results/cifar100_ganformer/cifar100_ganformer-000/network-snapshot-005208.pkl'

echo "Resuming from checkpoint: $CHECKPOINT"

# Run the training with resumption
echo "Resuming GANsformer training on CIFAR-100..."
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
  --pretrained-pkl=$CHECKPOINT \
  --result-dir=results/cifar100_ganformer \
  --train \
  --expname=cifar100_ganformer \
  --gpus=0 \
  --transformer \
  --reload
  --result-dir=/data/irb/surgery/pro00114885/conditionalImageGen/results/cifar100_ganformer

echo "Training completed at $(date)" 