#!/bin/bash
#SBATCH --job-name=generate_ffhq
#SBATCH --output=generate_ffhq_%j.out
#SBATCH --error=generate_ffhq_%j.err
#SBATCH --time=1:00:00
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

# Create results directory
mkdir -p results/ffhq_test

# Load CUDA
module load cuda12.0

# Print CUDA devices and PyTorch version
echo "CUDA devices available:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('PyTorch version:', torch.__version__)"

# Path to the pretrained FFHQ model
# Note: You'll need to download and place the model at this path
PRETRAINED_MODEL="pretrained_models/ffhq.pkl"

# Check if the model exists
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Error: Pretrained model not found at $PRETRAINED_MODEL"
    echo "You need to download the pretrained FFHQ GANsformer model first."
    echo "You can use gdown or another method to download it."
    exit 1
fi

# Run inference to generate 10 images
echo "Generating 10 images using pretrained FFHQ GANsformer model..."
python gansformer/generate.py \
    --network=$PRETRAINED_MODEL \
    --seeds=0-9 \
    --truncation-psi=0.7 \
    --outdir=results/ffhq_test \
    --transformer

echo "Generation completed at $(date)"
echo "Images saved to results/ffhq_test/" 