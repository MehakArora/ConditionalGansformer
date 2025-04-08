#!/bin/bash
#SBATCH --job-name=clip_mapper
#SBATCH --output=clip_mapper_%j.out
#SBATCH --error=clip_mapper_%j.err
#SBATCH --time=4:00:00
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

# Install CLIP if not already installed
pip install git+https://github.com/openai/CLIP.git

# Navigate to the project directory
cd $HOME/ConditionalGansformer

# Load CUDA
module load cuda12.0

# Print CUDA devices and PyTorch version
echo "CUDA devices available:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('PyTorch version:', torch.__version__)"

# Path to the pretrained FFHQ model
PRETRAINED_MODEL="pretrained_models/ffhq.pkl"

# Check if the model exists
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Pretrained model not found. Attempting to download..."
    python download_ffhq_model.py
    
    if [ ! -f "$PRETRAINED_MODEL" ]; then
        echo "Error: Failed to download pretrained model."
        exit 1
    fi
fi

# Create directories
mkdir -p models
mkdir -p results/clip_mapper

# Step 1: Train the CLIP-W mapper
echo "Training CLIP-W mapper..."
python clip_w_mapper.py \
    --model=$PRETRAINED_MODEL \
    --train \
    --mapper-path=models/clip_w_mapper.pt \
    --seed=42

# Step 2: Generate images using the trained mapper with different prompts
echo "Generating images from text prompts..."

# List of prompts to test
prompts=(
    "a person with glasses" 
    "a person with blonde hair" 
    "an elderly person" 
    "a smiling person" 
    "a person with curly hair" 
    "a person with a beard"
)

# Generate an image for each prompt
for prompt in "${prompts[@]}"; do
    echo "Generating image for prompt: '$prompt'"
    python clip_w_mapper.py \
        --model=$PRETRAINED_MODEL \
        --generate \
        --mapper-path=models/clip_w_mapper.pt \
        --prompt="$prompt" \
        --outdir=results/clip_mapper
done

echo "CLIP-W mapper training and generation completed at $(date)"
echo "Results saved to results/clip_mapper/" 