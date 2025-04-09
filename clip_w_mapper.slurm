#!/bin/bash
#SBATCH --job-name=clip_mapper_${SLURM_JOB_ID}
#SBATCH --output=clip_mapper_${SLURM_JOB_ID}.out
#SBATCH --error=clip_mapper_${SLURM_JOB_ID}.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"

# Activate your environment 

source activate ganformer


# Navigate to the project directory
cd $HOME/ConditionalGansformer

# Load CUDA
module load bmi/gcc-14.1.0

# Print CUDA devices and PyTorch version
echo "CUDA devices available:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('PyTorch version:', torch.__version__)"

# Path to the pretrained FFHQ model
PRETRAINED_MODEL="/opt/scratchspace/conditionalGansformer/ffhq_model/ffhq.pkl"

# Check if the model exists
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Pretrained model not found. Attempting to download..."
    python download_ffhq_model.py
    
    if [ ! -f "$PRETRAINED_MODEL" ]; then
        echo "Error: Failed to download pretrained model."
        exit 1
    fi
fi

cd $HOME/ConditionalGansformer/gansformer

# Create directories
mkdir -p /opt/scratchspace/conditionalGansformer/ffhq_clip_mapper
mkdir -p /opt/scratchspace/conditionalGansformer/ffhq_clip_mapper/results_${SLURM_JOB_ID}

# Step 1: Train the CLIP-W mapper
echo "Training CLIP-W mapper..."
python clip_w_mapper.py \
    --model=$PRETRAINED_MODEL \
    --train \
    --mapper-path=/opt/scratchspace/conditionalGansformer/ffhq_clip_mapper/clip_w_mapper.pt \
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
#Loop through each text prompt and generate corresponding images
for prompt in "${prompts[@]}"; do
    echo "Generating image for prompt: '$prompt'"
    python clip_w_mapper.py \
        --model=$PRETRAINED_MODEL \
        --generate \
        --mapper-path=/opt/scratchspace/conditionalGansformer/ffhq_clip_mapper/clip_w_mapper.pt \
        --prompt="$prompt" \
        --outdir=/opt/scratchspace/conditionalGansformer/ffhq_clip_mapper/results_${SLURM_JOB_ID}
done

echo "CLIP-W mapper training and generation completed at $(date)"
echo "Results saved to /opt/scratchspace/conditionalGansformer/ffhq_clip_mapper/results_${SLURM_JOB_ID}" 