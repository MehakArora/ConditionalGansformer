#!/bin/bash
#SBATCH --job-name=coco_ganformer
#SBATCH --output=coco_ganformer_%j.out
#SBATCH --error=coco_ganformer_%j.err
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

# Set variables
DATA_DIR="/data/irb/surgery/pro00114885/conditionalImageGen/coco/"
DATASET="coco_train2017"
RESULT_DIR="/data/irb/surgery/pro00114885/conditionalImageGen/results/coco_ganformer/"
IMG_SIZE=224
BATCH_SIZE=16
TOTAL_KIMG=25000  # Total number of thousands of real images to show the network

mkdir -p $RESULT_DIR

module load cuda12.0

# Step 2: Train unconditional GANsformer on COCO
echo "Starting unconditional GANsformer training on COCO..."
python gansformer/run_network.py \
    --baseline=GAN \
    --data-dir=$DATA_DIR \
    --dataset=$DATASET \
    --resolution=$IMG_SIZE \
    --resize-to-power-of-2=True \
    --batch-size=$BATCH_SIZE \
    --total-kimg=$TOTAL_KIMG \
    --g-lr=0.001 \
    --d-lr=0.001 \
    --mirror-augment \
    --transformer \
    --components-num=16 \
    --latent-size=512 \
    --normalize=layer \
    --integration=mul \
    --g-arch=resnet \
    --expname=coco_ganformer_uncond \
    --use-pos \
    --mapping-ltnt2ltnt \
    --result-dir=$RESULT_DIR \
    --train
echo "Unconditional training completed at $(date)"

# Step 3: Create CLIP embeddings for conditional training
#echo "Creating CLIP embeddings for conditional training..."
#pip install git+https://github.com/openai/CLIP.git
#python conditional_coco_utils.py \
#    --dataset-dir=$DATA_DIR/coco/$DATASET \
#    --mode=embedding \
#    --output-path=$DATA_DIR/coco/$DATASET/clip_labels.npy

# Step 4: Train conditional GANsformer with CLIP embeddings
# echo "Starting conditional GANsformer training with CLIP embeddings..."
# cp $DATA_DIR/coco/$DATASET/clip_labels.npy $DATA_DIR/coco/$DATASET/labels.npy

# python gansformer/run_network.py \
#     --data-dir=$DATA_DIR \
#     --dataset=$DATASET \
#     --resolution=$IMG_SIZE \
#     --resize-to-power-of-2=True \
#     --batch-size=$BATCH_SIZE \
#     --total-kimg=$TOTAL_KIMG \
#     --g-lr=0.002 \
#     --d-lr=0.002 \
#     --mirror-augment=true \
#     --transformer=true \
#     --components-num=16 \
#     --latent-size=512 \
#     --normalize=layer \
#     --integration=mul \
#     --kmeans=true \
#     --use-pos=true \
#     --mapping-ltnt2ltnt=true \
#     --style=true \
#     --g-arch=resnet \
#     --c-dim=512 \
#     --expname=coco_ganformer_cond

# echo "Conditional training completed at $(date)"

# Print completion message
# echo "All training jobs completed successfully!"
# echo "End Time: $(date)"