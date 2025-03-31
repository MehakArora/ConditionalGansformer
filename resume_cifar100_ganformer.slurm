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

# Load CUDA module for CUDA 12.0
module load cuda12.0

# Check and set CUDA environment variables
# The module should set CUDA_HOME, but we'll check and set it if needed
if [ -z "$CUDA_HOME" ]; then
    # Common locations for CUDA installation
    if [ -d "/usr/local/cuda-12.0" ]; then
        export CUDA_HOME=/usr/local/cuda-12.0
    elif [ -d "/cm/local/apps/cuda/12.0" ]; then
        export CUDA_HOME=/cm/local/apps/cuda/12.0
    else
        # Try to find it from nvcc path
        NVCC_PATH=$(which nvcc 2>/dev/null)
        if [ ! -z "$NVCC_PATH" ]; then
            export CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
        else
            echo "WARNING: Could not determine CUDA_HOME automatically."
            echo "Please check your CUDA installation and set CUDA_HOME manually."
        fi
    fi
fi

# Update library and executable paths
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:${PATH}

# Print CUDA environment for debugging
echo "CUDA environment:"
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
nvcc --version || echo "nvcc not found in PATH"

# Activate your environment 
source ~/venv/gansformer/bin/activate

# Navigate to the project directory
cd $HOME/ConditionalGansformer

# Check Python version
echo "Python version:"
python --version

# Install required dependencies
echo "Installing/updating required dependencies..."
pip install click==8.0.4 tqdm pillow==8.4.0

# Install TensorFlow with compatibility - try multiple approaches
echo "Installing compatible TensorFlow version..."
pip uninstall -y tensorflow tensorflow-gpu

# Option 1: Try regular TensorFlow 1.15 (might work on some systems without GPU)
echo "Trying tensorflow 1.15.0..."
pip install tensorflow==1.15.0 || echo "Failed to install tensorflow 1.15.0"

# Option 2: TensorFlow 2.x with compatibility package
if ! python -c "import tensorflow.contrib" 2>/dev/null; then
    echo "Installing TensorFlow 2.x with compatibility package..."
    pip install tensorflow==2.3.0 tensorflow-addons
    pip install tf-slim
    
    # Create compatibility.py file
    cat > ~/venv/gansformer/lib/python*/site-packages/tensorflow/contrib.py << 'EOF'
import tensorflow as tf
import tensorflow_addons as tfa
import tf_slim as slim

# Add contrib components as needed
contrib = type('contrib', (), {})
contrib.layers = slim
contrib.slim = slim
contrib.distributions = tf.compat.v1.distributions
contrib.optimizer = tfa.optimizers
tf.contrib = contrib
EOF

    echo "Created TensorFlow compatibility layer"
fi

# Choose the latest checkpoint
CHECKPOINT=$(ls -t results/cifar100_ganformer/network-snapshot-*.pkl | head -n 1)
if [ -z "$CHECKPOINT" ]; then
    echo "No checkpoint found to resume from!"
    exit 1
fi
echo "Resuming from checkpoint: $CHECKPOINT"

# Verify TensorFlow installation
echo "TensorFlow information:"
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('TensorFlow has contrib:', hasattr(tf, 'contrib')); print('TensorFlow CUDA available:', tf.test.is_built_with_cuda()); print('TensorFlow GPU available:', tf.test.is_gpu_available())"

# Run the training with resumption
echo "Resuming GANsformer training on CIFAR-100..."
python gansformer/run_network.py \
  --data-dir datasets \
  --dataset cifar100 \
  --resolution 32 \
  --components-num 8 \
  --latent-size 128 \
  --batch-size 32 \
  --batch-gpu 32 \
  --gamma 15 \
  --total-kimg 15000 \
  --mirror-augment \
  --autotune \
  --g-lr 0.0025 \
  --d-lr 0.0025 \
  --resume-pkl $CHECKPOINT \
  --train \
  --expname cifar100_ganformer \
  --gpus 0 \
  --reload

echo "Training completed at $(date)" 