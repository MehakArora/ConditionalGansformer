import os
import sys
import shutil
import urllib.request
import tarfile
from tqdm import tqdm

# Add PIL compatibility patch
def add_pil_compatibility():
    """Add compatibility for newer Pillow versions that removed ANTIALIAS constant"""
    try:
        import PIL.Image
        if not hasattr(PIL.Image, 'ANTIALIAS'):
            # For Pillow 9.0.0 and newer
            PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
            print("Added PIL compatibility patch for Pillow 9.0.0+ (ANTIALIAS)")
    except Exception as e:
        print(f"Warning: Could not add PIL compatibility patch: {e}")

def download_cifar100(data_dir='datasets', force_download=False):
    """
    Download CIFAR-100 dataset and process it using the GANsformer's built-in functionality.
    
    Args:
        data_dir: Root directory for datasets
        force_download: Whether to download even if files exist
    """
    # Add PIL compatibility patch
    add_pil_compatibility()
    
    # URLs for the CIFAR-100 dataset
    cifar100_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    
    # Create directories
    cifar_dir = os.path.join(data_dir, 'cifar100')
    cifar_raw_dir = os.path.join(cifar_dir, 'raw')
    os.makedirs(cifar_dir, exist_ok=True)
    os.makedirs(cifar_raw_dir, exist_ok=True)
    
    # Download the CIFAR-100 dataset
    tar_file = os.path.join(cifar_raw_dir, "cifar-100-python.tar.gz")
    if not os.path.exists(tar_file) or force_download:
        print(f"Downloading CIFAR-100 dataset from {cifar100_url}...")
        with urllib.request.urlopen(cifar100_url) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 8192
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
            
            with open(tar_file, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    progress_bar.update(len(buffer))
            progress_bar.close()
    
    # Extract the dataset
    extract_dir = os.path.join(cifar_raw_dir, "cifar-100-python")
    if not os.path.exists(extract_dir) or force_download:
        print("Extracting CIFAR-100 dataset...")
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=cifar_raw_dir)
    
    # Now create the dataset using simple image extraction
    try:
        # Add gansformer directory to path to import modules
        gansformer_dir = os.path.join(os.getcwd(), 'gansformer')
        if not os.path.exists(gansformer_dir):
            print("Warning: gansformer directory not found in current path.")
            print("Please run this script from the root directory of the GANsformer project.")
            return
        
        sys.path.append(gansformer_dir)
        
        # Import necessary modules
        import pickle
        import numpy as np
        from PIL import Image
        from gansformer.dataset_tool import create_from_imgs
        
        print("\nExtracting images from CIFAR-100 binary files...")
        
        # Create a temporary directory for the extracted images
        images_dir = os.path.join(cifar_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Load the training data
        train_file = os.path.join(extract_dir, 'train')
        if os.path.exists(train_file):
            with open(train_file, 'rb') as f:
                train_data = pickle.load(f, encoding='bytes')
                
            # Extract train images
            images = train_data[b'data']
            labels = train_data[b'fine_labels']
            filenames = train_data[b'filenames']
            
            print(f"Extracting {len(images)} training images...")
            for i, (image, label) in enumerate(zip(images, labels)):
                # Reshape image from flat array to 3D array (3, 32, 32)
                image = image.reshape(3, 32, 32).transpose(1, 2, 0)
                
                # Create PIL image
                pil_image = Image.fromarray(image)
                
                # Save image
                pil_image.save(os.path.join(images_dir, f'train_{i:05d}_{label:03d}.png'))
        else:
            print(f"Warning: Training file not found at {train_file}")
        
        # Load the test data
        test_file = os.path.join(extract_dir, 'test')
        if os.path.exists(test_file):
            with open(test_file, 'rb') as f:
                test_data = pickle.load(f, encoding='bytes')
                
            # Extract test images
            images = test_data[b'data']
            labels = test_data[b'fine_labels']
            
            print(f"Extracting {len(images)} test images...")
            for i, (image, label) in enumerate(zip(images, labels)):
                # Reshape image from flat array to 3D array (3, 32, 32)
                image = image.reshape(3, 32, 32).transpose(1, 2, 0)
                
                # Create PIL image
                pil_image = Image.fromarray(image)
                
                # Save image
                pil_image.save(os.path.join(images_dir, f'test_{i:05d}_{label:03d}.png'))
        else:
            print(f"Warning: Test file not found at {test_file}")
        
        # Create the dataset in GANsformer format
        print("\nCreating GANsformer dataset from extracted images...")
        # Add resolution parameter explicitly
        create_from_imgs(cifar_dir, images_dir, format='png')
        
        print("\nDataset is ready for training!")
        print("You can now train a model using:")
        print(f"python gansformer/run_network.py --data-dir={data_dir} --dataset=cifar100 --resolution=32")
    except ImportError as e:
        print(f"Error importing GANsformer modules: {e}")
        print("Please make sure you're running this script from the GANsformer project root.")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Download and prepare CIFAR-100 dataset for GANsformer')
    parser.add_argument('--data-dir', type=str, default='datasets', help='Directory to store the dataset')
    parser.add_argument('--force', action='store_true', help='Force download even if files exist')
    
    args = parser.parse_args()
    download_cifar100(data_dir=args.data_dir, force_download=args.force) 