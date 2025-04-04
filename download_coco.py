import os
import sys
import json
import urllib.request
import tarfile
import zipfile
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
import pickle

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

def download_file(url, destination, description=None):
    """Download a file with progress bar"""
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
    
    print(f"Downloading {description or url} to {destination}")
    with urllib.request.urlopen(url) as response:
        total_size = int(response.info().get('Content-Length', 0))
        block_size = 8192
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
        
        with open(destination, 'wb') as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                progress_bar.update(len(buffer))
        progress_bar.close()

def extract_archive(archive_path, extract_dir, description=None):
    """Extract a tar.gz or zip archive"""
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        
    print(f"Extracting {description or archive_path}...")
    
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

def resize_image(image_path, output_path, target_size=256):
    """Resize image to target size while maintaining aspect ratio"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            ratio = min(target_size / width, target_size / height)
            
            # Calculate new dimensions
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Resize the image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Create a new blank image with the target size
            new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            
            # Paste the resized image in the center
            offset_x = (target_size - new_width) // 2
            offset_y = (target_size - new_height) // 2
            new_img.paste(resized_img, (offset_x, offset_y))
            
            # Save the processed image
            new_img.save(output_path)
            return True
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

def process_images_for_gansformer(images_dir, output_dir, target_size=256):
    """Process COCO images into GANsformer format"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create base resolution directory
    base_res_dir = os.path.join(output_dir, str(target_size))
    os.makedirs(base_res_dir, exist_ok=True)
    
    # Create directories for downscaled versions
    current_size = target_size
    while current_size > 4:
        current_size = current_size // 2
        os.makedirs(os.path.join(output_dir, str(current_size)), exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    print(f"Processing {len(image_files)} images...")
    
    # Dictionary to track processed images and their original IDs
    processed_images = {}
    
    # Process each image
    for idx, filename in enumerate(tqdm(image_files)):
        image_id = int(os.path.splitext(filename)[0])
        source_path = os.path.join(images_dir, filename)
        target_path = os.path.join(base_res_dir, f"{idx}.png")
        
        if resize_image(source_path, target_path, target_size):
            processed_images[idx] = image_id
            
            # Create downscaled versions
            img = Image.open(target_path)
            current_size = target_size
            while current_size > 4:
                current_size = current_size // 2
                downscaled_img = img.resize((current_size, current_size), Image.LANCZOS)
                downscaled_img.save(os.path.join(output_dir, str(current_size), f"{idx}.png"))
    
    return processed_images

def extract_captions(annotations_file, image_id_map):
    """Extract captions for each processed image from annotations file"""
    print(f"Extracting captions from {annotations_file}...")
    
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Reverse the image_id_map to map from original COCO ID to our processed index
    reverse_map = {coco_id: processed_idx for processed_idx, coco_id in image_id_map.items()}
    
    # Dictionary to store captions for each processed image
    image_captions = {idx: [] for idx in image_id_map.keys()}
    
    # Extract captions
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        if image_id in reverse_map:
            processed_idx = reverse_map[image_id]
            image_captions[processed_idx].append(caption)
    
    # Create a list of captions (taking the first caption for each image)
    # Note: For GANsformer compatibility, we'll create a placeholder labels.npy first
    num_images = len(image_id_map)
    placeholder_labels = np.zeros((num_images, 1), dtype=np.float32)
    
    # Create a more complete captions dictionary for future use
    captions_data = {
        'image_id_map': image_id_map,
        'all_captions': image_captions,
        'primary_captions': {idx: captions[0] if captions else "" for idx, captions in image_captions.items()}
    }
    
    return placeholder_labels, captions_data

def download_coco(data_dir='datasets', target_size=256, split='train2017', force_download=False):
    """
    Download MS COCO dataset and prepare it for GANsformer
    
    Args:
        data_dir: Root directory for datasets
        target_size: Target image size (will create a square image)
        split: Dataset split ('train2017' or 'val2017')
        force_download: Whether to download even if files exist
    """
    # Add PIL compatibility patch
    add_pil_compatibility()
    
    # URLs for COCO dataset
    images_url = f"http://images.cocodataset.org/zips/{split}.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    # Create directories
    coco_dir = os.path.join(data_dir, 'coco')
    coco_raw_dir = os.path.join(coco_dir, 'raw')
    os.makedirs(coco_dir, exist_ok=True)
    os.makedirs(coco_raw_dir, exist_ok=True)
    
    # Download image dataset
    images_zip = os.path.join(coco_raw_dir, f"{split}.zip")
    if not os.path.exists(images_zip) or force_download:
        download_file(images_url, images_zip, f"COCO {split} images")
    
    # Download annotations
    annotations_zip = os.path.join(coco_raw_dir, "annotations.zip")
    if not os.path.exists(annotations_zip) or force_download:
        download_file(annotations_url, annotations_zip, "COCO annotations")
    
    # Extract images
    images_dir = os.path.join(coco_raw_dir, split)
    if not os.path.exists(images_dir) or force_download:
        extract_archive(images_zip, coco_raw_dir, f"{split} images")
    
    # Extract annotations
    annotations_dir = os.path.join(coco_raw_dir, 'annotations')
    if not os.path.exists(annotations_dir) or force_download:
        extract_archive(annotations_zip, coco_raw_dir, "annotations")
    
    # Create GANsformer dataset directory
    gansformer_dataset_dir = os.path.join(coco_dir, f'coco_{split}')
    
    # Process images for GANsformer
    print("\nProcessing images for GANsformer format...")
    image_id_map = process_images_for_gansformer(images_dir, gansformer_dataset_dir, target_size)
    
    # Extract captions
    captions_file = os.path.join(annotations_dir, f'captions_{split}.json')
    placeholder_labels, captions_data = extract_captions(captions_file, image_id_map)
    
    # Save labels.npy (placeholder for now)
    labels_path = os.path.join(gansformer_dataset_dir, 'labels.npy')
    with open(labels_path, 'wb') as f:
        np.save(f, placeholder_labels)
    
    # Save captions data for future use
    captions_path = os.path.join(gansformer_dataset_dir, 'captions.pkl')
    with open(captions_path, 'wb') as f:
        pickle.dump(captions_data, f)
    
    print("\nDataset is ready for training!")
    print(f"Processed {len(image_id_map)} images from COCO {split}")
    print(f"Dataset directory: {gansformer_dataset_dir}")
    print(f"Captions stored in: {captions_path}")
    print("\nYou can now train a model using:")
    print(f"python gansformer/run_network.py --data-dir={data_dir} --dataset=coco_{split} --resolution={target_size}")
    
    # Try to import the GANsformer dataset_tool to verify integration
    try:
        # Add gansformer directory to path to import modules
        gansformer_dir = os.path.join(os.getcwd(), 'gansformer')
        if os.path.exists(gansformer_dir):
            sys.path.append(gansformer_dir)
            from gansformer.dataset_tool import create_from_imgs
            print("\nGANsformer module found. You can also create the dataset using:")
            print(f"python -c \"from gansformer.dataset_tool import create_from_imgs; create_from_imgs('{gansformer_dataset_dir}', '{gansformer_dataset_dir}/{target_size}', format='png')\"")
    except ImportError as e:
        print(f"\nNote: Could not import GANsformer modules: {e}")
        print("Make sure you're running this script from the GANsformer project root.")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Download and prepare MS COCO dataset for GANsformer')
    parser.add_argument('--data-dir', type=str, default='datasets', help='Directory to store the dataset')
    parser.add_argument('--target-size', type=int, default=256, help='Target image size (width and height)')
    parser.add_argument('--split', type=str, default='train2017', choices=['train2017', 'val2017'], 
                        help='Dataset split to download (train2017 or val2017)')
    parser.add_argument('--force', action='store_true', help='Force download even if files exist')
    
    args = parser.parse_args()
    download_coco(data_dir=args.data_dir, target_size=args.target_size, 
                 split=args.split, force_download=args.force) 