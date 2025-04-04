import os
import pickle
import numpy as np
import torch
import sys

try:
    # Try to import CLIP
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

def load_coco_captions(dataset_dir):
    """
    Load the COCO captions data from the prepared dataset
    
    Args:
        dataset_dir: Path to the GANsformer COCO dataset directory
    
    Returns:
        Dictionary containing caption data
    """
    captions_path = os.path.join(dataset_dir, 'captions.pkl')
    if not os.path.exists(captions_path):
        raise FileNotFoundError(f"Captions file not found at {captions_path}. Run download_coco.py first.")
    
    with open(captions_path, 'rb') as f:
        captions_data = pickle.load(f)
    
    return captions_data

def create_caption_labels(dataset_dir, mode='embedding', output_path=None, model_name="ViT-B/32"):
    """
    Create labels from captions for conditional GANsformer training
    
    Args:
        dataset_dir: Path to the GANsformer COCO dataset directory
        mode: 'embedding' for CLIP embeddings, 'onehot' for cluster-based one-hot encoding
        output_path: Path to save the labels, defaults to dataset_dir/labels.npy
        model_name: CLIP model to use for embeddings
    
    Returns:
        Path to the created labels file
    """
    captions_data = load_coco_captions(dataset_dir)
    primary_captions = captions_data['primary_captions']
    
    # Sort by index to ensure order matches dataset
    indices = sorted(primary_captions.keys())
    captions = [primary_captions[idx] for idx in indices]
    
    output_path = output_path or os.path.join(dataset_dir, 'labels.npy')
    
    if mode == 'embedding':
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP is required for embedding mode. Install with: pip install git+https://github.com/openai/CLIP.git")
        
        # Load CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model_name, device=device)
        
        # Get text embeddings
        print(f"Generating CLIP embeddings for {len(captions)} captions...")
        text_inputs = clip.tokenize(captions).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            text_features = text_features.cpu().numpy()
        
        # Normalize embeddings
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        
        # Save as labels.npy
        with open(output_path, 'wb') as f:
            np.save(f, text_features.astype(np.float32))
        
        print(f"Created CLIP embedding labels with shape {text_features.shape} at {output_path}")
        return output_path
    
    elif mode == 'onehot':
        # This is a simplified approach - in practice, you'd want to cluster similar captions
        # For demonstration, we'll create a dummy one-hot encoding
        num_samples = len(captions)
        onehot = np.eye(num_samples, dtype=np.float32)
        
        with open(output_path, 'wb') as f:
            np.save(f, onehot)
        
        print(f"Created one-hot labels with shape {onehot.shape} at {output_path}")
        return output_path
    
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'embedding' or 'onehot'.")

class ClipSimilarityLoss:
    """
    CLIP-based similarity loss for conditional GANsformer training
    """
    def __init__(self, device="cuda"):
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP is required for similarity loss. Install with: pip install git+https://github.com/openai/CLIP.git")
        
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
    def compute_similarity(self, images, captions):
        """
        Compute similarity between generated images and their corresponding captions
        
        Args:
            images: Tensor of images [batch_size, 3, H, W] in range [-1, 1]
            captions: List of caption strings
            
        Returns:
            Similarity loss (lower means more similar)
        """
        # Preprocess images for CLIP (generated images are in range [-1, 1], normalize to [0, 1] first)
        normalized_images = (images + 1) / 2
        image_inputs = torch.nn.functional.interpolate(normalized_images, size=224)
        
        # Tokenize text
        text_inputs = clip.tokenize(captions).to(self.device)
        
        # Get features
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
            text_features = self.model.encode_text(text_inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Compute similarity (cosine similarity)
            similarity = (image_features * text_features).sum(dim=1)
        
        # Return negative similarity as loss (higher similarity = lower loss)
        return -similarity.mean()

def extend_gansformer_for_conditional_training():
    """
    Guide on how to extend GANsformer for conditional training with COCO captions
    """
    guide = """
How to Extend GANsformer for Conditional Training with COCO Captions
====================================================================

1. Prepare the COCO dataset:
   - Run download_coco.py to download and process the COCO dataset
   - This creates placeholder labels and stores captions separately

2. For simple conditional training with one-hot labels:
   - Use create_caption_labels(dataset_dir, mode='onehot')
   - Train with: python gansformer/run_network.py --data-dir=datasets --dataset=coco_train2017 --c-dim=X

3. For CLIP embedding-based conditioning:
   - Use create_caption_labels(dataset_dir, mode='embedding')
   - Set c_dim to the embedding dimension (512 for CLIP ViT-B/32)
   - Train with: python gansformer/run_network.py --data-dir=datasets --dataset=coco_train2017 --c-dim=512

4. For CLIP similarity loss:
   - Implement a custom training loop that includes the CLIP similarity loss
   - Use the ClipSimilarityLoss class from this file
   - Add this loss to the generator's loss with a weighting factor

Example code for training with CLIP similarity loss:
```python
# In your training loop
clip_loss = ClipSimilarityLoss(device=device)

# Get captions for current batch
batch_captions = [primary_captions[idx] for idx in batch_indices]

# Regular GAN loss
gen_loss = original_gen_loss

# Add CLIP similarity loss
images = generator(z, c)
similarity_loss = clip_loss.compute_similarity(images, batch_captions)
gen_loss = gen_loss + similarity_weight * similarity_loss
```

To implement this in the GANsformer codebase:
1. Modify gansformer/training/loss.py to include the CLIP similarity loss
2. Add a weight parameter for the CLIP loss to the command line arguments
3. Ensure the caption data is accessible during training

For even better results, consider fine-tuning the CLIP model on your specific domain.
"""
    print(guide)
    return guide

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Utilities for conditional GANsformer training with COCO captions')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to GANsformer COCO dataset directory')
    parser.add_argument('--mode', choices=['embedding', 'onehot'], default='embedding', 
                        help='Mode for creating labels (embedding or onehot)')
    parser.add_argument('--output-path', type=str, help='Output path for labels (default: dataset_dir/labels.npy)')
    parser.add_argument('--guide', action='store_true', help='Print guide for implementing conditional training')
    
    args = parser.parse_args()
    
    if args.guide:
        extend_gansformer_for_conditional_training()
    else:
        create_caption_labels(args.dataset_dir, args.mode, args.output_path) 