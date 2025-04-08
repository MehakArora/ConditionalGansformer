import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
from PIL import Image
import torchvision.transforms as transforms

class CLIPSimilarityLoss(nn.Module):
    """
    Implements CLIP similarity loss between real and generated images.
    This loss encourages generated images to have similar CLIP embeddings
    to the real images they're intended to match.
    """
    def __init__(self, device=None, clip_model="ViT-B/32", loss_weight=1.0, batch_size=16):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model_name = clip_model
        self.loss_weight = loss_weight
        self.batch_size = batch_size
        
        # Load the CLIP model
        print(f"Loading CLIP model {clip_model} for similarity loss...")
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        
        # Freeze the CLIP model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Create a preprocessing transform pipeline for generated images
        # This should match CLIP's expected input format
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                 (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def preprocess_generated(self, images):
        """
        Preprocess generated images to match CLIP's expected input format.
        
        Args:
            images: Tensor of shape [B, C, H, W] with values in range [-1, 1]
            
        Returns:
            Preprocessed images ready for CLIP
        """
        # Convert from [-1, 1] to [0, 1] range
        images = (images + 1) / 2.0
        
        # Apply CLIP's preprocessing
        return self.transform(images)
    
    def encode_images(self, images, is_generated=True):
        """
        Encode images using CLIP's image encoder.
        
        Args:
            images: Image tensor [B, C, H, W]
            is_generated: Whether these are generated images needing preprocessing
            
        Returns:
            CLIP embeddings [B, embedding_dim]
        """
        # Process in smaller batches to avoid memory issues
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, images.size(0), self.batch_size):
                batch = images[i:i+self.batch_size]
                
                # Apply preprocessing for generated images
                if is_generated:
                    batch = self.preprocess_generated(batch)
                
                # Get embeddings
                batch_embeddings = self.model.encode_image(batch)
                embeddings.append(batch_embeddings)
        
        # Concatenate all batch embeddings
        return torch.cat(embeddings, dim=0)
    
    def compute_similarity(self, real_embeddings, gen_embeddings):
        """
        Compute cosine similarity between real and generated embeddings.
        
        Args:
            real_embeddings: CLIP embeddings of real images [B, embedding_dim]
            gen_embeddings: CLIP embeddings of generated images [B, embedding_dim]
            
        Returns:
            Cosine similarity loss (1 - similarity)
        """
        # Normalize embeddings
        real_embeddings = F.normalize(real_embeddings, p=2, dim=1)
        gen_embeddings = F.normalize(gen_embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(real_embeddings, gen_embeddings)
        
        # Loss is 1 - similarity (so we minimize this to maximize similarity)
        loss = 1.0 - similarity.mean()
        
        return loss
    
    def forward(self, real_images, gen_images):
        """
        Compute CLIP similarity loss between real and generated images.
        
        Args:
            real_images: Real images tensor [B, C, H, W]
            gen_images: Generated images tensor [B, C, H, W]
            
        Returns:
            CLIP similarity loss
        """
        # Get CLIP embeddings
        real_embeddings = self.encode_images(real_images, is_generated=False)
        gen_embeddings = self.encode_images(gen_images, is_generated=True)
        
        # Compute similarity loss
        loss = self.compute_similarity(real_embeddings, gen_embeddings)
        
        # Apply weight
        return self.loss_weight * loss
    
    def load_precomputed_embeddings(self, path, device=None):
        """
        Load precomputed CLIP embeddings for real images.
        
        Args:
            path: Path to the numpy file containing embeddings
            device: Device to load the embeddings to
            
        Returns:
            Tensor of embeddings
        """
        device = device or self.device
        embeddings = np.load(path)
        return torch.tensor(embeddings, device=device)

# Example usage in a training loop:
"""
# Setup:
clip_loss = CLIPSimilarityLoss(device=device, loss_weight=0.5)

# Optionally, load precomputed embeddings for real images
real_embeddings = clip_loss.load_precomputed_embeddings('datasets/cifar100/labels_clip.npy')
real_embeddings_dataset = torch.utils.data.TensorDataset(real_embeddings)
real_embeddings_loader = torch.utils.data.DataLoader(real_embeddings_dataset, batch_size=batch_size)

# In training loop:
for real_data, real_labels in data_loader:
    # Get corresponding precomputed embeddings if using them
    real_emb_batch = next(iter(real_embeddings_loader))
    
    # Generate fake images
    fake_images = generator(z, real_labels)
    
    # Standard GAN loss
    d_loss = discriminator_loss(real_data, fake_images)
    g_loss = generator_loss(fake_images)
    
    # Add CLIP similarity loss
    if use_precomputed:
        # Use precomputed real embeddings
        clip_sim_loss = clip_loss.compute_similarity(real_emb_batch, 
                                                    clip_loss.encode_images(fake_images))
    else:
        # Compute embeddings on the fly
        clip_sim_loss = clip_loss(real_data, fake_images)
    
    # Combine losses
    total_g_loss = g_loss + clip_sim_loss
    
    # Update weights
    # ...
""" 