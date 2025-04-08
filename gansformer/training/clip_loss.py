import torch
from torch_utils import training_stats
from training.loss import StyleGAN2Loss
import numpy as np
import sys

try:
    import clip
    from torchvision import transforms
except ImportError:
    print("CLIP not found, installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
    import clip
    from torchvision import transforms
    print("CLIP installed successfully")

class StyleGAN2CLIPLoss(StyleGAN2Loss):
    """
    Extends StyleGAN2Loss to include a CLIP similarity loss between real and generated images.
    """
    def __init__(self, device, G, D, g_loss="logistic_ns", d_loss="logistic",
                style_mixing=0.9, component_mixing=0.0, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01,
                pl_weight=2.0, wgan_epsilon=0.001, clip_weight=1.0, clip_model="ViT-B/32", 
                clip_similarity_type="cosine", use_precomputed_embeddings=False, 
                precomputed_embeddings_path=None):
        super().__init__(device, G, D, g_loss, d_loss, style_mixing, component_mixing, 
                         r1_gamma, pl_batch_shrink, pl_decay, pl_weight, wgan_epsilon)
        
        # CLIP loss parameters
        self.clip_weight = clip_weight
        self.clip_model_name = clip_model
        self.clip_similarity_type = clip_similarity_type
        self.use_precomputed_embeddings = use_precomputed_embeddings
        
        # Load CLIP model
        print(f"Loading CLIP model {clip_model} for similarity loss...")
        self.clip_model, self.preprocess = clip.load(clip_model, device=device)
        
        # Freeze CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Load precomputed embeddings if specified
        self.precomputed_embeddings = None
        if use_precomputed_embeddings and precomputed_embeddings_path:
            print(f"Loading precomputed CLIP embeddings from {precomputed_embeddings_path}")
            self.precomputed_embeddings = torch.tensor(
                np.load(precomputed_embeddings_path), 
                device=device
            )
            print(f"Loaded embeddings with shape {self.precomputed_embeddings.shape}")
        
        # Create preprocessing transform for generated images
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def preprocess_for_clip(self, images):
        """
        Preprocess images to match CLIP's expected input format.
        
        Args:
            images: Tensor of shape [B, C, H, W] with values in range [-1, 1]
            
        Returns:
            Preprocessed images ready for CLIP
        """
        # Convert from [-1, 1] to [0, 1] range
        images = (images + 1) / 2.0
        
        # Apply CLIP's preprocessing
        return self.transform(images)
    
    def get_clip_embeddings(self, images):
        """
        Get CLIP embeddings for images.
        
        Args:
            images: Image tensor [B, C, H, W] in [-1, 1] range
            
        Returns:
            CLIP embeddings [B, embedding_dim]
        """
        with torch.no_grad():
            # Preprocess images for CLIP
            preprocessed = self.preprocess_for_clip(images)
            
            # Get embeddings
            embeddings = self.clip_model.encode_image(preprocessed)
            
            # Normalize embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
        return embeddings
    
    def compute_clip_similarity(self, real_emb, gen_emb):
        """
        Compute similarity between real and generated embeddings.
        
        Args:
            real_emb: CLIP embeddings of real images [B, embedding_dim]
            gen_emb: CLIP embeddings of generated images [B, embedding_dim]
            
        Returns:
            Similarity loss
        """
        if self.clip_similarity_type == "cosine":
            # Cosine similarity (higher is better)
            similarity = torch.sum(real_emb * gen_emb, dim=1)
            # Loss is 1 - similarity (to minimize)
            loss = 1.0 - similarity.mean()
        elif self.clip_similarity_type == "l2":
            # L2 distance (lower is better)
            loss = torch.norm(real_emb - gen_emb, dim=1).mean()
        else:
            raise ValueError(f"Unknown similarity type: {self.clip_similarity_type}")
        
        return loss
    
    def get_precomputed_embeddings(self, indices, batch_size):
        """
        Get precomputed embeddings for a batch of images.
        
        Args:
            indices: Indices of images in the dataset
            batch_size: Size of the batch
            
        Returns:
            Precomputed embeddings [batch_size, embedding_dim]
        """
        if self.precomputed_embeddings is None:
            raise ValueError("Precomputed embeddings not loaded")
        
        # Get embeddings for the current batch
        emb = self.precomputed_embeddings[indices]
        
        # Make sure we have the right batch size
        if emb.shape[0] < batch_size:
            # Pad with zeros if necessary
            pad_size = batch_size - emb.shape[0]
            emb = torch.cat([emb, torch.zeros(pad_size, emb.shape[1], device=emb.device)], dim=0)
        elif emb.shape[0] > batch_size:
            # Truncate if necessary
            emb = emb[:batch_size]
            
        return emb
        
    def accumulate_gradients(self, stage, real_img, real_c, gen_z, gen_c, sync, gain, indices=None):
        """
        Extended version of accumulate_gradients that includes CLIP similarity loss.
        
        Args:
            stage: Training stage ("G_main", "G_reg", "G_both", "D_main", "D_reg", "D_both")
            real_img: Real images
            real_c: Real image labels
            gen_z: Generator latent inputs
            gen_c: Generator label inputs
            sync: Whether to sync gradients
            gain: Gradient scaling factor
            indices: Indices of real images in the dataset (for precomputed embeddings)
        """
        # Call the parent class's accumulate_gradients
        super().accumulate_gradients(stage, real_img, real_c, gen_z, gen_c, sync, gain)
        
        # Apply CLIP similarity loss during G_main stage if clip_weight > 0
        if stage in ["G_main", "G_both"] and self.clip_weight > 0:
            with torch.autograd.profiler.record_function("CLIP_similarity_forward"):
                # Generate images if we haven't already
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                
                # Get embeddings for generated images
                gen_emb = self.get_clip_embeddings(gen_img)
                
                # Get embeddings for real images (either precomputed or computed on the fly)
                if self.use_precomputed_embeddings and self.precomputed_embeddings is not None and indices is not None:
                    real_emb = self.get_precomputed_embeddings(indices, gen_img.shape[0])
                else:
                    real_emb = self.get_clip_embeddings(real_img)
                
                # Compute CLIP similarity loss
                clip_loss = self.compute_clip_similarity(real_emb, gen_emb)
                clip_loss = clip_loss * self.clip_weight
                
                # Report the loss
                training_stats.report("Loss/CLIP/similarity", clip_loss)
                training_stats.report("Loss/G/clip", clip_loss)
                
            with torch.autograd.profiler.record_function("CLIP_similarity_backward"):
                clip_loss.mul(gain).backward() 