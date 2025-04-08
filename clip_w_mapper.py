import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
from tqdm import tqdm

sys.path.append('.')
from gansformer import loader

class TextEmbeddingDataset(Dataset):
    """Dataset of text prompts and their CLIP embeddings"""
    def __init__(self, text_prompts, clip_model, device='cuda'):
        self.text_prompts = text_prompts
        self.device = device
        
        # Tokenize and encode all prompts
        text_tokens = clip.tokenize(text_prompts).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            # Normalize features
            self.text_embeddings = text_features / text_features.norm(dim=1, keepdim=True)
    
    def __len__(self):
        return len(self.text_prompts)
    
    def __getitem__(self, idx):
        return {
            'prompt': self.text_prompts[idx],
            'embedding': self.text_embeddings[idx]
        }

class CLIPWMapper(nn.Module):
    """Maps CLIP text embeddings to offsets in W space"""
    def __init__(self, clip_dim=512, w_dim=512, k=1, hidden_dim=1024):
        super().__init__()
        
        self.clip_dim = clip_dim
        self.w_dim = w_dim
        self.k = k
        
        # MLP to map CLIP embeddings to W space offsets
        self.mapper = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, w_dim * k)
        )
    
    def forward(self, clip_embedding):
        batch_size = clip_embedding.shape[0]
        w_offset = self.mapper(clip_embedding)
        # Reshape to match GANsformer's W space
        w_offset = w_offset.reshape(batch_size, self.k, self.w_dim)
        return w_offset

class CLIPWMapperTrainer:
    """Trainer for the CLIP-W mapper"""
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Load the pretrained GANsformer
        print(f"Loading pretrained model from {model_path}")
        self.G = loader.load_network(model_path, 'G', None).to(device).eval()
        self.z_dim = self.G.z_dim
        self.w_dim = self.G.w_dim
        self.k = self.G.k
        
        # Load CLIP model
        print("Loading CLIP model")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Create mapper model
        self.mapper = CLIPWMapper(
            clip_dim=512,  # CLIP ViT-B/32 embedding dimension
            w_dim=self.w_dim,
            k=self.k
        ).to(device)
        
        print(f"Models initialized. Generator has {self.k} components, w_dim={self.w_dim}")
    
    def generate_w_samples(self, n_samples=1000, truncation_psi=0.7, seed=None):
        """Generate random W samples from the pretrained model"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        all_w = []
        batch_size = 16
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                curr_batch_size = min(batch_size, n_samples - i)
                z = torch.randn(curr_batch_size, self.k, self.z_dim).to(self.device)
                w = self.G.mapping(z, None)
                all_w.append(w)
        
        return torch.cat(all_w, dim=0)
    
    def train_mapper(self, text_prompts, w_mean=None, batch_size=8, epochs=50, lr=1e-4, save_path=None):
        """Train the mapper to generate W offsets from text prompts"""
        # Create dataset from text prompts
        dataset = TextEmbeddingDataset(text_prompts, self.clip_model, self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Compute mean W if not provided
        if w_mean is None:
            print("Computing mean W...")
            w_samples = self.generate_w_samples(1000)
            w_mean = w_samples.mean(0, keepdim=True)
        else:
            w_mean = w_mean.to(self.device)
        
        # Setup optimizer
        optimizer = optim.Adam(self.mapper.parameters(), lr=lr)
        
        # Training loop
        self.mapper.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                clip_embeddings = batch['embedding'].to(self.device)
                
                # Generate W offsets from CLIP embeddings
                w_offsets = self.mapper(clip_embeddings)
                
                # Apply offsets to mean W
                w = w_mean + w_offsets
                
                # Generate images
                images = self.G.synthesis(w)
                
                # Get CLIP image embeddings
                clip_images = []
                for img in images:
                    # Convert from [-1, 1] to [0, 1] and resize for CLIP
                    img_np = ((img.permute(1, 2, 0) + 1) / 2).clamp(0, 1)
                    img_resized = F.interpolate(img_np.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                    clip_images.append(img_resized.squeeze(0).permute(2, 0, 1))
                
                clip_images = torch.stack(clip_images).to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(clip_images)
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                
                # Compute similarity loss
                similarity = (image_features * clip_embeddings).sum(dim=1)
                loss = -similarity.mean()  # Maximize similarity (minimize negative similarity)
                
                # Add regularization to keep offsets small
                reg_loss = torch.mean(w_offsets.pow(2))
                loss = loss + 0.1 * reg_loss
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Save trained mapper
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.mapper.state_dict(), save_path)
            print(f"Mapper saved to {save_path}")
        
        return self.mapper, w_mean
    
    def generate_from_text(self, text_prompt, w_mean, seed=None, truncation_psi=0.7):
        """Generate an image from a text prompt using the trained mapper"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Get CLIP embedding for the text
        with torch.no_grad():
            text_token = clip.tokenize([text_prompt]).to(self.device)
            text_embedding = self.clip_model.encode_text(text_token)
            text_embedding = text_embedding / text_embedding.norm(dim=1, keepdim=True)
        
        # Generate W offset
        self.mapper.eval()
        with torch.no_grad():
            w_offset = self.mapper(text_embedding)
            
            # Apply offset to mean W with truncation
            w = w_mean + truncation_psi * w_offset
            
            # Generate image
            image = self.G.synthesis(w)
        
        return image[0]
    
    def save_image(self, image_tensor, output_path):
        """Save an image tensor to a file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert from [-1, 1] to [0, 1] and then to PIL Image
        img_np = ((image_tensor.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        # Save image
        pil_img.save(output_path)
        print(f"Saved image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CLIP-W mapper for text-guided GANsformer generation")
    parser.add_argument("--model", type=str, default="pretrained_models/ffhq.pkl", help="Path to pretrained GANsformer model")
    parser.add_argument("--train", action="store_true", help="Train the mapper model")
    parser.add_argument("--generate", action="store_true", help="Generate images from text prompts")
    parser.add_argument("--mapper-path", type=str, default="models/clip_w_mapper.pt", help="Path to save/load mapper model")
    parser.add_argument("--outdir", type=str, default="results/clip_mapper", help="Output directory for generated images")
    parser.add_argument("--prompt", type=str, default="a person with curly hair and glasses", help="Text prompt for image generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Check if CLIP is installed
    try:
        import clip
    except ImportError:
        print("CLIP is not installed. Please install it with:")
        print("pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)
    
    # Create trainer
    trainer = CLIPWMapperTrainer(args.model, args.device)
    
    if args.train:
        print("Training CLIP-W mapper...")
        
        # Example text prompts for training
        # You can expand this list or load from a file
        text_prompts = [
            "a person with glasses",
            "a person with a beard",
            "a smiling person",
            "a serious person",
            "a person with curly hair",
            "a person with straight hair",
            "a young person",
            "an elderly person",
            "a person with blonde hair",
            "a person with dark hair",
            "a surprised person",
            "a sad person",
            "a person with a hat",
            "a tired person",
            "a confident person"
        ]
        
        # Compute mean W
        w_samples = trainer.generate_w_samples(1000, seed=args.seed)
        w_mean = w_samples.mean(0, keepdim=True)
        
        # Train the mapper
        mapper, _ = trainer.train_mapper(
            text_prompts=text_prompts,
            w_mean=w_mean,
            batch_size=8,
            epochs=50,
            lr=1e-4,
            save_path=args.mapper_path
        )
        
        # Save the mean W
        os.makedirs(os.path.dirname(args.mapper_path), exist_ok=True)
        torch.save(w_mean, args.mapper_path.replace('.pt', '_w_mean.pt'))
        print(f"Mean W saved to {args.mapper_path.replace('.pt', '_w_mean.pt')}")
    
    if args.generate:
        print(f"Generating image from text prompt: {args.prompt}")
        
        # Load trained mapper
        if os.path.exists(args.mapper_path):
            trainer.mapper.load_state_dict(torch.load(args.mapper_path))
            print(f"Loaded mapper from {args.mapper_path}")
        else:
            print(f"Warning: Mapper not found at {args.mapper_path}. Using untrained mapper.")
        
        # Load mean W
        w_mean_path = args.mapper_path.replace('.pt', '_w_mean.pt')
        if os.path.exists(w_mean_path):
            w_mean = torch.load(w_mean_path).to(args.device)
            print(f"Loaded mean W from {w_mean_path}")
        else:
            print("Computing mean W...")
            w_samples = trainer.generate_w_samples(1000, seed=args.seed)
            w_mean = w_samples.mean(0, keepdim=True)
        
        # Create output directory
        os.makedirs(args.outdir, exist_ok=True)
        
        # Generate image from text prompt
        image = trainer.generate_from_text(args.prompt, w_mean, seed=args.seed)
        
        # Save the generated image
        output_path = os.path.join(args.outdir, f"{args.prompt.replace(' ', '_')}.png")
        trainer.save_image(image, output_path)
        
        print(f"Generated image saved to {output_path}") 