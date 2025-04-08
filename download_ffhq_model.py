import os
import gdown
import sys

def download_ffhq_ganformer_model(model_path):
    """Download the pretrained FFHQ GANsformer model"""
    
    print("Downloading pretrained FFHQ GANsformer model...")
    
    # Create directory for pretrained models if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # URL for the pretrained FFHQ GANsformer model
    # This is a placeholder URL - you'll need to replace it with the actual URL
    # for the pretrained FFHQ GANsformer model
    model_url = "https://drive.google.com/uc?id=1-b0vwevUQs6LI_EybdO8XJ5uYSx63vEa"
    
    output_path = model_path + "ffhq.pkl"
    
    # Check if model already exists
    if os.path.exists(output_path):
        print(f"Pretrained model already exists at {output_path}")
        return output_path
    
    try:
        # Download the model
        gdown.download(model_url, output_path, quiet=False)
        print(f"Downloaded pretrained model to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nManual download instructions:")
        print("1. Download the FFHQ GANsformer model from the official repository")
        print("2. Place it in the 'pretrained_models' directory as 'ffhq.pkl'")
        sys.exit(1)

if __name__ == "__main__":
    download_ffhq_ganformer_model("/opt/scratchspace/conditionalGansformer/ffhq_model/")  