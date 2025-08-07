"""
Example usage script for the ViT-GPT2 Image Captioning Model
Shows how to load the model from HuggingFace Hub and generate captions
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import argparse
from typing import Optional

# Import local modules
from model import ImageCaptionModel
from utils import get_transforms


def load_model_from_hf(repo_id: str, device: str = "cuda") -> tuple:
    """
    Load model from HuggingFace Hub
    
    Args:
        repo_id: HuggingFace repository ID
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer, transform)
    """
    print(f" Loading model from HuggingFace Hub: {repo_id}")
    
    # Download model file
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        cache_dir="./hf_cache"
    )
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = ImageCaptionModel(vocab_size=len(tokenizer))
    
    # Load weights
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Get transforms
    transform = get_transforms()
    
    print(f" Model loaded successfully on {device}")
    return model, tokenizer, transform


def load_local_model(model_path: str, device: str = "cuda") -> tuple:
    """
    Load model from local safetensor file
    
    Args:
        model_path: Path to local safetensor file
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer, transform)
    """
    print(f" Loading local model: {model_path}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = ImageCaptionModel(vocab_size=len(tokenizer))
    
    # Load weights
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Get transforms
    transform = get_transforms()
    
    print(f" Model loaded successfully on {device}")
    return model, tokenizer, transform


def generate_caption(
    model: ImageCaptionModel,
    image: torch.Tensor,
    tokenizer: GPT2Tokenizer,
    device: str = "cuda",
    max_length: int = 32,
    temperature: float = 1.0,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 0.95
) -> str:
    """
    Generate caption for an image
    
    Args:
        model: The image captioning model
        image: Preprocessed image tensor
        tokenizer: GPT2 tokenizer
        device: Device to run inference on
        max_length: Maximum caption length
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        
    Returns:
        Generated caption string
    """
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        # Encode image
        img_features = model.encoder(image)
        
        # Initialize with BOS token
        generated_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        
        for _ in range(max_length):
            # Get logits
            logits = model.decoder(img_features, generated_ids)
            next_token_logits = logits[0, -1, :] / temperature
            
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated tokens
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption


def caption_image(
    image_path: str,
    model_source: str,
    device: str = "cuda",
    max_length: int = 32,
    temperature: float = 1.0,
    do_sample: bool = True
) -> str:
    """
    Caption a single image
    
    Args:
        image_path: Path to the image file
        model_source: Either HF repo ID or local model path
        device: Device to run on
        max_length: Maximum caption length
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        
    Returns:
        Generated caption
    """
    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
        print(f" Loaded image: {image_path} ({image.size})")
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")
    
    # Load model
    if "/" in model_source and not model_source.endswith(('.safetensors', '.pt')):
        # Assume it's a HF repo ID
        model, tokenizer, transform = load_model_from_hf(model_source, device)
    else:
        # Assume it's a local path
        model, tokenizer, transform = load_local_model(model_source, device)
    
    # Preprocess image
    image_tensor = transform(image)
    
    # Generate caption
    print(" Generating caption...")
    caption = generate_caption(
        model, image_tensor, tokenizer, device, 
        max_length, temperature, do_sample
    )
    
    return caption


def main():
    parser = argparse.ArgumentParser(description="Generate captions for images using ViT-GPT2 model")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("model_source", help="HuggingFace repo ID or local model path")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda/cpu)")
    parser.add_argument("--max-length", type=int, default=32, help="Maximum caption length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
    
    args = parser.parse_args()
    
    try:
        caption = caption_image(
            image_path=args.image_path,
            model_source=args.model_source,
            device=args.device,
            max_length=args.max_length,
            temperature=args.temperature,
            do_sample=not args.greedy
        )
        
        print(f"\n Generated Caption: {caption}")
        
    except Exception as e:
        print(f" Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())