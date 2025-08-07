#!/usr/bin/env python3
"""
HuggingFace Hub Upload Script for ViT-GPT2 Image Captioning Model
Uploads safetensor files, model configuration, and creates a comprehensive model card.
"""

import os
import json
import torch
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from safetensors.torch import load_file
from transformers import GPT2Tokenizer
from model import ImageCaptionModel
import argparse
from datetime import datetime

def create_model_config():
    """Create model configuration for HuggingFace compatibility"""
    config = {
        "model_type": "vit-gpt2-captioning",
        "architecture": "ImageCaptionModel",
        "vision_config": {
            "image_size": 224,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "num_patches": 196  # (224/16)^2
        },
        "text_config": {
            "vocab_size": 50257,  # GPT2 vocab size
            "d_model": 768,
            "num_layers": 12,
            "num_heads": 12,
            "max_length": 512
        },
        "torch_dtype": "float32",
        "transformers_version": "4.30.0",
        "framework": "pytorch"
    }
    return config

def create_model_card(repo_id, model_size_mb, training_info=None):
    """Create a comprehensive model card"""
    
    model_card = f"""---
license: apache-2.0
base_model: gpt2
tags:
- image-captioning
- computer-vision
- pytorch
- safetensors
- vit
- gpt2
- coco
language:
- en
datasets:
- coco
pipeline_tag: image-to-text
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/cat-3113513_1280.jpg
  example_title: Cat
---

# ViT-GPT2 Image Captioning Model

This model combines a Vision Transformer (ViT) encoder with a GPT-2 decoder for automatic image captioning.

## Model Description

- **Model Type**: Image-to-Text Generation
- **Architecture**: Vision Transformer + GPT-2 Decoder with Cross-Attention
- **Base Models**: Custom ViT + GPT-2
- **Training Dataset**: MS COCO Captions
- **Model Size**: {model_size_mb:.1f} MB
- **Framework**: PyTorch with SafeTensors

## Architecture Details

### Vision Encoder (ViT)
- Image size: 224x224
- Patch size: 16x16
- Embedding dimension: 768
- Transformer layers: 12
- Attention heads: 12
- Flash Attention for efficiency

### Text Decoder (GPT-2 based)
- Vocabulary size: 50,257 (GPT-2 tokenizer)
- Hidden dimension: 768
- Transformer layers: 12
- Attention heads: 12
- Maximum sequence length: 512

## Usage

```python
import torch
from PIL import Image
from transformers import GPT2Tokenizer
from safetensors.torch import load_file

# Load model
model_weights = load_file("model.safetensors")
# Note: You'll need the model definition from the original repository

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load and preprocess image
image = Image.open("path/to/image.jpg").convert("RGB")
# Apply appropriate transforms (resize to 224x224, normalize, etc.)

# Generate caption
with torch.no_grad():
    # Forward pass through model
    caption = model.generate(image)
    caption_text = tokenizer.decode(caption, skip_special_tokens=True)
    print(f"Generated caption: {{caption_text}}")
```

## Training Details

### Training Data
- **Dataset**: MS COCO 2017 Training Set
- **Images**: ~118K training images
- **Captions**: ~590K captions (5 per image)

### Training Configuration
- **Batch Size**: 128 (optimized for H100 80GB)
- **Learning Rate**: 2e-4 with cosine annealing
- **Optimizer**: AdamW (weight_decay=0.01, betas=(0.9, 0.95))
- **Mixed Precision**: Enabled (FP16)
- **Gradient Clipping**: max_norm=1.0
- **Hardware**: NVIDIA H100 80GB

### Performance Optimizations
- Flash Attention for memory efficiency
- Gradient checkpointing
- Persistent DataLoader workers
- Mixed precision training

## Evaluation

The model is evaluated using BLEU scores on the validation set.

## Limitations

- Trained only on MS COCO dataset (English captions)
- May not generalize well to images significantly different from COCO
- Caption length limited to 32 tokens during training
- Potential biases from the training dataset

## Ethical Considerations

This model may exhibit biases present in the MS COCO dataset. Users should be aware of potential limitations when applying to diverse image types or contexts not well-represented in the training data.

## Citation

If you use this model, please cite the original Vision Transformer and GPT-2 papers:

```bibtex
@article{{dosovitskiy2020vit,
  title={{An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale}},
  author={{Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others}},
  journal={{arXiv preprint arXiv:2010.11929}},
  year={{2020}}
}}

@article{{radford2019gpt2,
  title={{Language Models are Unsupervised Multitask Learners}},
  author={{Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya}},
  year={{2019}}
}}
```

## Model Card Contact

For questions about this model, please open an issue in the original repository.

---

*Model uploaded on {datetime.now().strftime('%Y-%m-%d')}*
"""
    
    return model_card

def get_model_size(model_path):
    """Get model size in MB"""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        return size_bytes / (1024 * 1024)  # Convert to MB
    return 0

def validate_safetensor_file(file_path):
    """Validate that the safetensor file can be loaded"""
    try:
        weights = load_file(file_path)
        print(f"Successfully loaded {len(weights)} tensors from {file_path}")
        return True
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return False

def upload_model_to_hf(
    repo_id: str,
    safetensor_path: str = "vit-gpt2-captioning.safetensors",
    token: str = None,
    private: bool = False,
    commit_message: str = "Upload ViT-GPT2 image captioning model"
):
    """
    Upload model to HuggingFace Hub
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        safetensor_path: Path to the safetensor file
        token: HuggingFace API token
        private: Whether to create a private repository
        commit_message: Commit message for the upload
    """
    
    print(f"Starting upload to HuggingFace Hub: {repo_id}")
    
    # Validate inputs
    if not os.path.exists(safetensor_path):
        raise FileNotFoundError(f"Safetensor file not found: {safetensor_path}")
    
    if not validate_safetensor_file(safetensor_path):
        raise ValueError(f"Invalid safetensor file: {safetensor_path}")
    
    # Initialize HF API
    api = HfApi(token=token)
    
    try:
        # Create repository
        print(f"Creating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True
        )
        
        # Get model size for model card
        model_size_mb = get_model_size(safetensor_path)
        
        # Create and save model configuration
        print("Creating model configuration...")
        config = create_model_config()
        config_path = "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create and save model card
        print("Creating model card...")
        model_card = create_model_card(repo_id, model_size_mb)
        readme_path = "README.md"
        with open(readme_path, 'w') as f:
            f.write(model_card)
        
        # Upload files
        files_to_upload = [
            (safetensor_path, "model.safetensors"),
            (config_path, "config.json"),
            (readme_path, "README.md")
        ]
        
        for local_path, repo_path in files_to_upload:
            print(f"Uploading {local_path} -> {repo_path}")
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                token=token,
                commit_message=f"{commit_message} - {repo_path}"
            )
        
        # Also upload model.py for reference
        if os.path.exists("model.py"):
            print("Uploading model.py for reference...")
            upload_file(
                path_or_fileobj="model.py",
                path_in_repo="modeling_vit_gpt2.py",
                repo_id=repo_id,
                token=token,
                commit_message=f"{commit_message} - model architecture"
            )
        
        # Clean up temporary files
        for temp_file in [config_path, readme_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"Successfully uploaded model to: https://huggingface.co/{repo_id}")
        print(f"Model size: {model_size_mb:.1f} MB")
        
        return f"https://huggingface.co/{repo_id}"
        
    except Exception as e:
        print(f"Upload failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Upload ViT-GPT2 model to HuggingFace Hub")
    parser.add_argument("repo_id", help="HuggingFace repository ID (e.g., username/model-name)")
    parser.add_argument("--model-path", default="vit-gpt2-captioning.safetensors", 
                       help="Path to the safetensor model file")
    parser.add_argument("--token", help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--commit-message", default="Upload ViT-GPT2 image captioning model",
                       help="Commit message for the upload")
    
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("No HuggingFace token provided. Please either:")
        print("   1. Pass --token YOUR_TOKEN")
        print("   2. Set HF_TOKEN environment variable")
        print("   3. Run 'huggingface-cli login' first")
        return
    
    try:
        url = upload_model_to_hf(
            repo_id=args.repo_id,
            safetensor_path=args.model_path,
            token=token,
            private=args.private,
            commit_message=args.commit_message
        )
        print(f"\nUpload complete! Your model is available at: {url}")
        
    except Exception as e:
        print(f"\nUpload failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())