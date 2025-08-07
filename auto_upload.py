#!/usr/bin/env python3
"""
Automated upload script for trained models
Use this for batch uploads or CI/CD pipelines
"""

import os
import sys
from upload_to_hf import upload_model_to_hf

def auto_upload(
    repo_id: str,
    model_path: str = "vit-gpt2-captioning.safetensors",
    hf_token: str = None
):
    """
    Automatically upload model without user interaction
    
    Args:
        repo_id: HuggingFace repository ID
        model_path: Path to safetensor file
        hf_token: HuggingFace token (if not set as env var)
    """
    
    # Get token
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        print("No HuggingFace token found!")
        print("Set HF_TOKEN environment variable or pass --token")
        return False
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
    
    try:
        print(f"Auto-uploading {model_path} to {repo_id}")
        url = upload_model_to_hf(
            repo_id=repo_id,
            safetensor_path=model_path,
            token=token,
            commit_message="Automated upload of trained ViT-GPT2 model"
        )
        print(f"Upload successful: {url}")
        return True
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_upload.py <repo_id> [model_path]")
        print("Example: python auto_upload.py username/vit-gpt2-captioning")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "vit-gpt2-captioning.safetensors"
    
    success = auto_upload(repo_id, model_path)
    sys.exit(0 if success else 1)