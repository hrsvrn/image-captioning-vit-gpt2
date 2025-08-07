"""
Configuration classes for ViT-GPT2 Image Captioning Model
Compatible with HuggingFace transformers configuration system
"""

from typing import Dict, Any
import json


class ViTGPT2Config:
    """Configuration class for ViT-GPT2 image captioning model"""
    
    def __init__(
        self,
        # Vision configuration
        image_size: int = 224,
        patch_size: int = 16,
        vision_embed_dim: int = 768,
        vision_depth: int = 12,
        vision_num_heads: int = 12,
        
        # Text configuration  
        vocab_size: int = 50257,
        text_embed_dim: int = 768,
        text_depth: int = 12,
        text_num_heads: int = 12,
        max_length: int = 512,
        
        # Training configuration
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        
        # Special tokens
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        
        **kwargs
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_embed_dim = vision_embed_dim
        self.vision_depth = vision_depth
        self.vision_num_heads = vision_num_heads
        
        self.vocab_size = vocab_size
        self.text_embed_dim = text_embed_dim
        self.text_depth = text_depth
        self.text_num_heads = text_num_heads
        self.max_length = max_length
        
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
        # Computed properties
        self.num_patches = (image_size // patch_size) ** 2
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model_type": "vit-gpt2-captioning",
            "architecture": "ImageCaptionModel",
            "vision_config": {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "embed_dim": self.vision_embed_dim,
                "depth": self.vision_depth,
                "num_heads": self.vision_num_heads,
                "num_patches": self.num_patches
            },
            "text_config": {
                "vocab_size": self.vocab_size,
                "embed_dim": self.text_embed_dim,
                "depth": self.text_depth,
                "num_heads": self.text_num_heads,
                "max_length": self.max_length
            },
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "torch_dtype": "float32",
            "transformers_version": "4.30.0"
        }
    
    def save_pretrained(self, save_directory: str):
        """Save config to directory"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary"""
        # Flatten nested configs
        kwargs = {}
        if "vision_config" in config_dict:
            vision = config_dict["vision_config"]
            kwargs.update({
                "image_size": vision.get("image_size", 224),
                "patch_size": vision.get("patch_size", 16),
                "vision_embed_dim": vision.get("embed_dim", 768),
                "vision_depth": vision.get("depth", 12),
                "vision_num_heads": vision.get("num_heads", 12),
            })
        
        if "text_config" in config_dict:
            text = config_dict["text_config"]
            kwargs.update({
                "vocab_size": text.get("vocab_size", 50257),
                "text_embed_dim": text.get("embed_dim", 768),
                "text_depth": text.get("depth", 12),
                "text_num_heads": text.get("num_heads", 12),
                "max_length": text.get("max_length", 512),
            })
        
        # Add other fields
        for key in ["dropout", "attention_dropout", "bos_token_id", "eos_token_id", "pad_token_id"]:
            if key in config_dict:
                kwargs[key] = config_dict[key]
        
        return cls(**kwargs)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        """Load config from pretrained model"""
        import os
        config_file = os.path.join(model_name_or_path, "config.json")
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Default configuration
DEFAULT_CONFIG = ViTGPT2Config()