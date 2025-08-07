import torch
import torch.nn as nn
import timm
from transformers import GPT2LMHeadModel, GPT2Config

class ViTEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        # Use pretrained ViT from timm
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # num_classes=0 removes classifier
        self.feature_dim = self.vit.num_features  # 768 for base model
        
    def forward(self, x):
        # Extract features from pretrained ViT
        features = self.vit(x)  # (batch_size, 768)
        return features

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768):
        super().__init__()
        # Use pretrained GPT2 configuration
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_layer=12,
            n_head=12,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
        
        # Load pretrained GPT2 model
        self.gpt2 = GPT2LMHeadModel(config)
        
        # Load pretrained weights from GPT2-small
        pretrained_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Copy compatible weights
        self.gpt2.transformer.wte.weight.data[:pretrained_gpt2.config.vocab_size] = pretrained_gpt2.transformer.wte.weight.data
        self.gpt2.transformer.wpe.weight.data = pretrained_gpt2.transformer.wpe.weight.data
        
        # Copy transformer blocks
        for i in range(min(len(self.gpt2.transformer.h), len(pretrained_gpt2.transformer.h))):
            self.gpt2.transformer.h[i].load_state_dict(pretrained_gpt2.transformer.h[i].state_dict())
        
        # Copy layer norm
        self.gpt2.transformer.ln_f.load_state_dict(pretrained_gpt2.transformer.ln_f.state_dict())
        
        # Image projection layer to align ViT features with GPT2 embeddings
        self.image_proj = nn.Linear(d_model, d_model)
        
    def forward(self, encoded_img, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Project image features
        img_features = self.image_proj(encoded_img)  # (B, d_model)
        img_features = img_features.unsqueeze(1)  # (B, 1, d_model)
        
        # Get token embeddings
        token_embeddings = self.gpt2.transformer.wte(input_ids)  # (B, seq_len, d_model)
        
        # Concatenate image features with token embeddings
        # Image acts as the first "token"
        inputs_embeds = torch.cat([img_features, token_embeddings], dim=1)  # (B, seq_len+1, d_model)
        
        # Create position ids for the full sequence (including image)
        position_ids = torch.arange(0, seq_len + 1, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Forward through GPT2
        outputs = self.gpt2(inputs_embeds=inputs_embeds, position_ids=position_ids)
        
        # Return logits for text tokens only (skip the image token)
        return outputs.logits[:, 1:, :]  # (B, seq_len, vocab_size)

class ImageCaptionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = ViTEncoder(pretrained=True)  # Pretrained ViT
        self.decoder = CaptionDecoder(vocab_size)   # Pretrained GPT2-based decoder
        
    def forward(self, images, input_ids):
        img_embed = self.encoder(images)
        return self.decoder(img_embed, input_ids)


