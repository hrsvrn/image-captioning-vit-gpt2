import torch
import torch.nn as nn
from flash_attn.modules.mha import MHA as FlashSelfAttention

class FlashMHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.flash_attn = FlashSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        
    def forward(self, x):
        # Self-attention with residual
        x = x + self.flash_attn(self.ln1(x))
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x
class ViTEncoder(nn.Module):
    def __init__(self,num_heads=12, image_size=224, patch_size=16, emb_dim=768, depth=12):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.linear = nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, emb_dim))

        self.transformer = nn.Sequential(*[FlashMHA(emb_dim,num_heads) for _ in range(depth)])

    def forward(self, x):
        x = self.linear(x)
        x = x.flatten(2).transpose(1, 2)
        B, N, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :x.size(1), :]
        for block in self.transformer:
            x = block(x)
        return x[:, 0]

class CaptionDecoder(nn.Module):
    def __init__(self,vocab_size,d_model=768):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,768)
        self.pos_embedding=nn.Parameter(torch.randn(1,512,d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=12,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=12) 
        self.linear=nn.Linear(768,vocab_size)
    def forward(self, encoded_img, input_ids):
        B, seq_len = input_ids.shape
        
        # Token embeddings + positional embeddings
        x = self.embed(input_ids) + self.pos_embedding[:, :seq_len, :]
        
        # Image features as memory (cross-attention)
        memory = encoded_img.unsqueeze(1)  # (B, 1, d_model)
        
        # Create causal mask for autoregressive generation
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)
        
        x = self.decoder(x, memory, tgt_mask=tgt_mask)
        return self.linear(x)

class ImageCaptionModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.encoder=ViTEncoder()
        self.decoder=CaptionDecoder(vocab_size)
    def forward(self,images,input_ids):
        img_embed=self.encoder(images)
        return self.decoder(img_embed,input_ids)


