import torch
import torch.nn as nn
from flash_attn.modules.mha import FlashSelfAttention

class FlashMHA(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.flash_attn = FlashSelfAttention(causal=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x_ln = self.ln(x)
        out = self.flash_attn(x_ln)
        with torch.autocast(device_type=x.device.type,dtype=torch.float16):
            out=self.flash_attn(x_ln)
        return self.proj(out)

class ViTEncoder(nn.Module):
    def __init__(self, image_size=224, patch_size=16, emb_dim=768, depth=12):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.linear = nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, emb_dim))

        self.transformer = nn.Sequential(*[FlashMHA(emb_dim) for _ in range(depth)])

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
    def __init__(self,vocab_size):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,768)
        self.decoder=nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768,nhead=12),num_layers=12
        )
        self.linear=nn.Linear(768,vocab_size)
    def forward(self,encoded_img,input_ids):
        x=self.embed(input_ids)
        encoded_img=encoded_img.unsqueeze(1).repeat(1,x.size(1),1)
        x=self.decoder(x,encoded_img)
        return self.linear(x)

class ImageCaptionModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.encoder=ViTEncoder()
        self.decoder=CaptionDecoder(vocab_size)
    def forward(self,images,input_ids):
        img_embed=self.encoder(images)
        return self.decoder(img_embed,input_ids)


