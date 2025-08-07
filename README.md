# ViT-GPT2 Image Captioning Model

A high-performance image captioning model that combines Vision Transformer (ViT) encoder with GPT-2 decoder for automatic image description generation.

## Features

- **State-of-the-art Architecture**: ViT encoder + GPT-2 decoder with cross-attention
- **Flash Attention**: Memory-efficient attention mechanism for faster training
- **H100 Optimized**: Configured for NVIDIA H100 80GB GPU
- **HuggingFace Integration**: Easy upload and sharing on HuggingFace Hub
- **SafeTensors**: Secure model serialization format
- **Mixed Precision**: FP16 training for faster convergence

## Requirements

```bash
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Flash Attention >= 2.0.0
- SafeTensors >= 0.3.0
- HuggingFace Hub >= 0.16.0

## Model Architecture

### Vision Encoder (ViT)
- Image size: 224×224
- Patch size: 16×16  
- Embedding dim: 768
- Layers: 12
- Attention heads: 12
- Flash Attention enabled

### Text Decoder (GPT-2 based)
- Vocabulary: 50,257 tokens (GPT-2)
- Hidden dim: 768
- Layers: 12
- Attention heads: 12
- Max sequence length: 512

## Training

### Quick Start
```bash
python main.py
```

### Training Configuration
- **Batch Size**: 128 (optimized for H100 80GB)
- **Learning Rate**: 2e-4 with cosine annealing
- **Optimizer**: AdamW (weight_decay=0.01)
- **Dataset**: MS COCO 2017 (~118K images, ~590K captions)
- **Epochs**: 10 (configurable)

### Training Features
- Automatic COCO dataset download
- Gradient clipping (max_norm=1.0)
- Model checkpointing every epoch
- Weights & Biases logging
- Mixed precision training
- Learning rate scheduling

## Model Saving & Loading

### Save Model
The model is automatically saved as SafeTensors format:
```python
# Automatic saving after training
save_file(model.state_dict(), "vit-gpt2-captioning.safetensors")
```

### Load Model
```python
from safetensors.torch import load_file
from model import ImageCaptionModel

# Load model
model = ImageCaptionModel(vocab_size=50257)
state_dict = load_file("vit-gpt2-captioning.safetensors")
model.load_state_dict(state_dict)
model.eval()
```

## HuggingFace Hub Integration

### Upload Model
After training, upload your model to HuggingFace Hub:

#### Method 1: Interactive Upload (during training)
The training script will prompt you to upload after completion.

#### Method 2: Manual Upload
```bash
python upload_to_hf.py username/model-name --model-path vit-gpt2-captioning.safetensors
```

#### Method 3: Automated Upload
```bash
export HF_TOKEN="your_hf_token_here"
python auto_upload.py username/vit-gpt2-captioning
```

### Upload Features
- Automatic model card generation
- Configuration files creation
- Model architecture file upload
- Comprehensive documentation
- Metadata and usage examples

## Inference

### Generate Captions
```python
# Load from HuggingFace Hub
python example_usage.py image.jpg username/vit-gpt2-captioning

# Load from local file
python example_usage.py image.jpg vit-gpt2-captioning.safetensors
```

### Advanced Inference
```python
from example_usage import caption_image

caption = caption_image(
    image_path="path/to/image.jpg",
    model_source="username/vit-gpt2-captioning",  # or local path
    device="cuda",
    max_length=32,
    temperature=0.8,
    do_sample=True
)
print(f"Generated caption: {caption}")
```

### Inference Parameters
- **max_length**: Maximum caption length (default: 32)
- **temperature**: Sampling temperature (default: 1.0)
- **do_sample**: Use sampling vs greedy decoding (default: True)
- **top_k**: Top-k sampling (default: 50)
- **top_p**: Nucleus sampling (default: 0.95)

## Evaluation

The model is evaluated using BLEU scores:
```python
from evaluate import evaluate

bleu_score = evaluate(model, dataloader, tokenizer, device)
print(f"BLEU Score: {bleu_score:.4f}")
```

## Configuration

### Model Configuration
Edit `config.py` to modify model architecture:
```python
from config import ViTGPT2Config

config = ViTGPT2Config(
    image_size=224,
    patch_size=16,
    vision_embed_dim=768,
    vocab_size=50257,
    # ... other parameters
)
```

### Training Configuration
Modify hyperparameters in `main.py`:
```python
BATCH_SIZE = 128      # Adjust for your GPU
EPOCHS = 10           # Number of training epochs
DEVICE = "cuda"       # Training device
```

## Project Structure

```
image-captioning/
├── main.py              # Main training script
├── model.py             # Model architecture
├── dataset.py           # COCO dataset handling
├── train.py             # Training loop
├── evaluate.py          # Evaluation metrics
├── utils.py             # Utility functions
├── config.py            # Model configuration
├── upload_to_hf.py      # HuggingFace upload script
├── auto_upload.py       # Automated upload
├── example_usage.py     # Inference examples
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Hardware Optimization

### H100 80GB Optimizations
- Batch size: 128 (can go up to 256)
- DataLoader workers: 16 with persistent workers
- Flash Attention for memory efficiency
- Mixed precision training
- Gradient checkpointing available

### Memory Usage
- Model parameters: ~150M
- Training memory: ~20-30GB with batch size 128
- Inference memory: ~2-3GB per image

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size in main.py
   BATCH_SIZE = 64  # or 32
   ```

2. **Flash Attention Not Available**
   ```bash
   # Install flash attention
   pip install flash-attn --no-build-isolation
   ```

3. **Dataset Download Issues**
   ```python
   # Manual download in dataset.py
   download_coco_dataset("./coco_data")
   ```

4. **HuggingFace Token Issues**
   ```bash
   # Login to HuggingFace CLI
   huggingface-cli login
   
   # Or set environment variable
   export HF_TOKEN="your_token_here"
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words"
- **GPT-2**: Radford et al., "Language Models are Unsupervised Multitask Learners"
- **Flash Attention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention"
- **MS COCO Dataset**: Lin et al., "Microsoft COCO: Common Objects in Context"

## Performance

### Training Metrics
- Training time: ~8-12 hours on H100 (10 epochs)
- BLEU-4 score: ~0.25-0.30 (typical range)
- Memory usage: ~25GB training, ~3GB inference

### Benchmarks
- Images/second: ~400-500 (training)
- Caption generation: ~0.1-0.2s per image (inference)
- Model size: ~600MB (SafeTensors format)

---

For questions or issues, please open a GitHub issue or contact the maintainers.