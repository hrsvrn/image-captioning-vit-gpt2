import os
import torch
from torch.utils.data import DataLoader
import wandb
from dataset import CocoDataset, download_coco_dataset
from model import ImageCaptionModel
from utils import get_tokenizer, get_transforms
from train import train
from evaluate import evaluate
from safetensors.torch import save_file
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
BATCH_SIZE = 128# Increased for H100 80GB VRAM
EPOCHS = 3
DATA_DIR = "./coco_data"
IMAGE_DIR = os.path.join(DATA_DIR, "train2017")
ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations/captions_train2017.json")

# Setup
wandb.init(project="vit-gpt2-captioning")git
download_coco_dataset(DATA_DIR)
tokenizer = get_tokenizer()
dataset = CocoDataset(IMAGE_DIR, ANNOTATION_FILE, tokenizer, get_transforms())
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
model = ImageCaptionModel(vocab_size=len(tokenizer)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    avg_loss = train(model, dataloader, optimizer, tokenizer, DEVICE, scheduler)
    print(f"Average Training Loss: {avg_loss:.4f}")
    bleu = evaluate(model, dataloader, tokenizer, DEVICE, max_samples=1000)
    print(f"Validation BLEU Score: {bleu:.4f}")
    wandb.log({"val/bleu": bleu, "epoch": epoch + 1})
    
    # Save checkpoint every epoch
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'bleu_score': bleu
    }
    torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')

print("Training Completed.")

# Save final model
model_save = model.state_dict()
safetensor_path = "vit-gpt2-captioning.safetensors"
save_file(model_save, safetensor_path)
print(f"Model saved to {safetensor_path}")

# Optional: Upload to HuggingFace Hub
upload_to_hf = input("\nWould you like to upload the model to HuggingFace Hub? (y/n): ").lower().strip()
if upload_to_hf in ['y', 'yes']:
    repo_id = input("Enter HuggingFace repository ID (e.g., username/model-name): ").strip()
    if repo_id:
        try:
            from upload_to_hf import upload_model_to_hf
            import os
            
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                print("Please set HF_TOKEN environment variable or run 'huggingface-cli login'")
                print("You can also upload manually later using: python upload_to_hf.py")
            else:
                print("Uploading to HuggingFace Hub...")
                url = upload_model_to_hf(repo_id, safetensor_path, hf_token)
                print(f"Model uploaded successfully: {url}")
        except ImportError:
            print("Upload module not available. Use: python upload_to_hf.py")
        except Exception as e:
            print(f"Upload failed: {e}")
            print("You can upload manually later using: python upload_to_hf.py")
    else:
        print("No repository ID provided. Skipping upload.")
else:
    print("Model saved locally. You can upload later using: python upload_to_hf.py")

