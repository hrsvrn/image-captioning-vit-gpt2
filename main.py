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
DEVICE = torch.device("cuda")
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "./coco_data"
IMAGE_DIR = os.path.join(DATA_DIR, "train2017")
ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations/captions_train2017.json")

# Setup
wandb.init(project="vit-gpt2-captioning")
download_coco_dataset(DATA_DIR)
tokenizer = get_tokenizer()
dataset = CocoDataset(IMAGE_DIR, ANNOTATION_FILE, tokenizer, get_transforms())
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
model = ImageCaptionModel(vocab_size=len(tokenizer)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    avg_loss = train(model, dataloader, optimizer, tokenizer, DEVICE)
    print(f"Average Training Loss: {avg_loss:.4f}")
    bleu = evaluate(model, dataloader, tokenizer, DEVICE)
    print(f"Validation BLEU Score: {bleu:.4f}")
    wandb.log({"val/bleu": bleu, "epoch": epoch + 1})

print("Training Completed.")
model_save=model.state_dict()
save_file(model_save,"vit-gpt2-captioning.safetensors")

