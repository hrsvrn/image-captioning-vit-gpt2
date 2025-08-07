
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import torch.nn.functional as F

def train(model, dataloader, optimizer, tokenizer, device, scheduler=None, max_grad_norm=1.0):
    """
    Training function with gradient clipping and learning rate scheduling
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        tokenizer: Tokenizer for handling special tokens
        device: Device to train on
        scheduler: Learning rate scheduler (optional)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
    """
    scaler = GradScaler('cuda')
    model.train()
    total_loss = 0
    
    for batch_idx, (images, input_ids, _) in enumerate(tqdm(dataloader, desc="Training")):
        images, input_ids = images.to(device), input_ids.to(device)
        optimizer.zero_grad()
        
        with autocast('cuda'):
            logits = model(images, input_ids[:, :-1])
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                input_ids[:, 1:].reshape(-1), 
                ignore_index=tokenizer.pad_token_id
            )
        
        scaler.scale(loss).backward()
        
        # GRADIENT CLIPPING - Add this before optimizer step
        scaler.unscale_(optimizer)  # Unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Log training metrics
        wandb.log({
            "train/loss": loss.item(),
            "train/learning_rate": optimizer.param_groups[0]['lr'],
            "train/batch": batch_idx
        })
    
    # LEARNING RATE SCHEDULING - Add this at the end of epoch
    if scheduler is not None:
        scheduler.step()
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss
