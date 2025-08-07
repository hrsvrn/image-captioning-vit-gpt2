import torch
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

def evaluate(model, dataloader, tokenizer, device, max_samples=500):
    """
    Evaluate model with BLEU scores
    
    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        tokenizer: Tokenizer
        device: Device to run on
        max_samples: Maximum number of samples to evaluate (for speed)
    """
    model.eval()
    references = []
    hypotheses = []
    
    samples_processed = 0
    max_samples_per_batch = max_samples // len(dataloader) if len(dataloader) > 0 else max_samples
    
    with torch.no_grad():
        for batch_idx, (images, input_ids, _) in enumerate(tqdm(dataloader, desc="Evaluating", total=min(len(dataloader), max_samples//dataloader.batch_size))):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Encode images once
            img_feats = model.encoder(images)
            
            # Generate captions for entire batch
            generated_ids = torch.full((batch_size, 1), tokenizer.bos_token_id, device=device)
            
            for step in range(16):  # max generation length - reduced for faster validation
                logits = model.decoder(img_feats, generated_ids)
                next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
                
                # Check for EOS tokens - break if all sequences have EOS
                if (next_tokens == tokenizer.eos_token_id).all():
                    break
                    
                # Early stopping if most sequences have EOS
                eos_count = (next_tokens == tokenizer.eos_token_id).sum().item()
                if eos_count >= batch_size * 0.8:  # 80% of sequences have EOS
                    break
            
            # Process generated captions
            for i in range(batch_size):
                gen_caption = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                ref_caption = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                
                hypotheses.append(gen_caption.split())
                references.append([ref_caption.split()])
                
                samples_processed += 1
                
            # Break if we've processed enough samples
            if samples_processed >= max_samples:
                print(f"Evaluation stopped at {samples_processed} samples for efficiency")
                break
    
    # Calculate BLEU scores
    if not references or not hypotheses:
        print("No samples to evaluate")
        return 0.0
        
    bleu_scores = [sentence_bleu(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    print(f"Evaluated {len(bleu_scores)} samples")
    return avg_bleu