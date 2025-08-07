"""
Fast evaluation script with keyboard interrupt handling
Evaluates model on a subset of data for quick feedback
"""

import torch
import signal
import sys
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

class EvaluationInterrupt(Exception):
    """Custom exception for graceful evaluation interruption"""
    pass

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully during evaluation"""
    print("\nEvaluation interrupted by user. Calculating BLEU with current samples...")
    raise EvaluationInterrupt()

def fast_evaluate(model, dataloader, tokenizer, device, max_samples=100, max_gen_length=16):
    """
    Fast evaluation with keyboard interrupt handling
    
    Args:
        model: The model to evaluate
        dataloader: Validation dataloader  
        tokenizer: Tokenizer
        device: Device to run on
        max_samples: Maximum samples to evaluate (default: 100 for speed)
        max_gen_length: Maximum generation length (default: 16 for speed)
    
    Returns:
        Average BLEU score
    """
    # Set up signal handler for Ctrl+C
    original_handler = signal.signal(signal.SIGINT, signal_handler)
    
    model.eval()
    references = []
    hypotheses = []
    samples_processed = 0
    
    try:
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Fast Evaluation", total=min(len(dataloader), (max_samples + dataloader.batch_size - 1) // dataloader.batch_size))
            
            for batch_idx, (images, input_ids, _) in enumerate(progress_bar):
                images = images.to(device)
                batch_size = images.size(0)
                
                # Limit batch size if needed
                remaining_samples = max_samples - samples_processed
                if remaining_samples <= 0:
                    break
                    
                actual_batch_size = min(batch_size, remaining_samples)
                images = images[:actual_batch_size]
                input_ids = input_ids[:actual_batch_size]
                
                # Encode images once
                img_feats = model.encoder(images)
                
                # Generate captions
                generated_ids = torch.full((actual_batch_size, 1), tokenizer.bos_token_id, device=device)
                
                for step in range(max_gen_length):
                    logits = model.decoder(img_feats, generated_ids)
                    next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
                    
                    # Early stopping if all sequences have EOS
                    if (next_tokens == tokenizer.eos_token_id).all():
                        break
                
                # Process captions
                for i in range(actual_batch_size):
                    gen_caption = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                    ref_caption = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    
                    # Skip empty captions
                    if gen_caption.strip() and ref_caption.strip():
                        hypotheses.append(gen_caption.split())
                        references.append([ref_caption.split()])
                    
                    samples_processed += 1
                
                # Update progress
                progress_bar.set_postfix({
                    'samples': samples_processed,
                    'target': max_samples
                })
                
                if samples_processed >= max_samples:
                    break
                    
    except EvaluationInterrupt:
        print(f"Evaluation interrupted. Processed {samples_processed} samples.")
    
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)
    
    # Calculate BLEU scores
    if not references or not hypotheses:
        print("No valid samples to evaluate")
        return 0.0
    
    bleu_scores = []
    for ref, hyp in zip(references, hypotheses):
        try:
            bleu = sentence_bleu(ref, hyp)
            bleu_scores.append(bleu)
        except:
            # Skip invalid samples
            continue
    
    if not bleu_scores:
        print("No valid BLEU scores calculated")
        return 0.0
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    print(f"Evaluated {len(bleu_scores)} valid samples")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    
    return avg_bleu

if __name__ == "__main__":
    print("Fast evaluation script - can be interrupted with Ctrl+C")
    # This would be called from main.py or as standalone