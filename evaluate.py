import torch
from nltk.translate.bleu_score import sentence_bleu

def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for images, input_ids, _ in dataloader:
            images = images.to(device)
            batch_size = images.size(0)
            
            # Encode images once
            img_feats = model.encoder(images)
            
            # Generate captions for entire batch
            generated_ids = torch.full((batch_size, 1), tokenizer.bos_token_id, device=device)
            
            for _ in range(32):  # max generation length
                logits = model.decoder(img_feats, generated_ids)
                next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
                
                # Check for EOS tokens (simplified)
                if (next_tokens == tokenizer.eos_token_id).all():
                    break
            
            # Process generated captions
            for i in range(batch_size):
                gen_caption = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                ref_caption = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                
                hypotheses.append(gen_caption.split())
                references.append([ref_caption.split()])
    
    # Calculate BLEU scores
    bleu_scores = [sentence_bleu(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0