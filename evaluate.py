import torch
from nltk.translate.bleu_score import sentence_bleu

def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    references = []
    hypotheses = []
    with torch.no_grad():
        for images, input_ids, _ in dataloader:
            images = images.to(device)
            img_feats = model.encoder(images)
            for img_feat in img_feats:
                generated = [tokenizer.bos_token_id]
                for _ in range(32):
                    input_tensor = torch.tensor(generated).unsqueeze(0).to(device)
                    logits = model.decoder(img_feat.unsqueeze(0), input_tensor)
                    next_token = logits[0, -1].argmax().item()
                    if next_token == tokenizer.eos_token_id:
                        break
                    generated.append(next_token)
                caption = tokenizer.decode(generated, skip_special_tokens=True)
                hypotheses.append(caption.split())
                references.append([tokenizer.decode(input_ids[0], skip_special_tokens=True).split()])
    bleu = sum(sentence_bleu(r, h) for r, h in zip(references, hypotheses)) / len(references)
    return bleu
