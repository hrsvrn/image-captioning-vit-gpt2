from torchvision import transforms
from transformers import GPT2Tokenizer

def get_tokenizer():
    tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token=tokenizer.eos_token
    return tokenizer


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])


