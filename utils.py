from torchvision import transforms
from transformers import GPT2Tokenizer

def get_tokenizer():
    tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token=tokenizer.eos_token
    return tokenizer


def get_transforms():
    # Use ImageNet normalization for pretrained ViT
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])


