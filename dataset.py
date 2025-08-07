import os
import json
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets.utils import download_and_extract_archive

class CocoDataset(Dataset):
    def __init__(self, image_dir, annotation_file, tokenizer, transform=None, max_length=32):
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.image_id_to_filename = {x["id"]: x["file_name"] for x in annotations["images"]}
        self.entries = annotations["annotations"]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image_path = os.path.join(self.image_dir, self.image_id_to_filename[entry["image_id"]])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = entry["caption"]
        tokens = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        return image, tokens.input_ids.squeeze(), tokens.attention_mask.squeeze()

def download_coco_dataset(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    download_and_extract_archive("http://images.cocodataset.org/zips/train2017.zip", data_dir)
    download_and_extract_archive("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", data_dir)
