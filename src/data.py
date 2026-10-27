from pathlib import Path
import json
from collections import defaultdict

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

class COCODataset(Dataset):

    """
    Loads image and annotation files from a COCO dataset
    """

    def __init__(
            self,
            img_dir,
            ann_file,
            transform=None
    ):
        super().__init__()

        self.img_dir = Path(img_dir)
        self.ann_file = Path(ann_file)

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Image directory {self.img_dir} does not exist.")
        if not self.ann_file.is_file():
            raise FileNotFoundError(f"Annotation file {self.ann_file} does not exist.")

        if transform is None:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]) # Necessary transformations only
        else:
            self.transform = transform

        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        cat_ids = sorted([c.get("id") for c in coco_data["categories"]])
        self.cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

        self.num_classes = len(cat_ids)

        self.image_id_to_filename = {img.get("id"): img.get("file_name") for img in coco_data["images"]}

        self.image_id_to_categories = defaultdict(set)
        for ann in coco_data["annotations"]:
            if ann.get("iscrowd", 0) == 0:
                self.image_id_to_categories[ann["image_id"]].add(ann.get("category_id"))

        self.image_ids = list(self.image_id_to_filename.keys())

    def __getitem__(self, index):

        img_id = self.image_ids[index]

        filename = self.image_id_to_filename[img_id]
        img_path = self.img_dir / filename

        with Image.open(img_path) as img:
            img = self.transform(img.convert("RGB"))

        target = torch.zeros(size=(self.num_classes, ), dtype=torch.float32)
        categories = self.image_id_to_categories[img_id]

        for cat_id in categories:
            if cat_id in self.cat_to_idx:
                target[self.cat_to_idx[cat_id]] = 1.0

        return img, target

    def __len__(self):
        return len(self.image_ids)

def get_dataloader(dataset: Dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=8) -> DataLoader:

    """
    Create a dataloader for a given dataset
    """

    data_ldr = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return data_ldr