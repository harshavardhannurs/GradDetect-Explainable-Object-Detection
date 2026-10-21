from pathlib import Path
import json
from collections import defaultdict

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

COCO_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

class COCODataset(Dataset):

    """
    img_dir: Path to the directory of images (e.g., .../train2017)
    ann_file: Path to instances JSON (e.g., .../instances_train2017.json)
    transform: torchvision v2 transform pipeline
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
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        target = torch.zeros(size=(self.num_classes, 1), dtype=torch.float32)
        categories = self.image_id_to_categories[img_id]

        for cat_id in categories:
            if cat_id in self.cat_to_idx:
                target[self.cat_to_idx[cat_id]] = 1.0

        return img, target

    def __len__(self):
        return len(self.image_ids)

def get_dataloader(dataset: Dataset, batch_size=64, shuffle=False, drop_last=False) -> Tuple[DataLoader, DataLoader]:

    """
    Create a dataloader for a given dataset
    """

    data_ldr = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return data_ldr