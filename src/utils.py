from typing import Sequence

import torch

from sklearn.metrics import recall_score, precision_score, f1_score

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

def calculate_metrics(y_true, y_pred) -> list[float]:

    """
    Calculate accuracy, recall, precision, and f1-score for a given set of predictions and targets.
    """
    return [
        recall_score(
            y_true, y_pred, average='macro', zero_division=0
        ),
        precision_score(
            y_true, y_pred, average='macro', zero_division=0
        ),
        f1_score(
            y_true, y_pred, average='macro', zero_division=0
        )
    ]

def denormalize(
        X: torch.Tensor,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:

    if X.ndim != 3:
        raise ValueError("Input tensor must have 3 dimensions (channels, height, width).")

    out = X.clone()

    mean = torch.as_tensor(mean, dtype=X.dtype, device=X.device).reshape(-1, 1, 1)
    std = torch.as_tensor(std, dtype=X.dtype, device=X.device).reshape(-1, 1, 1)

    out = out * std + mean

    return out

