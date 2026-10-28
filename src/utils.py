from typing import Sequence

import torch

from sklearn.metrics import recall_score, precision_score, f1_score

COCO_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    47  : 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
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

