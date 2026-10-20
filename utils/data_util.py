from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Tuple

from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}

class TransformDataset(Dataset):
    """
    Use this class to create dataloaders for fine-tuning the resnet-18 model.
    """

    def __init__(self, tensors, transform=None):  # tensors: A tuple of tensors
        super().__init__()
        if tensors[0].size(0) != tensors[1].size(0):
            raise AssertionError("Size mismatch between tensors")

        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is None:
            x = self.tensors[0][index]
        else:
            x = self.transform(self.tensors[0][index])

        return x, self.tensors[1][index]

    def __len__(self):
        return self.tensors[0].size(0)

def extract_image_arrays(dataset: VOCDetection, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:

    image_paths = dataset.images
    num_imgs = len(image_paths)

    imgs = np.empty((num_imgs, target_size[1], target_size[0], 3), dtype=np.uint8)

    for i, path in enumerate(image_paths):
        with Image.open(path) as img:
            imgs[i] = np.asarray(img.convert('RGB').resize(target_size))

    return imgs

def extract_multihot_labels(dataset: VOCDetection):

    labels = np.zeros((len(dataset), len(VOC_CLASSES)), dtype=np.float32)

    for i, annotation in enumerate(dataset.annotations):

        tree = ET.parse(annotation)
        root = tree.getroot()

        multi_hot = np.zeros(len(VOC_CLASSES), dtype=np.uint8)

        for obj in root.findall('object'):

            difficult = obj.find('difficult')
            if difficult is not None and int(difficult.text) == 1:
                continue

            class_name = obj.find('name').text.strip().lower()
            if class_name in CLASS_TO_IDX:
                multi_hot[CLASS_TO_IDX[class_name]] = 1.0

        labels[i] = multi_hot

    return labels

def get_voc_train_test_split():

    """
    Loads, processes, and caches PASCAL VOC splits into .npy files
    These files are used to avoid re-downloading and re-processing the dataset
    and will be stored in the <project_root>/datasets/VOC directory.

    :returns: X_train, X_test, y_train, y_test
    """

    project_root = Path(__file__).resolve().parent.parent
    path = project_root / 'datasets' / 'VOC'

    x_train_path = path / 'X_train.npy'
    x_test_path = path / 'X_test.npy'
    y_train_path = path / 'y_train.npy'
    y_test_path = path / 'y_test.npy'

    cache_files = [x_train_path, x_test_path, y_train_path, y_test_path]
    if all(f.exists() for f in cache_files):
        return (
            np.load(x_train_path),
            np.load(x_test_path),
            np.load(y_train_path),
            np.load(y_test_path)
        )

    path.mkdir(parents=True, exist_ok=True)

    print('No cached VOC splits found. Downloading the dataset...')
    train_dataset = torchvision.datasets.VOCDetection(root=path, download=True, image_set='train', year='2012')
    test_dataset = torchvision.datasets.VOCDetection(root=path, download=True, image_set='val', year='2012')

    print('Creating VOC splits...')
    train_imgs = extract_image_arrays(train_dataset)
    test_imgs = extract_image_arrays(test_dataset)

    train_labels = extract_multihot_labels(train_dataset)
    test_labels = extract_multihot_labels(test_dataset)

    try:
        print('Caching VOC splits...')
        np.save(x_train_path, train_imgs)
        np.save(x_test_path, test_imgs)
        np.save(y_train_path, train_labels)
        np.save(y_test_path, test_labels)
    except OSError as e:
        raise RuntimeError(f"Failed to cache VOC dataset to {path}: {e}") from e

    return train_imgs, test_imgs, train_labels, test_labels

def get_data_loaders(X_train, X_test, y_train, y_test, batch_size=64, shuffle=True):

    """
    Transforms the numpy data into a dataloader for fine-tuning the resnet-18 model.
    @return: Train dataloader, Test dataloader
    """

    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=.5),
        v2.RandomAffine(degrees=15, translate=(.1, .1)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    X_train = torch.from_numpy(X_train).permute(0, 3, 1, 2)
    X_test = torch.from_numpy(X_test).permute(0, 3, 1, 2)
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    train_dataset = TransformDataset((X_train, y_train), transform=train_transform)
    test_dataset = TransformDataset((X_test, y_test), transform=test_transform)

    train_ldr = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_ldr = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_ldr, test_ldr







