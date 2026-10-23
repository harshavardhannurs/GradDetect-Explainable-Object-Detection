import sys
from pathlib import Path

import numpy as np

import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def load_model(model_name:str = 'vgg') -> nn.Module:

    """
    Loads pretrained VGG13 with batch normalization or ResNet18
    """

    match model_name:
        case 'vgg':
            vgg = torchvision.models.vgg13_bn(weights=torchvision.models.VGG13_BN_Weights.DEFAULT)
            for n, p in vgg.named_parameters():
                p.requires_grad = False
                if 'classifier' in n:
                    p.requires_grad = True
            vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, 80)
            return vgg
        case 'resnet':
            resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            for n, p in resnet.named_parameters():
                p.requires_grad = False
                if 'layer4.1' in n or 'fc' in n:
                    p.requires_grad = True
            resnet.fc = nn.Linear(resnet.fc.in_features, 80)
            return resnet
        case _:
            raise ValueError(f"Model {model_name} not found.")

def calculate_metrics(y_true, y_pred):

    """
    Calculate accuracy, recall, precision, and f1-score for a given set of predictions and targets.
    """
    return [
        accuracy_score(y_true, y_pred),
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

def train_model(model, train_ldr: DataLoader, val_ldr: DataLoader, epochs: int = 100, device = 'cpu'):

    """
    Trains the model for a specified number of epochs using training and validation dataloaders.
    """

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    epoch_train_loss = np.zeros(shape=(epochs,))
    epoch_train_metrics = np.zeros(shape=(epochs, 4)) # First column for accuracy, second for recall, third for precision, fourth for f1-score

    epoch_val_loss = np.zeros(shape=(epochs,))
    epoch_val_metrics = np.zeros(shape=(epochs, 4))

    for epochi in range(epochs):
        model.train()

        train_losses = []
        train_targets = []
        train_predictions = []

        for X, y in train_ldr:

            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)
            predictions = (torch.sigmoid(y_hat) >= 0.5).float()

            loss = loss_fn(y_hat, y)

            train_losses.append(loss.item())

            train_targets.append(y.cpu().numpy())
            train_predictions.append(predictions.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_train_loss[epochi] = np.mean(train_losses)

        epoch_train_metrics[epochi] = calculate_metrics(
            np.concatenate(train_targets, axis=0),
            np.concatenate(train_predictions, axis=0)
        )

        model.eval()

        val_losses = []
        val_targets = []
        val_predictions = []

        with torch.no_grad():
            for X, y in val_ldr:

                X, y = X.to(device), y.to(device)
                y_hat = model(X)

                loss = loss_fn(y_hat, y)
                predictions = (torch.sigmoid(y_hat) >= 0.5).float()

                val_losses.append(loss.item())

                val_targets.append(y.cpu().numpy())
                val_predictions.append(predictions.cpu().numpy())

        epoch_val_loss[epochi] = np.mean(val_losses)

        epoch_val_metrics[epochi] = calculate_metrics(
            np.concatenate(val_targets, axis=0),
            np.concatenate(val_predictions, axis=0)
        )

        sys.stdout.write(f'\r Epoch {epochi+1}/{epochs}')
        sys.stdout.flush()

    return model, epoch_train_loss, epoch_train_metrics, epoch_val_loss, epoch_val_metrics

def save_model(model: nn.Module, path: Path=None):

    """
    Saves the model to a specified path. If no path is provided, the model is saved to the 'models' directory.
    """

    if path is None:
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / 'models' / f'{model.__class__.__name__}.pt'
    else:
        path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.save(model.state_dict(), path)
        print(f'Model saved to {path}')
    except Exception as e:
        raise RuntimeError(f"Failed to save model to {path}: {e}") from e