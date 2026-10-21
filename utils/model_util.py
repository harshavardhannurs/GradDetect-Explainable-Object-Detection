import utils.data_util as du
import sys
from pathlib import Path

import numpy as np
import torchvision
import torch
import torch.nn as nn

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

def train_model(model, train_ldr: Dalaloader, val_ldr: Dalaloader, epochs: int = 100, device = 'cpu'):

    """
    Trains the model for a specified number of epochs using training and validation dataloaders.
    """

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    epoch_loss = []
    epoch_train_acc = []
    epoch_val_acc = []

    for epochi in range(epochs):
        model.train()

        batch_loss = []
        batch_acc = []

        for X, y in train_ldr:

            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)
            predictions = (torch.sigmoid(y_hat) >= 0.5).float()
            acc = (predictions == y).float().mean()
            batch_acc.append(acc.item())

            loss = loss_fn(y_hat, y)
            batch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss.append(np.mean(batch_loss))
        epoch_train_acc.append(np.mean(batch_acc))

        model.eval()
        with torch.no_grad():
            X, y = next(iter(val_ldr))
            X, y = X.to(device), y.to(device)
            y_hat = model.to(device)
        predictions = (torch.sigmoid(y_hat) >= 0.5).float()
        acc = (predictions == y).float().mean()
        epoch_val_acc.append(acc.item())

        sys.stdout.write(f'\r Epoch {epochi+1}/{epochs}')

    return model, epoch_loss, epoch_train_acc, epoch_val_acc

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