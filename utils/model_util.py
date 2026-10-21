import utils.data_util as du
import sys
from pathlib import Path

import numpy as np
import torchvision
import torch
import torch.nn as nn

def load_model() -> nn.Module:

    """
    Future work: Add support to load other pre-trained models. Currently only VGG13 with batch normalization is supported.
    """

    vgg = torchvision.models.vgg13_bn(weights=torchvision.models.VGG13_BN_Weights.DEFAULT)

    for n, p in vgg.named_parameters():
        p.requires_grad = False
        if 'classifier.6' in n:
            p.requires_grad = True
        print(n, p.requires_grad)
    vgg.classifier[6] = nn.Linear(4096, 80)

    return vgg

def train_model(model, train_ldr: Dalaloader, test_ldr: Dalaloader, epochs: int = 100, device = 'cpu'):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    epoch_loss = []
    epoch_acc = []

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
        epoch_acc.append(np.mean(batch_acc))

        sys.stdout.write(f'\r Epoch {epochi+1}/{epochs}')

    return model, epoch_loss, epoch_acc

def save_model(model: nn.Module, path: Path=None):

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