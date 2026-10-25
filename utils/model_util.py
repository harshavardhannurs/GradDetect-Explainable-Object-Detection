import sys
from pathlib import Path
from typing import Tuple, Union, Optional

import numpy as np

import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import recall_score, precision_score, f1_score

def load_model(model_name:str = 'vgg', trainable_blocks=2) -> nn.Module:

    """
    Loads pretrained VGG13 with batch normalization or ResNet18.
    Unfreezes the entire classification head and <trainable_blocks> number of convolution blocks.
    """

    match model_name:
        case 'vgg':
            vgg_layers = [0, 7, 14, 21, 28]

            vgg = torchvision.models.vgg13_bn(weights=torchvision.models.VGG13_BN_Weights.DEFAULT)
            for n, p in vgg.named_parameters():
                p.requires_grad = False
                if 'classifier' in n:
                    p.requires_grad = True

            start_idx = vgg_layers[-trainable_blocks]
            for n, p in getattr(vgg, 'features')[start_idx:].named_parameters():
                p.requires_grad = True

            vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, 80)
            return vgg
        case 'resnet':
            resnet_layers = ['layer1', 'layer2', 'layer3', 'layer4']

            resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            for n, p in resnet.named_parameters():
                p.requires_grad = False
                if 'fc' in n:
                    p.requires_grad = True
            for layer in resnet_layers[-trainable_blocks:]:
                for n, p in getattr(resnet, layer).named_parameters():
                    p.requires_grad = True

            resnet.fc = nn.Linear(resnet.fc.in_features, 80)
            return resnet
        case _:
            raise ValueError(f"Model {model_name} not found.")

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

def freeze_bn_running_stats(model: nn.Module) -> None:

    """
    Freezing BatchNorm parameters only freezes the learnable scale (gamma) and shift (beta), but the running mean and variance can still change in training mode.
    Set frozen BatchNorm layers to evaluation mode to keep their pretrained running statistics fixed.
    """

    for module in model.modules(): # Iterates over everything implementing nn.Module
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if all(not p.requires_grad for p in module.parameters()):
                module.eval() # For a BatchNorm layer, calling module.eval() stops its running mean and running variance from being updated during the forward pass

def save_model(model: nn.Module, path: Optional[Union[Path, str]] = None) -> None:

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
    except Exception as e:
        raise RuntimeError(f"Failed to save model to {path}: {e}") from e

def evaluate_model(model: nn.Module, val_ldr: DataLoader, device = 'cpu') \
        -> Tuple[float, float, float, float]:

    model.eval()
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    batch_val_losses = []
    val_targets = []
    val_predictions = []

    with torch.no_grad():
        for X, y in val_ldr:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)

            loss = loss_fn(y_hat, y)
            predictions = (torch.sigmoid(y_hat) >= 0.5).float()

            batch_val_losses.append(loss.item())

            val_targets.append(y.detach().cpu().numpy())
            val_predictions.append(predictions.detach().cpu().numpy())

    recall, precision, f1 = calculate_metrics(
        np.concatenate(val_targets, axis=0),
        np.concatenate(val_predictions, axis=0)
    )

    return np.mean(batch_val_losses, dtype=float), recall, precision, f1

def train_model(
        model,
        train_ldr: DataLoader,
        val_ldr: DataLoader,
        epochs: int = 100,
        device = 'cpu',
        optimizer = None,
        patience=5
) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Trains the model for a specified number of epochs using training and validation dataloaders.
    """

    model.to(device)

    if optimizer is None:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    epoch_train_loss = np.zeros(shape=(epochs,))
    epoch_train_metrics = np.zeros(shape=(epochs, 3)) # First column for recall, second for precision, and third for f1-score

    epoch_val_loss = np.zeros(shape=(epochs,))
    epoch_val_metrics = np.zeros(shape=(epochs, 3))

    best_val_loss = np.inf
    patience_counter = 0

    project_root = Path(__file__).resolve().parent.parent
    checkpoint_file = project_root / 'models' / f'{model.__class__.__name__}.pt'

    try:
        for epochi in range(epochs):
            model.train()
            freeze_bn_running_stats(model)

            batch_train_losses = []
            train_targets = []
            train_predictions = []

            for X, y in train_ldr:

                X = X.to(device)
                y = y.to(device)

                y_hat = model(X)
                predictions = (torch.sigmoid(y_hat) >= 0.5).float()

                loss = loss_fn(y_hat, y)

                batch_train_losses.append(loss.item())

                train_targets.append(y.detach().cpu().numpy())
                train_predictions.append(predictions.detach().cpu().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            train_loss = np.mean(batch_train_losses)
            epoch_train_loss[epochi] = train_loss

            epoch_train_metrics[epochi] = calculate_metrics(
                np.concatenate(train_targets, axis=0),
                np.concatenate(train_predictions, axis=0)
            )

            val_loss, recall, precision, f1 = evaluate_model(model, val_ldr, device)

            epoch_val_loss[epochi] = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_model(model, checkpoint_file)
            else:
                patience_counter += 1

            epoch_val_metrics[epochi] = [recall, precision, f1]

            sys.stdout.write(f'\r Epoch {epochi+1}/{epochs}')
            sys.stdout.flush()

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epochi + 1} epochs.")
                completed = epochi + 1
                if checkpoint_file.exists():
                    model.load_state_dict(torch.load(checkpoint_file, map_location=device, weights_only=True))

                """
                weights_only: A security guard. By setting weights_only=True, PyTorch restricts deserialization strictly to raw tensors and basic containers (like dictionaries and lists), preventing execution of arbitrary or malicious Python code hidden inside a checkpoint file.
                map_location: Overrides the original hardware tag (e.g., 'cuda:0') baked into the checkpoint, safely redirecting tensor allocation to your target device ('cpu', 'cuda', 'mps') at read-time.
                This eliminates device-mismatch crashes (e.g., opening GPU-saved weights on a device without that specific GPU) and allows 
                cross-device remapping via a string, torch.device, or a custom device-mapping dictionary.
                """

                return (
                    model,
                    epoch_train_loss[:completed],
                    epoch_train_metrics[:completed],
                    epoch_val_loss[:completed],
                    epoch_val_metrics[:completed],
                )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Returning best model checkpoint and statistics.")

    if checkpoint_file.exists():
        model.load_state_dict(torch.load(checkpoint_file, map_location=device, weights_only=True))

    return (
            model,
            epoch_train_loss,
            epoch_train_metrics,
            epoch_val_loss,
            epoch_val_metrics
        )