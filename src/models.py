from pathlib import Path
from typing import Union, Optional

import torchvision
import torch
import torch.nn as nn

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

def freeze_bn_running_stats(model: nn.Module) -> None:

    """
    Freezing BatchNorm parameters only freezes the learnable scale (gamma) and shift (beta), but the running mean and variance can still change in training mode.
    Set frozen BatchNorm layers to evaluation mode to keep their pretrained running statistics fixed.
    """

    for module in model.modules(): # Iterates over everything implementing nn.Module
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if all(not p.requires_grad for p in module.parameters()):
                module.eval() # For a BatchNorm layer, calling module.eval() stops its running mean and running variance from being updated during the forward pass