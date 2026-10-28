from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

from src.utils import destandardize_image

class GradCAM:

    def __init__(self, model, target_layer, device='cpu'):

        """
        target_layer can be:
        - int: layer index inside model.features (e.g., 31 for VGG13_BN)
        - str: layer attribute name (e.g., 'layer4' for ResNet18)
        - nn.Module: directly passing the target module (e.g. resnet.layer4[-1].conv2)
        """

        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None

        if isinstance(target_layer, nn.Module):
            self.target_module = target_layer
        elif isinstance(target_layer, int):
            self.target_module = self.model.features[target_layer]
        elif isinstance(target_layer, str):
            layer_block = getattr(self.model, target_layer)
            self.target_module = layer_block[-1].conv2
        else:
            raise TypeError("Unsupported target_layer type. Pass an int, str, or nn.Module.")

    def forward_hook(self, module, inputs, output):
        self.activations = output.detach().clone()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach().clone()

    def generate(self, X):

        self.model.eval()

        self.model.to(self.device)
        X = X.to(self.device)

        forward_handle = self.target_module.register_forward_hook(self.forward_hook)
        backward_handle = self.target_module.register_full_backward_hook(self.backward_hook)

        try:
            y_hat = self.model(X)

            predicted_class_ids = torch.where(
                torch.sigmoid(y_hat[0]) >= 0.5
            )[0].detach().cpu()

            predicted_probs = torch.sigmoid(y_hat[0])[predicted_class_ids].detach().cpu()

            class_cams = []

            for target_id in predicted_class_ids:

                self.model.zero_grad()

                score = y_hat[0, target_id] # 0 because we are using batch size 1

                score.backward(retain_graph=True)
                # retain_graph=True, because after first backward pass, PyTorch deletes the computation graph to free up the memory.

                # Calculate weight(alpha) for each channel k
                weights = self.gradients.mean(dim=(0, 2, 3))

                # Multiply each channel k by weight(alpha) k
                activations = self.activations[0]

                # Sum across channels and apply ReLU
                cam = (activations * weights[:, None, None]).sum(dim=0)
                cam = F.relu(cam)

                # Normalize between 0 and 1
                cam_min, cam_max = cam.min(), cam.max()
                if cam_max > cam_min:
                    cam = (cam - cam_min) / (cam_max - cam_min)

                class_cams.append(cam.detach().cpu())

            return class_cams, predicted_class_ids, predicted_probs

        finally:
            forward_handle.remove()
            backward_handle.remove()

def detect_objects(
        img,
        model,
        threshold=.5,
        device='cpu'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    `img` is assumed to be standardized.
    """

    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input image must be of type torch.Tensor.")

    if img.ndim != 4:
        raise ValueError(f"Input image must have 4 dimensions (1, 3, H, W).")

    target_layer = None
    if model.__class__.__name__ == 'VGG':
        target_layer = 31
    elif model.__class__.__name__ == 'ResNet':
        target_layer = 'layer4'

    gc = GradCAM(model, target_layer=target_layer, device=device)

    class_cams, predicted_class_ids, predicted_probs = gc.generate(img)

    resize_cam = v2.Resize((img.shape[-2], img.shape[-1]))

    objs = np.zeros(shape=(len(class_cams), 4))

    for i in range(len(class_cams)):

        cam = class_cams[i]
        cam = cam.unsqueeze(dim=0)
        cam = resize_cam(cam)
        cam = cam > threshold
        cam = cam.squeeze(dim=0)

        ys, xs = cam.nonzero(as_tuple=True)

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        objs[i, :] = [x_min, y_min, x_max, y_max]

    return (
        destandardize_image(img.squeeze(dim=0)).permute(1, 2, 0).detach().cpu().numpy(),
        objs,
        predicted_class_ids.numpy(),
        predicted_probs.numpy()
    )
