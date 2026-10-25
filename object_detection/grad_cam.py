import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Resolve target layer module dynamically
        if isinstance(target_layer, nn.Module):
            self.target_module = target_layer
        elif isinstance(target_layer, int):
            self.target_module = self.model.features[target_layer]
        elif isinstance(target_layer, str):
            layer_block = getattr(self.model, target_layer)
            # Pick the final conv layer of the residual block
            self.target_module = layer_block[-1].conv2
        else:
            raise TypeError("Unsupported target_layer type. Pass an int, str, or nn.Module.")

    def forward_hook(self, module, inputs, output):
        self.activations = output.detach().clone()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach().clone()

    def generate(self, X, target_class_index):

        self.model.eval()
        self.model.zero_grad()

        self.model.to(self.device)
        X = X.to(self.device)

        forward_handle = self.target_module.register_forward_hook(self.forward_hook)
        backward_handle = self.target_module.register_full_backward_hook(self.backward_hook)

        try:
            y_hat = self.model(X)
            score = y_hat[0, target_class_index] # 0 because we are using batch size 1

            score.backward()

            # Calculate weight(alpha) for each channel k
            weights = self.gradients.mean(dim=(0, 2, 3))

            # Multiply each channel k by weight(alpha) k
            activations = self.activations[0]
            for i in range(activations.shape[0]):
                activations[i] *= weights[i]

            # Sum across channels and apply ReLU
            cam = torch.sum(activations, dim=0)
            cam = F.relu(cam)

            # Normalize between 0 and 1
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)

            return cam.detach().cpu().numpy()

        finally:
            forward_handle.remove()
            backward_handle.remove()

