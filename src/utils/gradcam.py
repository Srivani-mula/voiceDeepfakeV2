import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        self.model.model.layer4.register_forward_hook(self.save_activation)
        self.model.model.layer4.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):

        output = self.model(input_tensor)
        self.model.zero_grad()

        loss = output[:, class_idx]
        loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]

        for i in range(pooled_gradients.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        heatmap = cv2.resize(heatmap, (200, 128))

        return heatmap