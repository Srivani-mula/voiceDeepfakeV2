import torch
import torch.nn as nn


def get_weighted_loss(dataset):
    labels = [dataset[i][1].item() for i in range(len(dataset))]

    n_real = labels.count(0)
    n_spoof = labels.count(1)

    weight_real = 1.0 / n_real
    weight_spoof = 1.0 / n_spoof

    weights = torch.tensor([weight_real, weight_spoof], dtype=torch.float)
    return nn.CrossEntropyLoss(weight=weights)
