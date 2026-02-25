import torch
from src.models.cnn_baseline import CNNBaseline

model = CNNBaseline()
x = torch.randn(4, 1, 80, 401)  # batch of Log-Mel features
y = model(x)

print("Output shape:", y.shape)
