import torch
from src.models.cnn_model import CNNModel

model = CNNModel()

# 🔴 Load your trained weights here if you have them
# Example:
# model.load_state_dict(torch.load("resnet_dev.pth"))

torch.save(model.state_dict(), "models/best_model.pth")
print("Model saved successfully")
