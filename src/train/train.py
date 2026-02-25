import torch
from torch.utils.data import DataLoader

from src.data.feature_dataset import ASVspoof2019FeatureDataset
from src.models.cnn_model import SpoofCNN
from src.utils.loss import get_weighted_loss


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = ASVspoof2019FeatureDataset(split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = SpoofCNN().to(device)
    criterion = get_weighted_loss(train_dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(20):
        total_loss = 0

        for x, y in train_loader:
            x = x.to(device)      # [B, 1, 80, 300]
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/20 | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "spoof_cnn.pth")


if __name__ == "__main__":
    train()
