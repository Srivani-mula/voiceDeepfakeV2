import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from tqdm import tqdm

from src.models.resnet_model import ResNet18
from src.data.asvspoof2019_dataset import ASVspoof2019Dataset


# =====================================================
# EER FUNCTION
# =====================================================
def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_index]
    return eer * 100


# =====================================================
# TRAINING FUNCTION
# =====================================================
def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    batch_size = 16
    num_epochs = 5
    learning_rate = 1e-4

    root_path = "data/raw/asvspoof2019/LA"

    train_dataset = ASVspoof2019Dataset(root_path, split="train")
    dev_dataset = ASVspoof2019Dataset(root_path, split="dev")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = ResNet18(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_eer = float("inf")

    for epoch in range(num_epochs):

        print(f"\n==============================")
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"==============================")

        # ================= TRAIN =================
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc="Training", leave=False)

        for inputs, labels in train_bar:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        avg_train_loss = running_loss / len(train_loader)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_scores = []

        dev_bar = tqdm(dev_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for inputs, labels in dev_bar:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                spoof_scores = probs[:, 1]

                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(spoof_scores.cpu().numpy())

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                dev_bar.set_postfix(
                    val_loss=f"{loss.item():.4f}"
                )

        avg_val_loss = val_loss / len(dev_loader)
        accuracy = 100 * correct / total
        eer = compute_eer(
            np.array(all_labels),
            np.array(all_scores)
        )

        print(f"\nTrain Loss      : {avg_train_loss:.4f}")
        print(f"Validation Loss : {avg_val_loss:.4f}")
        print(f"Validation Acc  : {accuracy:.2f}%")
        print(f"Validation EER  : {eer:.2f}%")

        # Save best model
        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), "best_model.pth")
            print("🔥 Best model saved (lowest EER)")

    print("\nTraining Complete.")
    print(f"Best EER achieved: {best_eer:.2f}%")


if __name__ == "__main__":
    train()