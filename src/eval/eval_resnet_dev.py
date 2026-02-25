import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.models.pretrained_resnet import PretrainedResNet18
from src.data.asvspoof2019_feature_dataset import ASVspoof2019FeatureDataset
from src.metrics.eer import compute_eer

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cpu")
BATCH_SIZE = 16
MAX_TIME = 300
MODEL_PATH = "resnet18_asvspoof.pth"


# =========================
# INPUT PREPARATION
# =========================
def prepare_input(x):
    """
    Final ResNet input shape: [B, 1, 80, 300]
    """

    # Remove accidental extra dims
    # Handles [B,1,1,80,T], [B,1,80,T], [B,80,T]
    while x.dim() > 3:
        x = x.squeeze(1)

    if x.dim() != 3:
        raise RuntimeError(f"Invalid input shape: {x.shape}")

    # Time crop / pad
    x = x[:, :, :MAX_TIME]
    if x.shape[-1] < MAX_TIME:
        pad = MAX_TIME - x.shape[-1]
        x = F.pad(x, (0, pad))

    # Add channel dimension (ONLY 1 CHANNEL)
    x = x.unsqueeze(1)  # [B, 1, 80, 300]

    return x


# =========================
# EVALUATION
# =========================
def evaluate():
    print("Using device:", DEVICE)

    dataset = ASVspoof2019FeatureDataset(split="dev")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = PretrainedResNet18(num_classes=2)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for x, y in loader:
            x = prepare_input(x).to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            bonafide_scores = probs[:, 1]
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(bonafide_scores.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    eer, threshold = compute_eer(
        np.array(all_scores),
        np.array(all_labels)
    )

    print("\n===== DEV RESULTS =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"EER      : {eer:.4f}")
    print(f"Threshold: {threshold:.4f}")

    np.save("dev_scores.npy", np.array(all_scores))
    np.save("dev_labels.npy", np.array(all_labels))
    print("Saved dev_scores.npy and dev_labels.npy")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    evaluate()
