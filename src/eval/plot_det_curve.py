import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Load saved scores
scores = np.load("dev_scores.npy")   # bonafide probabilities
labels = np.load("dev_labels.npy")   # 0=spoof, 1=bonafide

# Compute ROC
fpr, tpr, thresholds = roc_curve(labels, scores)
fnr = 1 - tpr

# Compute EER
eer_idx = np.nanargmin(np.abs(fnr - fpr))
eer = fpr[eer_idx]
eer_threshold = thresholds[eer_idx]

print(f"EER: {eer:.4f}")
print(f"EER Threshold: {eer_threshold:.4f}")

# Plot DET curve
plt.figure(figsize=(6, 6))
plt.plot(fpr * 100, fnr * 100, label="DET Curve")
plt.scatter(
    fpr[eer_idx] * 100,
    fnr[eer_idx] * 100,
    color="red",
    label=f"EER = {eer*100:.2f}%"
)

plt.xlabel("False Positive Rate (%)")
plt.ylabel("False Negative Rate (%)")
plt.title("DET Curve (ASVspoof2019 Dev)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("det_curve_resnet18.png", dpi=300)
plt.show()

print("Saved dev_scores.npy and dev_labels.npy")
