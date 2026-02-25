import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(scores, labels):
    """
    scores : numpy array (higher = more bonafide)
    labels : numpy array (0 = spoof, 1 = bonafide)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    return eer, eer_threshold
