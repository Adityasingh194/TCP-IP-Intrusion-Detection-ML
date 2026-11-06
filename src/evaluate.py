"""Metrics and threshold selection."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob):
    """Positive class = anomaly, so recall = share of attacks caught."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def youden_threshold(y_true, y_prob):
    """Threshold that maximises Youden's J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    finite = np.isfinite(thresholds)
    best = np.argmax((tpr - fpr)[finite])
    return float(thresholds[finite][best])
