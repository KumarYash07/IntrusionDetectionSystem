import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             roc_auc_score, average_precision_score, roc_curve)


def closed_metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y, pred), precision=p, recall=r, macro_f1=f)


def detection_metrics(is_unk, score, fpr_target):
    """Threshold-free known-vs-unknown separation. Unknown is the positive class."""
    fpr, tpr, _ = roc_curve(is_unk, score)
    return dict(auroc=roc_auc_score(is_unk, score), aupr=average_precision_score(is_unk, score),
                tpr_at_fpr=float(np.interp(fpr_target, fpr, tpr)))


def decision_metrics(y_true, pred_known, is_unk, flagged, unk_label):
    """Open-set decision metrics. Flagged samples get the 'unknown' label."""
    pred = np.where(flagged, unk_label, pred_known)
    true = np.where(is_unk, unk_label, y_true)
    return dict(known_fpr=float(flagged[~is_unk].mean()), unknown_detection_rate=float(flagged[is_unk].mean()),
                unknown_precision=float((flagged & is_unk).sum() / max(flagged.sum(), 1)),
                open_set_f1=f1_score(true, pred, average="macro", zero_division=0))
