"""Test-set metrics for one fitted model."""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import CLASSES


def evaluate(y_true, y_pred):
    """Accuracy, macro-F1, per-class precision/recall/F1 and the confusion matrix."""
    report = classification_report(
        y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": {c: report[c] for c in CLASSES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }
