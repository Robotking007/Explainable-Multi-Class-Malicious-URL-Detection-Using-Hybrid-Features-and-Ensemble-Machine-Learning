"""
Task 4: Model Evaluation & MLOps Quality Gate.
Ensures model metrics meet production threshold criteria before deployment.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "models" / "url_dl_metrics.json"

MIN_ACCURACY_THRESHOLD = 0.80
MIN_MACRO_F1_THRESHOLD = 0.75


def evaluate_model(
    metrics_path: str | Path | None = None,
    min_accuracy: float = MIN_ACCURACY_THRESHOLD,
    min_f1: float = MIN_MACRO_F1_THRESHOLD,
) -> dict:
    target_path = Path(metrics_path) if metrics_path else METRICS_PATH
    print(f"[Airflow Task: evaluate_model] Reading metrics from {target_path}...")

    if not target_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {target_path}")

    with open(target_path) as f:
        rep = json.load(f)

    accuracy = rep.get("accuracy", 0.0)
    macro_f1 = rep.get("macro avg", {}).get("f1-score", 0.0)
    macro_precision = rep.get("macro avg", {}).get("precision", 0.0)
    macro_recall = rep.get("macro avg", {}).get("recall", 0.0)

    print(f"Model Evaluation Summary:")
    print(f"  Accuracy:        {accuracy * 100:.2f}% (Threshold: {min_accuracy * 100:.2f}%)")
    print(f"  Macro-Precision: {macro_precision:.4f}")
    print(f"  Macro-Recall:    {macro_recall:.4f}")
    print(f"  Macro-F1:        {macro_f1:.4f} (Threshold: {min_f1:.4f})")

    # Quality Gate Check
    if accuracy < min_accuracy or macro_f1 < min_f1:
        error_msg = (
            f"Quality Gate FAILED: Model performance does not meet production thresholds. "
            f"Accuracy: {accuracy:.4f} (required >= {min_accuracy}), "
            f"Macro-F1: {macro_f1:.4f} (required >= {min_f1}). Pipeline halted."
        )
        print(f"[Airflow Task: evaluate_model] ERROR: {error_msg}")
        raise ValueError(error_msg)

    summary = {
        "status": "PASSED_QUALITY_GATE",
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "verdict": "APPROVED_FOR_DEPLOYMENT",
    }
    print(f"[Airflow Task: evaluate_model] SUCCESS: {summary}")
    return summary


if __name__ == "__main__":
    evaluate_model()
