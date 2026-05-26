from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this script directly (adds repo root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn import config
from models.cnn.utils import evaluate_targets, save_metrics, save_metrics_json


def _load_predictions(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(path, low_memory=False)
    target_columns = list(config.TARGET_COLUMNS)
    true_cols = [f"true_{t}" for t in target_columns]
    pred_cols = [f"pred_{t}" for t in target_columns]
    missing = [c for c in [config.SCENARIO_ID_COL, *true_cols, *pred_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")

    y_true = df.loc[:, true_cols].copy()
    y_true.columns = target_columns
    y_pred = df.loc[:, pred_cols].copy()
    y_pred.columns = target_columns
    return y_true, y_pred, target_columns


def main() -> None:
    for p in [config.METRICS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    val_pred_path = config.PREDICTIONS_DIR / "val_predictions.csv"
    test_pred_path = config.PREDICTIONS_DIR / "test_predictions.csv"
    if not test_pred_path.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {test_pred_path}. Run scripts/cnn/predict_cnn.py first."
        )

    if val_pred_path.exists():
        y_true_v, y_pred_v, target_columns = _load_predictions(val_pred_path)
        val_metrics = evaluate_targets(y_true=y_true_v, y_pred=y_pred_v, target_columns=target_columns)
        save_metrics(val_metrics, config.METRICS_DIR / "val_metrics.csv")
        save_metrics_json(val_metrics, config.METRICS_DIR / "val_metrics.json")

    y_true, y_pred, target_columns = _load_predictions(test_pred_path)
    test_metrics = evaluate_targets(y_true=y_true, y_pred=y_pred, target_columns=target_columns)

    save_metrics(test_metrics, config.METRICS_DIR / "test_metrics.csv")
    save_metrics_json(test_metrics, config.METRICS_DIR / "test_metrics.json")

    print("Evaluated CNN model")
    print("Saved metrics:", config.METRICS_DIR)
    print("Loaded predictions:", test_pred_path)


if __name__ == "__main__":
    main()
