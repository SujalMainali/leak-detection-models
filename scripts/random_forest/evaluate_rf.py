from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running this script directly (adds repo root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.random_forest import config
from models.random_forest.utils import evaluate_targets, save_metrics, save_metrics_json


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
    model_path = Path(config.MODEL_BUNDLE_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Save metrics into the per-tag outputs folder that matches the model.
    tag = model_path.parent.name
    metrics_dir = config.OUTPUTS_RUN_DIR / "metrics" / "random_forest" / tag

    predictions_dir = config.OUTPUTS_RUN_DIR / "predictions" / "random_forest" / tag

    def eval_split(name: str) -> pd.DataFrame:
        pred_path = predictions_dir / f"{name}_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(
                f"Missing predictions file: {pred_path}. Run scripts/random_forest/predict_rf.py first."
            )

        y_true, y_pred, target_cols = _load_predictions(pred_path)
        metrics = evaluate_targets(y_true=y_true, y_pred=y_pred, target_columns=target_cols)
        save_metrics(metrics, metrics_dir / f"{name}_metrics.csv")
        save_metrics_json(metrics, metrics_dir / f"{name}_metrics.json")
        return metrics

    metrics_dir.mkdir(parents=True, exist_ok=True)

    val_metrics = eval_split("val")
    test_metrics = eval_split("test")

    print("Evaluation complete")
    print("Loaded predictions from:", predictions_dir)
    print("Val metrics:\n", val_metrics)
    print("Test metrics:\n", test_metrics)


if __name__ == "__main__":
    main()
