from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running this script directly (adds repo root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.split_data import load_splits
from models.random_forest import config
from models.random_forest.rf_model import TrainedRandomForestBaseline
from models.random_forest.utils import evaluate_targets, save_metrics, save_metrics_json


def main() -> None:
    model_path = Path(config.MODEL_BUNDLE_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    splits = load_splits(splits_dir=config.SPLITS_DIR, prefix="rf")
    model = TrainedRandomForestBaseline.load(model_path)

    id_col = config.SCENARIO_ID_COL
    target_cols = list(config.TARGET_COLUMNS)

    def eval_split(name: str, df_split: pd.DataFrame) -> pd.DataFrame:
        x = df_split[model.feature_columns]
        y_true = df_split[target_cols]
        y_pred = model.predict(x)
        metrics = evaluate_targets(y_true=y_true, y_pred=y_pred, target_columns=target_cols)
        save_metrics(metrics, config.METRICS_DIR / f"{name}_metrics.csv")
        save_metrics_json(metrics, config.METRICS_DIR / f"{name}_metrics.json")
        return metrics

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    val_metrics = eval_split("val", splits.val)
    test_metrics = eval_split("test", splits.test)

    print("Evaluation complete")
    print("Val metrics:\n", val_metrics)
    print("Test metrics:\n", test_metrics)


if __name__ == "__main__":
    main()
