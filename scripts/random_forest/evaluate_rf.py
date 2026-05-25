from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

# Allow running this script directly (adds repo root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.split_data import apply_split_ids, load_split_ids
from models.random_forest import config
from models.random_forest.rf_model import TrainedRandomForestBaseline
from models.random_forest.utils import evaluate_targets, save_metrics, save_metrics_json
from data_preprocessing.normalization import inverse_transform_targets


def main() -> None:
    model_path = Path(config.MODEL_BUNDLE_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    processed_path = Path(config.PREDICT_INPUT_CSV_PATH)
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")

    df = pd.read_csv(processed_path, low_memory=False)

    # Apply shared split IDs so results match the train/val/test split used in training.
    try:
        ids_split = load_split_ids(splits_dir=config.SPLIT_IDS_DIR, id_col=config.SCENARIO_ID_COL, prefix="rf_ids")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Shared split IDs not found under {config.SPLIT_IDS_DIR}. Run train_rf_raw.py or train_rf_engineered.py first.\n{e}"
        )

    splits = apply_split_ids(df, id_col=config.SCENARIO_ID_COL, split_ids_res=ids_split)
    model = TrainedRandomForestBaseline.load(model_path)

    scalers_path = model_path.parent / "scalers.joblib"
    scalers_obj = joblib.load(scalers_path) if scalers_path.exists() else {}
    target_scaler = scalers_obj.get("target_scaler")

    target_cols = list(config.TARGET_COLUMNS)

    # Save metrics into the per-tag outputs folder that matches the model.
    tag = model_path.parent.name
    metrics_dir = config.OUTPUTS_RUN_DIR / "metrics" / "random_forest" / tag

    def eval_split(name: str, df_split: pd.DataFrame) -> pd.DataFrame:
        x = df_split[model.feature_columns]
        y_true = df_split[target_cols]
        y_pred = model.predict(x)

        y_true_orig = inverse_transform_targets(y_true, target_columns=target_cols, target_scaler=target_scaler)
        y_pred_orig = inverse_transform_targets(y_pred, target_columns=target_cols, target_scaler=target_scaler)

        metrics = evaluate_targets(y_true=y_true_orig, y_pred=y_pred_orig, target_columns=target_cols)
        save_metrics(metrics, metrics_dir / f"{name}_metrics.csv")
        save_metrics_json(metrics, metrics_dir / f"{name}_metrics.json")
        return metrics

    metrics_dir.mkdir(parents=True, exist_ok=True)

    val_metrics = eval_split("val", splits.val)
    test_metrics = eval_split("test", splits.test)

    print("Evaluation complete")
    print("Val metrics:\n", val_metrics)
    print("Test metrics:\n", test_metrics)


if __name__ == "__main__":
    main()
