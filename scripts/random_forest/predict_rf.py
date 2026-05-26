from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

# Allow running this script directly (adds repo root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.random_forest import config
from models.random_forest.rf_model import TrainedRandomForestBaseline
from data_preprocessing.normalization import inverse_transform_targets
from data_preprocessing.split_data import apply_split_ids, load_split_ids


def main() -> None:
    input_csv = Path(config.PREDICT_INPUT_CSV_PATH)
    model_path = Path(config.MODEL_BUNDLE_PATH)
    output_csv = Path(config.PREDICT_OUTPUT_CSV_PATH)

    df = pd.read_csv(input_csv, low_memory=False)
    model = TrainedRandomForestBaseline.load(model_path)

    try:
        ids_split = load_split_ids(splits_dir=config.SPLIT_IDS_DIR, id_col=config.SCENARIO_ID_COL, prefix="rf_ids")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Shared split IDs not found under {config.SPLIT_IDS_DIR}. Run train_rf_raw.py or train_rf_engineered.py first.\n{e}"
        )

    splits = apply_split_ids(df, id_col=config.SCENARIO_ID_COL, split_ids_res=ids_split)

    # Tag matches the model directory name so runs stay comparable
    tag = model_path.parent.name
    predictions_dir = config.OUTPUTS_RUN_DIR / "predictions" / "random_forest" / tag
    predictions_dir.mkdir(parents=True, exist_ok=True)

    target_cols = list(model.target_columns)

    scalers_path = model_path.parent / "scalers.joblib"
    if scalers_path.exists():
        scalers_obj = joblib.load(scalers_path)
        target_scaler = scalers_obj.get("target_scaler")
    else:
        target_scaler = None

    def _predict_split(df_split: pd.DataFrame, name: str) -> None:
        preds = model.predict(df_split[model.feature_columns])
        y_true = df_split[target_cols]

        preds_o = inverse_transform_targets(preds, target_columns=target_cols, target_scaler=target_scaler)
        y_true_o = inverse_transform_targets(y_true, target_columns=target_cols, target_scaler=target_scaler)

        # Combined true+pred (source of truth for evaluation/plotting)
        out = pd.concat(
            [
                df_split[[config.SCENARIO_ID_COL]].reset_index(drop=True),
                y_true_o.add_prefix("true_"),
                preds_o.add_prefix("pred_"),
            ],
            axis=1,
        )
        out.to_csv(predictions_dir / f"{name}_predictions.csv", index=False)

        # Predictions only (convenience)
        out_pred_only = pd.concat(
            [df_split[[config.SCENARIO_ID_COL]].reset_index(drop=True), preds_o.reset_index(drop=True)], axis=1
        )
        out_pred_only.to_csv(predictions_dir / f"{name}_predictions_only.csv", index=False)

    _predict_split(splits.val, "val")
    _predict_split(splits.test, "test")
    # Legacy output: full-dataset predictions only (kept for backwards compatibility)
    preds_all = model.predict(df[model.feature_columns])
    preds_all_o = inverse_transform_targets(preds_all, target_columns=target_cols, target_scaler=target_scaler)
    out_legacy = pd.concat([df[[config.SCENARIO_ID_COL]].reset_index(drop=True), preds_all_o], axis=1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_legacy.to_csv(output_csv, index=False)

    print("Wrote:")
    print(" -", (predictions_dir / "val_predictions.csv").as_posix())
    print(" -", (predictions_dir / "test_predictions.csv").as_posix())
    print(" -", output_csv.as_posix(), "(legacy)")


if __name__ == "__main__":
    main()
