from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running this script directly (adds repo root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.rf_preprocessing import run_rf_preprocessing
from data_preprocessing.split_data import load_splits, save_splits, split_wide_dataframe
from models.random_forest import config
from models.random_forest.rf_model import train_three_random_forests
from models.random_forest.utils import evaluate_targets, save_metrics, save_metrics_json, save_predictions


def _load_or_build_processed() -> pd.DataFrame:
    if config.PROCESSED_CSV_PATH.exists():
        df_existing = pd.read_csv(config.PROCESSED_CSV_PATH, low_memory=False)

        id_col = config.SCENARIO_ID_COL
        target_cols = list(config.TARGET_COLUMNS)
        feature_cols = [c for c in df_existing.columns if c not in [id_col, *target_cols]]

        missing_in_targets = int(df_existing[target_cols].isna().any(axis=1).sum()) if target_cols else 0
        missing_in_features = int(df_existing[feature_cols].isna().any(axis=1).sum()) if feature_cols else 0

        if missing_in_targets or missing_in_features:
            print(
                "Processed CSV contains missing values; rebuilding from raw. "
                f"missing_targets_rows={missing_in_targets}, missing_feature_rows={missing_in_features}"
            )
            config.PROCESSED_CSV_PATH.unlink()
        else:
            return df_existing

    res = run_rf_preprocessing(
        raw_csv_path=config.RAW_CSV_PATH,
        processed_csv_path=config.PROCESSED_CSV_PATH,
        scenario_id_col=config.SCENARIO_ID_COL,
        target_columns=config.TARGET_COLUMNS,
        sensor_node_prefixes=config.SENSOR_NODE_PREFIXES,
        valid_hours=config.VALID_HOURS,
    )
    print(
        "Preprocessing summary: "
        f"dropped_missing_targets={res.dropped_rows_missing_targets}, "
        f"dropped_missing_features={res.dropped_rows_missing_features}, "
        f"dropped_total={res.dropped_rows_total}"
    )
    return res.df


def main() -> None:
    if config.REBUILD_PROCESSED_ON_RUN and config.PROCESSED_CSV_PATH.exists():
        config.PROCESSED_CSV_PATH.unlink()

    df = _load_or_build_processed()

    # Identify columns
    id_col = config.SCENARIO_ID_COL
    target_cols = list(config.TARGET_COLUMNS)
    feature_cols = [c for c in df.columns if c not in [id_col, *target_cols]]

    # Ensure output dirs exist
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Splits
    if config.REBUILD_SPLITS_ON_RUN:
        if config.SPLITS_DIR.exists():
            for p in config.SPLITS_DIR.glob("rf_*.csv"):
                p.unlink()

    if (config.SPLITS_DIR / "rf_train.csv").exists():
        splits = load_splits(splits_dir=config.SPLITS_DIR, prefix="rf")
    else:
        splits = split_wide_dataframe(
            df,
            id_col=id_col,
            train_fraction=config.TRAIN_FRACTION,
            val_fraction=config.VAL_FRACTION,
            test_fraction=config.TEST_FRACTION,
            random_state=config.RANDOM_STATE,
        )
        save_splits(splits, splits_dir=config.SPLITS_DIR, prefix="rf")

    def split_xy(d: pd.DataFrame):
        x = d[feature_cols]
        y = d[target_cols]
        ids = d[[id_col]]
        return ids, x, y

    train_ids, x_train, y_train = split_xy(splits.train)
    val_ids, x_val, y_val = split_xy(splits.val)
    test_ids, x_test, y_test = split_xy(splits.test)

    rf_params = dict(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        min_samples_split=config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.MIN_SAMPLES_LEAF,
        max_features=config.MAX_FEATURES,
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE,
    )

    model = train_three_random_forests(
        x_train=x_train,
        y_train=y_train,
        feature_columns=feature_cols,
        target_columns=target_cols,
        rf_params=rf_params,
    )

    model_path = model.save(config.MODELS_DIR)

    # Evaluate
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    val_metrics = evaluate_targets(y_true=y_val, y_pred=val_pred, target_columns=target_cols)
    test_metrics = evaluate_targets(y_true=y_test, y_pred=test_pred, target_columns=target_cols)

    save_metrics(val_metrics, config.METRICS_DIR / "val_metrics.csv")
    save_metrics_json(val_metrics, config.METRICS_DIR / "val_metrics.json")
    save_metrics(test_metrics, config.METRICS_DIR / "test_metrics.csv")
    save_metrics_json(test_metrics, config.METRICS_DIR / "test_metrics.json")

    save_predictions(
        df_ids=val_ids,
        y_true=y_val,
        y_pred=val_pred,
        out_path=config.PREDICTIONS_DIR / "val_predictions.csv",
        target_columns=target_cols,
    )
    save_predictions(
        df_ids=test_ids,
        y_true=y_test,
        y_pred=test_pred,
        out_path=config.PREDICTIONS_DIR / "test_predictions.csv",
        target_columns=target_cols,
    )

    print("Trained Random Forest baseline")
    print("Processed dataset:", config.PROCESSED_CSV_PATH)
    print("Saved model:", model_path)
    print("Saved metrics:", config.METRICS_DIR)
    print("Saved predictions:", config.PREDICTIONS_DIR)


if __name__ == "__main__":
    main()
