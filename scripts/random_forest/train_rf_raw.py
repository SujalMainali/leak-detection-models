from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# Allow running this script directly (adds repo root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.rf_preprocessing import run_rf_preprocessing
from data_preprocessing.normalization import (
    inverse_transform_targets,
    load_normalization_artifacts,
    needs_normalization_refresh,
    normalize_and_persist_processed_dataset,
)
from data_preprocessing.split_data import (
    apply_split_ids,
    load_split_ids,
    save_split_ids,
    split_ids,
)
from models.random_forest import config
from models.random_forest.rf_model import train_three_random_forests
from models.random_forest.utils import evaluate_targets, save_metrics, save_metrics_json, save_predictions


def _to_jsonable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, range):
        return list(x)
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def _config_module_snapshot(cfg_module) -> dict:
    snap: dict[str, object] = {}
    for k, v in vars(cfg_module).items():
        if isinstance(k, str) and k.isupper():
            snap[k] = _to_jsonable(v)
    return snap


def _output_dirs(tag: str) -> dict[str, Path]:
    outputs_dir = config.OUTPUTS_RUN_DIR
    return {
        "models": outputs_dir / "models" / "random_forest" / tag,
        "metrics": outputs_dir / "metrics" / "random_forest" / tag,
        "predictions": outputs_dir / "predictions" / "random_forest" / tag,
        "plots": outputs_dir / "plots" / "random_forest" / tag,
    }


def _load_or_build_raw_processed() -> pd.DataFrame:
    if config.RF_RAW_FEATURES_CSV_PATH.exists() and not config.REBUILD_PROCESSED_ON_RUN:
        df_existing = pd.read_csv(config.RF_RAW_FEATURES_CSV_PATH, low_memory=False)
        id_col = config.SCENARIO_ID_COL
        target_cols = list(config.TARGET_COLUMNS)
        feature_cols = [c for c in df_existing.columns if c not in [id_col, *target_cols]]
        if not needs_normalization_refresh(
            config.RF_RAW_FEATURES_CSV_PATH,
            normalize_features=config.NORMALIZE_FEATURES,
            normalize_targets=config.NORMALIZE_TARGETS,
            feature_columns=feature_cols,
            target_columns=target_cols,
        ):
            return df_existing
        print("Normalization sidecars missing/mismatched for raw processed CSV; rebuilding from raw.")

    res = run_rf_preprocessing(
        raw_csv_path=config.RAW_CSV_PATH,
        processed_csv_path=config.RF_RAW_FEATURES_CSV_PATH,
        scenario_id_col=config.SCENARIO_ID_COL,
        target_columns=config.TARGET_COLUMNS,
        sensor_node_prefixes=config.SENSOR_NODE_PREFIXES,
        valid_hours=config.VALID_HOURS,
    )
    print(
        "Raw preprocessing summary: "
        f"dropped_missing_targets={res.dropped_rows_missing_targets}, "
        f"dropped_missing_features={res.dropped_rows_missing_features}, "
        f"dropped_total={res.dropped_rows_total}"
    )
    return res.df


def main() -> None:
    tag = "raw_baseline"
    out_dirs = _output_dirs(tag)
    for p in out_dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    df = _load_or_build_raw_processed()

    id_col = config.SCENARIO_ID_COL
    target_cols = list(config.TARGET_COLUMNS)
    feature_cols = [c for c in df.columns if c not in [id_col, *target_cols]]

    # Shared split IDs (for fair comparison against engineered feature sets)
    if config.REBUILD_SHARED_SPLIT_IDS_ON_RUN:
        if config.SPLIT_IDS_DIR.exists():
            for p in config.SPLIT_IDS_DIR.glob("rf_ids_*.csv"):
                p.unlink()

    if (config.SPLIT_IDS_DIR / "rf_ids_train.csv").exists():
        ids_split = load_split_ids(splits_dir=config.SPLIT_IDS_DIR, id_col=id_col, prefix="rf_ids")
    else:
        ids_split = split_ids(
            df,
            id_col=id_col,
            train_fraction=config.TRAIN_FRACTION,
            val_fraction=config.VAL_FRACTION,
            test_fraction=config.TEST_FRACTION,
            random_state=config.RANDOM_STATE,
        )
        save_split_ids(ids_split, splits_dir=config.SPLIT_IDS_DIR, id_col=id_col, prefix="rf_ids")

    splits_pre = apply_split_ids(df, id_col=id_col, split_ids_res=ids_split)

    # Normalize and persist processed CSV (fit scalers on train split only)
    if needs_normalization_refresh(
        config.RF_RAW_FEATURES_CSV_PATH,
        normalize_features=config.NORMALIZE_FEATURES,
        normalize_targets=config.NORMALIZE_TARGETS,
        feature_columns=feature_cols,
        target_columns=target_cols,
    ):
        df, artifacts = normalize_and_persist_processed_dataset(
            df,
            processed_csv_path=config.RF_RAW_FEATURES_CSV_PATH,
            train_df=splits_pre.train,
            feature_columns=feature_cols,
            target_columns=target_cols,
            normalize_features=config.NORMALIZE_FEATURES,
            normalize_targets=config.NORMALIZE_TARGETS,
            method=config.NORMALIZATION_METHOD,
        )
    else:
        artifacts = load_normalization_artifacts(config.RF_RAW_FEATURES_CSV_PATH)

    splits = apply_split_ids(df, id_col=id_col, split_ids_res=ids_split)

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

    cfg_snapshot = {
        "model": "random_forest",
        "tag": tag,
        "config": _config_module_snapshot(config),
        "rf_params": _to_jsonable(rf_params),
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "splits": {
            "num_train": int(len(splits.train)),
            "num_val": int(len(splits.val)),
            "num_test": int(len(splits.test)),
        },
    }
    (out_dirs["models"] / "config_snapshot.json").write_text(
        json.dumps(cfg_snapshot, indent=2), encoding="utf-8"
    )

    model = train_three_random_forests(
        x_train=x_train,
        y_train=y_train,
        feature_columns=feature_cols,
        target_columns=target_cols,
        rf_params=rf_params,
    )

    model_path = model.save(out_dirs["models"])

    joblib.dump(
        {"feature_scaler": artifacts.feature_scaler, "target_scaler": artifacts.target_scaler},
        out_dirs["models"] / "scalers.joblib",
    )

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    y_val_orig = inverse_transform_targets(y_val, target_columns=target_cols, target_scaler=artifacts.target_scaler)
    y_test_orig = inverse_transform_targets(y_test, target_columns=target_cols, target_scaler=artifacts.target_scaler)
    val_pred_orig = inverse_transform_targets(
        val_pred, target_columns=target_cols, target_scaler=artifacts.target_scaler
    )
    test_pred_orig = inverse_transform_targets(
        test_pred, target_columns=target_cols, target_scaler=artifacts.target_scaler
    )

    val_metrics = evaluate_targets(y_true=y_val_orig, y_pred=val_pred_orig, target_columns=target_cols)
    test_metrics = evaluate_targets(y_true=y_test_orig, y_pred=test_pred_orig, target_columns=target_cols)

    save_metrics(val_metrics, out_dirs["metrics"] / "val_metrics.csv")
    save_metrics_json(val_metrics, out_dirs["metrics"] / "val_metrics.json")
    save_metrics(test_metrics, out_dirs["metrics"] / "test_metrics.csv")
    save_metrics_json(test_metrics, out_dirs["metrics"] / "test_metrics.json")

    save_predictions(
        df_ids=val_ids,
        y_true=y_val_orig,
        y_pred=val_pred_orig,
        out_path=out_dirs["predictions"] / "val_predictions.csv",
        target_columns=target_cols,
    )
    save_predictions(
        df_ids=test_ids,
        y_true=y_test_orig,
        y_pred=test_pred_orig,
        out_path=out_dirs["predictions"] / "test_predictions.csv",
        target_columns=target_cols,
    )

    print("Trained Random Forest (raw baseline)")
    print("Processed dataset:", config.RF_RAW_FEATURES_CSV_PATH)
    print("Saved model:", model_path)
    print("Saved metrics:", out_dirs["metrics"])
    print("Saved predictions:", out_dirs["predictions"])


if __name__ == "__main__":
    main()
