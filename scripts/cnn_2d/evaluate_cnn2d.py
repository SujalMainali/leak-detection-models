from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow running this script directly (adds repo root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.split_data import apply_split_ids, load_split_ids
from models.cnn_2d import config
from models.cnn_2d.cnn2d_model import Cnn2dModelConfig, build_cnn2d_model
from models.cnn_2d.utils import evaluate_targets, save_metrics, save_metrics_json, save_predictions


def _require_torch():
    try:
        import torch

        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required to evaluate the CNN-2D model. Install it in LeakEnv: `pip install torch`."
        ) from e


def _load_processed() -> dict:
    data = np.load(config.CNN2D_READY_NPZ_PATH, allow_pickle=True)
    scalers = joblib.load(config.CNN2D_SCALERS_PATH) if config.CNN2D_SCALERS_PATH.exists() else {}
    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.float32),
        "scenario_ids": data["scenario_ids"].astype(int),
        "target_columns": list(data["target_columns"].tolist()),
        "target_scaler": scalers.get("target_scaler"),
    }


def _inverse_targets(arr: np.ndarray, target_scaler) -> np.ndarray:
    if target_scaler is None:
        return arr
    return target_scaler.inverse_transform(arr)


def main() -> None:
    torch = _require_torch()

    for p in [config.METRICS_DIR, config.PREDICTIONS_DIR, config.PLOTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    processed = _load_processed()
    X = processed["X"]
    y = processed["y"]
    scenario_ids = processed["scenario_ids"]
    target_columns = list(processed["target_columns"])
    target_scaler = processed["target_scaler"]

    ids_split = load_split_ids(splits_dir=config.SPLIT_IDS_DIR, id_col=config.SCENARIO_ID_COL, prefix="cnn2d_ids")
    df_ids = pd.DataFrame({config.SCENARIO_ID_COL: scenario_ids.astype(int)})
    splits = apply_split_ids(df_ids, id_col=config.SCENARIO_ID_COL, split_ids_res=ids_split)

    val_set = set(map(int, splits.val[config.SCENARIO_ID_COL].tolist()))
    test_set = set(map(int, splits.test[config.SCENARIO_ID_COL].tolist()))

    val_mask = np.array([int(sid) in val_set for sid in scenario_ids], dtype=bool)
    test_mask = np.array([int(sid) in test_set for sid in scenario_ids], dtype=bool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kernel_sizes = [(int(k[0]), int(k[1])) for k in config.KERNEL_SIZES]
    pool_sizes = [(int(p[0]), int(p[1])) for p in config.POOL_SIZES]

    model_cfg = Cnn2dModelConfig(
        in_channels=1,
        conv_filters=list(map(int, config.CONV_FILTERS)),
        kernel_sizes=kernel_sizes,
        pool_sizes=pool_sizes,
        dropout_rates=list(map(float, config.DROPOUT_RATES)),
        dense_units=list(map(int, config.DENSE_UNITS)),
        use_global_avg_pool=bool(config.USE_GLOBAL_AVG_POOL),
        use_batch_norm=bool(config.USE_BATCH_NORM),
        num_outputs=len(target_columns),
    )

    model = build_cnn2d_model(model_cfg).to(device)
    state = torch.load(config.MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    def predict(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xb = torch.from_numpy(x_np).to(device)
            pred = model(xb).detach().cpu().numpy()
        return pred

    y_val_pred = predict(X[val_mask])
    y_test_pred = predict(X[test_mask])

    y_val_true = y[val_mask]
    y_test_true = y[test_mask]

    y_val_true_o = _inverse_targets(y_val_true, target_scaler)
    y_val_pred_o = _inverse_targets(y_val_pred, target_scaler)
    y_test_true_o = _inverse_targets(y_test_true, target_scaler)
    y_test_pred_o = _inverse_targets(y_test_pred, target_scaler)

    df_val_true = pd.DataFrame(y_val_true_o, columns=target_columns)
    df_val_pred = pd.DataFrame(y_val_pred_o, columns=target_columns)
    df_test_true = pd.DataFrame(y_test_true_o, columns=target_columns)
    df_test_pred = pd.DataFrame(y_test_pred_o, columns=target_columns)

    val_metrics = evaluate_targets(y_true=df_val_true, y_pred=df_val_pred, target_columns=target_columns)
    test_metrics = evaluate_targets(y_true=df_test_true, y_pred=df_test_pred, target_columns=target_columns)

    save_metrics(val_metrics, config.METRICS_DIR / "val_metrics.csv")
    save_metrics_json(val_metrics, config.METRICS_DIR / "val_metrics.json")
    save_metrics(test_metrics, config.METRICS_DIR / "test_metrics.csv")
    save_metrics_json(test_metrics, config.METRICS_DIR / "test_metrics.json")

    save_predictions(
        scenario_ids=scenario_ids[val_mask],
        y_true=y_val_true_o,
        y_pred=y_val_pred_o,
        out_path=config.PREDICTIONS_DIR / "val_predictions.csv",
        target_columns=target_columns,
    )
    save_predictions(
        scenario_ids=scenario_ids[test_mask],
        y_true=y_test_true_o,
        y_pred=y_test_pred_o,
        out_path=config.PREDICTIONS_DIR / "test_predictions.csv",
        target_columns=target_columns,
    )

    # Plots (optional; requires matplotlib)
    try:
        import matplotlib.pyplot as plt  # type: ignore

        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(target_columns):
            yt = y_test_true_o[:, i]
            yp = y_test_pred_o[:, i]
            lo = float(min(np.min(yt), np.min(yp)))
            hi = float(max(np.max(yt), np.max(yp)))

            plt.figure(figsize=(6, 6))
            plt.scatter(yt, yp, s=8, alpha=0.6)
            plt.plot([lo, hi], [lo, hi], linestyle="--")
            plt.xlabel(f"True {t}")
            plt.ylabel(f"Pred {t}")
            plt.title(f"CNN-2D: Pred vs True ({t})")
            out_path = config.PLOTS_DIR / f"pred_vs_true__{t}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
    except Exception:
        pass

    print("Evaluated CNN-2D model")
    print("Model:", config.MODEL_PATH)
    print("Saved metrics:", config.METRICS_DIR)
    print("Saved predictions:", config.PREDICTIONS_DIR)
    print("Saved plots:", config.PLOTS_DIR)


if __name__ == "__main__":
    main()
