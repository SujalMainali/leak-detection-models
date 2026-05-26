from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow running this script directly (adds repo root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.cnn2d_preprocessing import preprocess_cnn2d_long_csv
from data_preprocessing.split_data import apply_split_ids, load_split_ids, save_split_ids, split_ids
from models.cnn_2d import config
from models.cnn_2d.cnn2d_model import Cnn2dModelConfig, build_cnn2d_model


def _to_jsonable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, range):
        return list(x)
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def _config_module_snapshot(cfg_module) -> dict:
    snap: dict[str, object] = {}
    for k, v in vars(cfg_module).items():
        if isinstance(k, str) and k.isupper():
            snap[k] = _to_jsonable(v)
    return snap


def _require_torch():
    try:
        import torch

        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required to train the CNN-2D model. Install it in LeakEnv: `pip install torch`."
        ) from e


@dataclass(frozen=True)
class TrainHistoryRow:
    epoch: int
    train_loss: float
    val_loss: float


def _output_dirs() -> dict[str, Path]:
    return {
        "models": config.MODEL_DIR,
        "metrics": config.METRICS_DIR,
        "predictions": config.PREDICTIONS_DIR,
        "plots": config.PLOTS_DIR,
        "histories": config.HISTORIES_DIR,
    }


def _load_or_build_processed(train_ids: np.ndarray | None) -> dict:
    if (
        config.CNN2D_READY_NPZ_PATH.exists()
        and config.CNN2D_SCALERS_PATH.exists()
        and not config.REBUILD_PROCESSED_ON_RUN
    ):
        data = np.load(config.CNN2D_READY_NPZ_PATH, allow_pickle=True)
        scalers = joblib.load(config.CNN2D_SCALERS_PATH)
        return {
            "X": data["X"],
            "y": data["y"],
            "scenario_ids": data["scenario_ids"],
            "sensor_columns": list(data["sensor_columns"].tolist()),
            "target_columns": list(data["target_columns"].tolist()),
            "pressure_scaler": scalers.get("pressure_scaler"),
            "target_scaler": scalers.get("target_scaler"),
        }

    train_ids_seq = None if train_ids is None else [int(x) for x in train_ids.tolist()]

    res, artifacts, meta = preprocess_cnn2d_long_csv(
        csv_path=config.CNN_LONG_CSV_PATH,
        scenario_id_col=config.SCENARIO_ID_COL,
        hour_col=config.HOUR_COL,
        target_columns=config.TARGET_COLUMNS,
        sensor_prefixes=config.SENSOR_NODE_PREFIXES,
        num_hours=config.NUM_HOURS,
        normalize_pressures=config.NORMALIZE_PRESSURES,
        pressure_normalization_mode=config.PRESSURE_NORMALIZATION_MODE,
        normalize_targets=config.NORMALIZE_TARGETS,
        train_scenario_ids=train_ids_seq,
        out_npz_path=config.CNN2D_READY_NPZ_PATH,
        out_scalers_path=config.CNN2D_SCALERS_PATH,
        out_meta_path=config.CNN2D_META_PATH,
    )

    return {
        "X": res.x,
        "y": res.y,
        "scenario_ids": res.scenario_ids,
        "sensor_columns": res.sensor_columns,
        "target_columns": res.target_columns,
        "pressure_scaler": artifacts.pressure_scaler,
        "target_scaler": artifacts.target_scaler,
    }


def _make_splits(scenario_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_ids = pd.DataFrame({config.SCENARIO_ID_COL: scenario_ids.astype(int)})

    if config.REBUILD_SHARED_SPLIT_IDS_ON_RUN and config.SPLIT_IDS_DIR.exists():
        for p in config.SPLIT_IDS_DIR.glob("cnn2d_ids_*.csv"):
            p.unlink()

    if (config.SPLIT_IDS_DIR / "cnn2d_ids_train.csv").exists():
        ids_split = load_split_ids(splits_dir=config.SPLIT_IDS_DIR, id_col=config.SCENARIO_ID_COL, prefix="cnn2d_ids")
    else:
        ids_split = split_ids(
            df_ids,
            id_col=config.SCENARIO_ID_COL,
            train_fraction=config.TRAIN_FRACTION,
            val_fraction=config.VAL_FRACTION,
            test_fraction=config.TEST_FRACTION,
            random_state=config.RANDOM_STATE,
        )
        save_split_ids(ids_split, splits_dir=config.SPLIT_IDS_DIR, id_col=config.SCENARIO_ID_COL, prefix="cnn2d_ids")

    splits = apply_split_ids(df_ids, id_col=config.SCENARIO_ID_COL, split_ids_res=ids_split)
    return (
        splits.train[config.SCENARIO_ID_COL].to_numpy(dtype=int),
        splits.val[config.SCENARIO_ID_COL].to_numpy(dtype=int),
        splits.test[config.SCENARIO_ID_COL].to_numpy(dtype=int),
    )


def main() -> None:
    torch = _require_torch()
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    for p in _output_dirs().values():
        p.mkdir(parents=True, exist_ok=True)

    # Create split IDs early so preprocessing can fit scalers on train only.
    df_ids = pd.read_csv(config.CNN_LONG_CSV_PATH, usecols=[config.SCENARIO_ID_COL], low_memory=False)
    scenario_ids_unique = df_ids[config.SCENARIO_ID_COL].drop_duplicates().to_numpy(dtype=int)
    train_ids, val_ids, test_ids = _make_splits(scenario_ids_unique)

    processed = _load_or_build_processed(train_ids)
    X = processed["X"].astype(np.float32)  # (N, 24, 21, 1)
    y = processed["y"].astype(np.float32)
    scenario_ids = processed["scenario_ids"].astype(int)

    train_set = set(map(int, train_ids))
    val_set = set(map(int, val_ids))

    train_mask = np.array([int(sid) in train_set for sid in scenario_ids], dtype=bool)
    val_mask = np.array([int(sid) in val_set for sid in scenario_ids], dtype=bool)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(int(config.TORCH_SEED))
    np.random.seed(int(config.TORCH_SEED))

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
        num_outputs=len(config.TARGET_COLUMNS),
    )

    cfg_snapshot = {
        "model": "cnn_2d",
        "config": _config_module_snapshot(config),
        "model_cfg": asdict(model_cfg),
        "device": str(device),
        "splits": {
            "num_train_ids": int(len(train_ids)),
            "num_val_ids": int(len(val_ids)),
            "num_test_ids": int(len(test_ids)),
        },
        "data": {
            "num_samples": int(X.shape[0]),
            "num_hours": int(X.shape[1]),
            "num_sensors": int(X.shape[2]),
            "num_channels": int(X.shape[3]),
            "num_targets": int(y.shape[1]),
        },
    }
    (config.METRICS_DIR / "config_snapshot.json").write_text(
        json.dumps(cfg_snapshot, indent=2), encoding="utf-8"
    )
    config.HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    (config.HISTORIES_DIR / "config_snapshot.json").write_text(
        json.dumps(cfg_snapshot, indent=2), encoding="utf-8"
    )

    model = build_cnn2d_model(model_cfg).to(device)

    if config.LOSS != "huber":
        raise ValueError(f"Unsupported LOSS: {config.LOSS}")

    loss_fn = nn.HuberLoss(delta=float(config.HUBER_DELTA), reduction="mean")

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(config.LEARNING_RATE),
        weight_decay=float(config.WEIGHT_DECAY),
    )

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    lr_no_improve = 0

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=int(config.BATCH_SIZE),
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=int(config.BATCH_SIZE),
        shuffle=False,
        drop_last=False,
    )

    history: list[TrainHistoryRow] = []

    for epoch in range(1, int(config.EPOCHS) + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append(TrainHistoryRow(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        improved = val_loss < best_val - 1e-7
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            lr_no_improve = 0

            config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_PATH)
        else:
            epochs_no_improve += 1
            lr_no_improve += 1

        if lr_no_improve >= int(config.REDUCE_LR_PATIENCE):
            for pg in opt.param_groups:
                pg["lr"] = max(float(pg["lr"]) * float(config.REDUCE_LR_FACTOR), float(config.MIN_LR))
            lr_no_improve = 0
            print("Reduced LR; now:", opt.param_groups[0]["lr"])

        if epochs_no_improve >= int(config.EARLY_STOPPING_PATIENCE):
            print("Early stopping triggered.")
            break

    hist_path = config.METRICS_DIR / "training_history.csv"
    hist_path2 = config.HISTORIES_DIR / "training_history.csv"
    df_hist = pd.DataFrame([asdict(r) for r in history])
    df_hist.to_csv(hist_path, index=False)
    config.HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    df_hist.to_csv(hist_path2, index=False)

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "epochs_ran": int(len(history)),
        "model_path": str(config.MODEL_PATH),
    }
    (config.METRICS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (config.HISTORIES_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTrained CNN-2D (baseline)")
    print("Saved model:", config.MODEL_PATH)
    print("Saved history:", hist_path)
    print("Saved history (copy):", hist_path2)


if __name__ == "__main__":
    main()
