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
from models.cnn import config
from models.cnn.cnn_model import CnnModelConfig, build_cnn_model
from models.cnn.utils import save_predictions


def _require_torch():
    try:
        import torch

        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required to run CNN predictions. Install it in LeakEnv: `pip install torch`."
        ) from e


def main() -> None:
    torch = _require_torch()

    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(config.CNN_READY_NPZ_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)
    scenario_ids = data["scenario_ids"].astype(int)
    target_columns = list(data["target_columns"].tolist())

    scalers = joblib.load(config.CNN_SCALERS_PATH) if config.CNN_SCALERS_PATH.exists() else {}
    target_scaler = scalers.get("target_scaler")

    ids_split = load_split_ids(splits_dir=config.SPLIT_IDS_DIR, id_col=config.SCENARIO_ID_COL, prefix="cnn_ids")
    df_ids = pd.DataFrame({config.SCENARIO_ID_COL: scenario_ids.astype(int)})
    splits = apply_split_ids(df_ids, id_col=config.SCENARIO_ID_COL, split_ids_res=ids_split)

    test_set = set(map(int, splits.test[config.SCENARIO_ID_COL].tolist()))
    test_mask = np.array([int(sid) in test_set for sid in scenario_ids], dtype=bool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = CnnModelConfig(
        num_sensors=int(X.shape[2]),
        conv_filters=list(map(int, config.CONV_FILTERS)),
        kernel_size=int(config.KERNEL_SIZE),
        pool_size=int(config.POOL_SIZE),
        dropout_rates=list(map(float, config.DROPOUT_RATES)),
        dense_units=list(map(int, config.DENSE_UNITS)),
        use_global_avg_pool=bool(config.USE_GLOBAL_AVG_POOL),
        num_outputs=len(target_columns),
    )

    model = build_cnn_model(model_cfg).to(device)
    state = torch.load(config.MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        xb = torch.from_numpy(X[test_mask]).to(device)
        pred = model(xb).detach().cpu().numpy()

    if target_scaler is not None:
        pred = target_scaler.inverse_transform(pred)

    save_predictions(
        scenario_ids=scenario_ids[test_mask],
        y_true=None,
        y_pred=pred,
        out_path=config.PREDICTIONS_DIR / "test_predictions_only.csv",
        target_columns=target_columns,
    )

    print("Wrote:", (config.PREDICTIONS_DIR / "test_predictions_only.csv").as_posix())


if __name__ == "__main__":
    main()
