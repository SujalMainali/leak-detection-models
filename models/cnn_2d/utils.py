from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class TargetMetrics:
    target: str
    mae: float
    rmse: float
    r2: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_targets(
    *,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    target_columns: Sequence[str],
) -> pd.DataFrame:
    rows: List[TargetMetrics] = []
    for t in target_columns:
        m = compute_metrics(y_true[t].to_numpy(dtype=float), y_pred[t].to_numpy(dtype=float))
        rows.append(TargetMetrics(target=t, **m))
    return pd.DataFrame([asdict(r) for r in rows]).sort_values("target")


def save_metrics(df_metrics: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(out_path, index=False)


def save_metrics_json(df_metrics: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(df_metrics.to_dict(orient="records"), indent=2), encoding="utf-8")


def save_predictions(
    *,
    scenario_ids: np.ndarray,
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
    out_path: Path,
    target_columns: Sequence[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame({"scenario_id": scenario_ids.astype(int)})

    if y_true is not None:
        for i, t in enumerate(target_columns):
            out[f"true_{t}"] = y_true[:, i]

    for i, t in enumerate(target_columns):
        out[f"pred_{t}"] = y_pred[:, i]

    out.to_csv(out_path, index=False)
