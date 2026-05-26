from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CnnPreprocessResult:
    x: np.ndarray  # (N, 24, S)
    y: np.ndarray  # (N, 3)
    scenario_ids: np.ndarray  # (N,)
    sensor_columns: List[str]
    target_columns: List[str]
    hour_column: str
    scenario_id_column: str


@dataclass(frozen=True)
class CnnScalerArtifacts:
    pressure_scaler: StandardScaler | None
    target_scaler: StandardScaler | None


@dataclass(frozen=True)
class CnnDatasetMeta:
    version: int
    source_csv: str
    scenario_id_col: str
    hour_col: str
    target_columns: List[str]
    sensor_columns: List[str]
    num_scenarios: int
    num_dropped_non_finite_scenarios: int
    dropped_non_finite_scenario_ids: List[int]
    num_hours: int
    num_sensors: int
    normalize_pressures: bool
    normalize_targets: bool
    fitted_on_scenarios: int


def detect_sensor_columns(df: pd.DataFrame, *, prefixes: Sequence[str]) -> List[str]:
    prefixes_u = tuple(p.upper() for p in prefixes)
    cols: List[str] = []
    for c in map(str, df.columns):
        cu = c.upper()
        if cu.startswith(prefixes_u) and "_HOUR" not in cu:
            cols.append(c)
    # deterministic ordering
    return sorted(cols)


def _fit_pressure_scaler(x_train: np.ndarray) -> StandardScaler:
    # Fit per-sensor scaler across all time steps.
    # x_train: (N, T, S) -> (N*T, S)
    n, t, s = x_train.shape
    flat = x_train.reshape(n * t, s)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def _transform_pressure_scaler(scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    n, t, s = x.shape
    flat = x.reshape(n * t, s)
    flat2 = scaler.transform(flat)
    return flat2.reshape(n, t, s)


def preprocess_cnn_long_csv(
    *,
    csv_path: Path,
    scenario_id_col: str,
    hour_col: str,
    target_columns: Sequence[str],
    sensor_prefixes: Sequence[str],
    num_hours: int,
    normalize_pressures: bool,
    normalize_targets: bool,
    # split info for fitting scalers (train only)
    train_scenario_ids: Sequence[int] | None = None,
    out_npz_path: Path | None = None,
    out_scalers_path: Path | None = None,
    out_meta_path: Path | None = None,
) -> Tuple[CnnPreprocessResult, CnnScalerArtifacts, CnnDatasetMeta]:
    df = pd.read_csv(csv_path, low_memory=False)

    required = [scenario_id_col, hour_col, *target_columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CNN long CSV: {missing}")

    sensor_columns = detect_sensor_columns(df, prefixes=sensor_prefixes)
    if not sensor_columns:
        raise ValueError(
            "No sensor columns detected. Expected columns that start with one of "
            f"{list(sensor_prefixes)} (e.g., NODE_3005)."
        )

    # Keep only scenarios with complete hour coverage 0..num_hours-1
    df2 = df.loc[:, [scenario_id_col, hour_col, *target_columns, *sensor_columns]].copy()
    df2[scenario_id_col] = pd.to_numeric(df2[scenario_id_col], errors="raise").astype(int)
    df2[hour_col] = pd.to_numeric(df2[hour_col], errors="raise").astype(int)

    df2 = df2.sort_values([scenario_id_col, hour_col], kind="mergesort")

    per_scenario_rows = df2.groupby(scenario_id_col, sort=False).size()
    good_ids = per_scenario_rows[per_scenario_rows == num_hours].index
    df2 = df2[df2[scenario_id_col].isin(set(map(int, good_ids)))].copy()

    # Validate hours are exactly 0..num_hours-1 for each scenario
    expected_hours = set(range(0, num_hours))
    hours_per_scenario = df2.groupby(scenario_id_col, sort=False)[hour_col].agg(lambda s: set(map(int, s.values)))
    bad = hours_per_scenario[hours_per_scenario.apply(lambda s: s != expected_hours)]
    if not bad.empty:
        examples = bad.head(5).to_dict()
        raise ValueError(f"Hour coverage mismatch for some scenarios (examples): {examples}")

    scenario_ids = df2[scenario_id_col].drop_duplicates().to_numpy(dtype=int)
    n = int(scenario_ids.shape[0])
    s = int(len(sensor_columns))

    # Build X: (N, T, S)
    x = np.empty((n, num_hours, s), dtype=float)
    y = np.empty((n, len(target_columns)), dtype=float)

    grouped = df2.groupby(scenario_id_col, sort=False)
    for i, sid in enumerate(scenario_ids):
        g = grouped.get_group(int(sid)).sort_values(hour_col)
        x[i, :, :] = g.loc[:, sensor_columns].to_numpy(dtype=float)
        # targets are constant per scenario; take first row
        y[i, :] = g.loc[:, list(target_columns)].iloc[0].to_numpy(dtype=float)

    # Drop scenarios with any non-finite values (NaN/Inf) in inputs or targets.
    x_finite = np.isfinite(x.reshape(n, -1)).all(axis=1)
    y_finite = np.isfinite(y).all(axis=1)
    finite_mask = x_finite & y_finite

    dropped_non_finite_ids: List[int] = []
    if not bool(np.all(finite_mask)):
        dropped_non_finite_ids = scenario_ids[~finite_mask].astype(int).tolist()
        scenario_ids = scenario_ids[finite_mask]
        x = x[finite_mask]
        y = y[finite_mask]
        n = int(scenario_ids.shape[0])
        if n == 0:
            raise ValueError(
                "All scenarios were dropped due to non-finite values in inputs/targets. "
                f"Example dropped IDs: {dropped_non_finite_ids[:10]}"
            )

    # Fit scalers on train only if train_scenario_ids given, else fit on all (not recommended)
    train_ids_set = None
    if train_scenario_ids is not None:
        train_ids_set = set(map(int, train_scenario_ids))

    if normalize_pressures:
        if train_ids_set is None:
            x_fit = x
            fitted_on = int(n)
        else:
            fit_mask = np.array([int(sid) in train_ids_set for sid in scenario_ids], dtype=bool)
            x_fit = x[fit_mask]
            fitted_on = int(fit_mask.sum())
        pressure_scaler = _fit_pressure_scaler(x_fit)
        x = _transform_pressure_scaler(pressure_scaler, x)
    else:
        pressure_scaler = None
        fitted_on = int(n if train_ids_set is None else len(train_ids_set))

    if normalize_targets:
        if train_ids_set is None:
            y_fit = y
        else:
            fit_mask = np.array([int(sid) in train_ids_set for sid in scenario_ids], dtype=bool)
            y_fit = y[fit_mask]
        target_scaler = StandardScaler()
        target_scaler.fit(y_fit)
        y = target_scaler.transform(y)
    else:
        target_scaler = None

    res = CnnPreprocessResult(
        x=x,
        y=y,
        scenario_ids=scenario_ids,
        sensor_columns=list(sensor_columns),
        target_columns=list(map(str, target_columns)),
        hour_column=str(hour_col),
        scenario_id_column=str(scenario_id_col),
    )

    artifacts = CnnScalerArtifacts(pressure_scaler=pressure_scaler, target_scaler=target_scaler)

    meta = CnnDatasetMeta(
        version=2,
        source_csv=str(csv_path),
        scenario_id_col=str(scenario_id_col),
        hour_col=str(hour_col),
        target_columns=list(map(str, target_columns)),
        sensor_columns=list(sensor_columns),
        num_scenarios=int(n),
        num_dropped_non_finite_scenarios=int(len(dropped_non_finite_ids)),
        dropped_non_finite_scenario_ids=list(map(int, dropped_non_finite_ids)),
        num_hours=int(num_hours),
        num_sensors=int(s),
        normalize_pressures=bool(normalize_pressures),
        normalize_targets=bool(normalize_targets),
        fitted_on_scenarios=int(fitted_on),
    )

    if out_npz_path is not None:
        out_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz_path,
            X=res.x,
            y=res.y,
            scenario_ids=res.scenario_ids,
            sensor_columns=np.array(res.sensor_columns, dtype=object),
            target_columns=np.array(res.target_columns, dtype=object),
        )

    if out_scalers_path is not None:
        out_scalers_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pressure_scaler": artifacts.pressure_scaler, "target_scaler": artifacts.target_scaler},
            out_scalers_path,
        )

    if out_meta_path is not None:
        out_meta_path.parent.mkdir(parents=True, exist_ok=True)
        out_meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    return res, artifacts, meta
