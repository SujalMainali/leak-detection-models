from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_preprocessing.cnn_preprocessing import detect_sensor_columns


@dataclass(frozen=True)
class Cnn2dPreprocessResult:
    x: np.ndarray  # (N, 24, S, 1)
    y: np.ndarray  # (N, 3)
    scenario_ids: np.ndarray  # (N,)
    sensor_columns: List[str]
    target_columns: List[str]
    hour_column: str
    scenario_id_column: str


@dataclass(frozen=True)
class Cnn2dScalerArtifacts:
    pressure_scaler: StandardScaler | None
    target_scaler: StandardScaler | None


@dataclass(frozen=True)
class Cnn2dDatasetMeta:
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
    pressure_normalization_mode: str
    fitted_on_scenarios: int


def _fit_pressure_scaler_per_sensor(x_train: np.ndarray) -> StandardScaler:
    # x_train: (N, T, S)
    n, t, s = x_train.shape
    flat = x_train.reshape(n * t, s)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def _transform_pressure_scaler_per_sensor(scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    n, t, s = x.shape
    flat = x.reshape(n * t, s)
    flat2 = scaler.transform(flat)
    return flat2.reshape(n, t, s)


def preprocess_cnn2d_long_csv(
    *,
    csv_path: Path,
    scenario_id_col: str,
    hour_col: str,
    target_columns: Sequence[str],
    sensor_prefixes: Sequence[str],
    num_hours: int,
    normalize_pressures: bool,
    pressure_normalization_mode: str,
    normalize_targets: bool,
    train_scenario_ids: Sequence[int] | None = None,
    out_npz_path: Path | None = None,
    out_scalers_path: Path | None = None,
    out_meta_path: Path | None = None,
) -> Tuple[Cnn2dPreprocessResult, Cnn2dScalerArtifacts, Cnn2dDatasetMeta]:
    df = pd.read_csv(csv_path, low_memory=False)

    required = [scenario_id_col, hour_col, *target_columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CNN-2D long CSV: {missing}")

    sensor_columns = detect_sensor_columns(df, prefixes=sensor_prefixes)
    if not sensor_columns:
        raise ValueError(
            "No sensor columns detected. Expected columns that start with one of "
            f"{list(sensor_prefixes)} (e.g., NODE_3005)."
        )

    df2 = df.loc[:, [scenario_id_col, hour_col, *target_columns, *sensor_columns]].copy()
    df2[scenario_id_col] = pd.to_numeric(df2[scenario_id_col], errors="raise").astype(int)
    df2[hour_col] = pd.to_numeric(df2[hour_col], errors="raise").astype(int)
    df2 = df2.sort_values([scenario_id_col, hour_col], kind="mergesort")

    per_scenario_rows = df2.groupby(scenario_id_col, sort=False).size()
    good_ids = per_scenario_rows[per_scenario_rows == num_hours].index
    df2 = df2[df2[scenario_id_col].isin(set(map(int, good_ids)))].copy()

    expected_hours = set(range(0, num_hours))
    hours_per_scenario = df2.groupby(scenario_id_col, sort=False)[hour_col].agg(lambda s: set(map(int, s.values)))
    bad = hours_per_scenario[hours_per_scenario.apply(lambda s: s != expected_hours)]
    if not bad.empty:
        examples = bad.head(5).to_dict()
        raise ValueError(f"Hour coverage mismatch for some scenarios (examples): {examples}")

    scenario_ids = df2[scenario_id_col].drop_duplicates().to_numpy(dtype=int)
    n = int(scenario_ids.shape[0])
    s = int(len(sensor_columns))

    # Base pressure tensor: (N, T, S)
    x3 = np.empty((n, num_hours, s), dtype=float)
    y = np.empty((n, len(target_columns)), dtype=float)

    grouped = df2.groupby(scenario_id_col, sort=False)
    for i, sid in enumerate(scenario_ids):
        g = grouped.get_group(int(sid)).sort_values(hour_col)
        x3[i, :, :] = g.loc[:, sensor_columns].to_numpy(dtype=float)
        y[i, :] = g.loc[:, list(target_columns)].iloc[0].to_numpy(dtype=float)

    # Drop scenarios with any non-finite values (NaN/Inf) in inputs or targets.
    x_finite = np.isfinite(x3.reshape(n, -1)).all(axis=1)
    y_finite = np.isfinite(y).all(axis=1)
    finite_mask = x_finite & y_finite

    dropped_non_finite_ids: List[int] = []
    if not bool(np.all(finite_mask)):
        dropped_non_finite_ids = scenario_ids[~finite_mask].astype(int).tolist()
        scenario_ids = scenario_ids[finite_mask]
        x3 = x3[finite_mask]
        y = y[finite_mask]
        n = int(scenario_ids.shape[0])
        if n == 0:
            raise ValueError(
                "All scenarios were dropped due to non-finite values in inputs/targets. "
                f"Example dropped IDs: {dropped_non_finite_ids[:10]}"
            )

    train_ids_set = None
    if train_scenario_ids is not None:
        train_ids_set = set(map(int, train_scenario_ids))

    fitted_on = int(n if train_ids_set is None else len(train_ids_set))

    if normalize_pressures:
        if pressure_normalization_mode != "per_sensor":
            raise ValueError(f"Unsupported PRESSURE_NORMALIZATION_MODE: {pressure_normalization_mode}")

        if train_ids_set is None:
            x_fit = x3
            fitted_on = int(n)
        else:
            fit_mask = np.array([int(sid) in train_ids_set for sid in scenario_ids], dtype=bool)
            x_fit = x3[fit_mask]
            fitted_on = int(fit_mask.sum())

        pressure_scaler = _fit_pressure_scaler_per_sensor(x_fit)
        x3 = _transform_pressure_scaler_per_sensor(pressure_scaler, x3)
    else:
        pressure_scaler = None

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

    # Expand channel dimension: (N, T, S, 1)
    x = x3[:, :, :, None]

    res = Cnn2dPreprocessResult(
        x=x,
        y=y,
        scenario_ids=scenario_ids,
        sensor_columns=list(sensor_columns),
        target_columns=list(map(str, target_columns)),
        hour_column=str(hour_col),
        scenario_id_column=str(scenario_id_col),
    )

    artifacts = Cnn2dScalerArtifacts(pressure_scaler=pressure_scaler, target_scaler=target_scaler)

    meta = Cnn2dDatasetMeta(
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
        pressure_normalization_mode=str(pressure_normalization_mode),
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
