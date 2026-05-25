from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd


def _safe_int(value: object) -> int:
    """Convert pandas/numpy scalar-ish values to a plain Python int.

    This also avoids Pylance false-positives around `numpy.typing.Scalar`.
    """
    # numpy scalars often have `.item()`; if not, `int()` will handle normal Python ints.
    if hasattr(value, "item"):
        return int(cast(object, value).item())  # type: ignore[attr-defined]
    return int(cast(int, value))


@dataclass(frozen=True)
class RfFeatureEngineeringResult:
    df: pd.DataFrame
    feature_columns: List[str]
    target_columns: List[str]
    feature_set_tag: str
    dropped_rows_missing_targets: int
    dropped_rows_missing_features: int
    dropped_rows_total: int
    feature_columns_path: Optional[Path]


def _normalize_col(col: str) -> str:
    return re.sub(r"\s+", "_", str(col).strip().lower())


def _extract_sensor_base(col: str) -> str:
    # Keep original base string but strip the trailing _HourXX / _hourXX
    return re.sub(r"(?i)_hour\d{1,3}$", "", str(col).strip())


def _parse_sensor_col(
    col: str,
    allowed_prefixes: Sequence[str],
) -> Optional[Tuple[str, str, int, str]]:
    """Return (node_type_norm, node_id_norm, hour, base_original)."""
    cn = _normalize_col(col)
    prefixes_norm = [re.escape(_normalize_col(p)) for p in allowed_prefixes]
    if not prefixes_norm:
        return None

    prefix_group = "|".join(prefixes_norm)
    m = re.match(
        rf"^(?P<prefix>{prefix_group})_(?P<node_id>[a-z0-9]+)_hour(?P<hour>\d{{1,3}})$",
        cn,
    )
    if not m:
        return None

    base_original = _extract_sensor_base(col)
    return (
        m.group("prefix"),
        m.group("node_id"),
        int(m.group("hour")),
        base_original,
    )


def detect_sensor_columns(
    columns: Iterable[str],
    *,
    allowed_prefixes: Sequence[str],
    valid_hours: range,
) -> Tuple[List[str], pd.DataFrame]:
    """Detect sensor columns and return them in stable (node_type,node_id,hour) order."""
    rows: List[Dict[str, object]] = []
    for c in columns:
        out = _parse_sensor_col(c, allowed_prefixes)
        if out is None:
            continue
        node_type, node_id, hour, base = out
        if hour not in valid_hours:
            continue
        rows.append({"column": str(c), "node_type": node_type, "node_id": node_id, "hour": hour, "base": base})

    if not rows:
        return [], pd.DataFrame(columns=["column", "node_type", "node_id", "hour", "base"])

    info = pd.DataFrame(rows).sort_values(["node_type", "node_id", "hour"])
    return info["column"].tolist(), info


def _compute_first_differences(
    df: pd.DataFrame,
    *,
    sensor_info: pd.DataFrame,
    valid_hours: range,
) -> Tuple[pd.DataFrame, List[str]]:
    """Create per-sensor first differences P(t) - P(t-1) for hours in valid_hours."""
    diff_cols: List[str] = []
    diff_data: Dict[str, pd.Series] = {}

    if sensor_info.empty:
        return pd.DataFrame(index=df.index), []

    for base, grp in sensor_info.groupby("base", sort=False):
        hour_to_col = {_safe_int(r.hour): str(r.column) for r in grp.itertuples(index=False)}
        for hour in valid_hours:
            if hour == min(valid_hours):
                continue
            prev = hour - 1
            if prev not in hour_to_col or hour not in hour_to_col:
                # Keep a stable schema: create the column but it may contain NaN.
                col_name = f"{base}_DiffHour{hour}"
                diff_cols.append(col_name)
                diff_data[col_name] = pd.Series(np.nan, index=df.index)
                continue

            col_prev = hour_to_col[prev]
            col_cur = hour_to_col[hour]
            col_name = f"{base}_DiffHour{hour}"
            diff_cols.append(col_name)
            diff_data[col_name] = df[col_cur] - df[col_prev]

    diff_df = pd.DataFrame(diff_data)
    return diff_df, diff_cols


def _compute_leak_window_summary(
    df: pd.DataFrame,
    *,
    sensor_info: pd.DataFrame,
    leak_start_col: str,
    leak_duration_col: str,
    valid_hours: range,
) -> Tuple[pd.DataFrame, List[str]]:
    if sensor_info.empty:
        return pd.DataFrame(index=df.index), []

    if leak_start_col not in df.columns or leak_duration_col not in df.columns:
        raise ValueError(
            f"Leak-window summary enabled but leak window columns missing: {leak_start_col}, {leak_duration_col}"
        )

    hours = np.array(list(valid_hours), dtype=int)
    if hours.min() != 0 or hours.max() != 23 or hours.size != 24:
        raise ValueError("Leak-window summary currently expects valid_hours=range(0, 24).")

    # Build per-row leak mask: [n, 24]
    start = df[leak_start_col].to_numpy(dtype=float)
    dur = df[leak_duration_col].to_numpy(dtype=float)

    start_int = np.nan_to_num(start, nan=0.0).astype(int)
    dur_int = np.nan_to_num(dur, nan=0.0).astype(int)
    start_int = np.clip(start_int, 0, 23)
    end_int = np.clip(start_int + dur_int, 0, 24)

    h = hours[None, :]
    leak_mask = (h >= start_int[:, None]) & (h < end_int[:, None])
    base_mask = ~leak_mask

    leak_count = leak_mask.sum(axis=1).astype(float)
    base_count = base_mask.sum(axis=1).astype(float)

    out: Dict[str, np.ndarray] = {}
    out_cols: List[str] = []

    for base, grp in sensor_info.groupby("base", sort=False):
        hour_to_col = {_safe_int(r.hour): str(r.column) for r in grp.itertuples(index=False)}
        cols_0_23 = [hour_to_col.get(i) for i in range(0, 24)]
        if any(c is None for c in cols_0_23):
            # If the dataset is missing some hours for this sensor, skip summaries for it.
            continue

        values = df[cols_0_23].to_numpy(dtype=float)

        leak_sum = (values * leak_mask).sum(axis=1)
        base_sum = (values * base_mask).sum(axis=1)

        leak_mean = np.divide(leak_sum, leak_count, out=np.full_like(leak_sum, np.nan), where=leak_count > 0)
        base_mean = np.divide(base_sum, base_count, out=np.full_like(base_sum, np.nan), where=base_count > 0)

        min_pressure = values.min(axis=1)
        min_hour = values.argmin(axis=1).astype(float)

        diffs = np.diff(values, axis=1)  # Hour1 - Hour0, ..., Hour23 - Hour22
        min_diff = diffs.min(axis=1)
        max_drop = np.maximum(0.0, -min_diff)
        max_drop_hour = (diffs.argmin(axis=1) + 1).astype(float)

        feats = {
            f"{base}_LeakMean": leak_mean,
            f"{base}_BaseMean": base_mean,
            f"{base}_LeakMinusBase": leak_mean - base_mean,
            f"{base}_MinPressure": min_pressure,
            f"{base}_MaxDrop": max_drop,
            f"{base}_MinHour": min_hour,
            f"{base}_MaxDropHour": max_drop_hour,
        }

        for k, v in feats.items():
            out[k] = v
            out_cols.append(k)

    summary_df = pd.DataFrame(out, index=df.index)
    return summary_df, out_cols


def _feature_set_tag(*, use_first_differences: bool, use_leak_window_summary: bool) -> str:
    if use_first_differences and use_leak_window_summary:
        return "feature_set_2"
    if use_first_differences and not use_leak_window_summary:
        return "feature_set_1"
    return "raw_only"


def build_feature_engineered_dataframe(
    df_raw: pd.DataFrame,
    *,
    scenario_id_col: str,
    target_columns: Sequence[str],
    sensor_node_prefixes: Sequence[str],
    valid_hours: range,
    use_raw_pressure: bool,
    use_first_differences: bool,
    use_leak_window_summary: bool,
    leak_start_col: str,
    leak_duration_col: str,
) -> RfFeatureEngineeringResult:
    if scenario_id_col not in df_raw.columns:
        raise ValueError(f"scenario_id column not found: {scenario_id_col}")

    missing_targets = [c for c in target_columns if c not in df_raw.columns]
    if missing_targets:
        raise ValueError(f"Target columns missing from raw CSV: {missing_targets}")

    sensor_cols, sensor_info = detect_sensor_columns(
        df_raw.columns,
        allowed_prefixes=sensor_node_prefixes,
        valid_hours=valid_hours,
    )
    if not sensor_cols:
        raise ValueError("No sensor columns detected. Check SENSOR_NODE_PREFIXES and naming scheme.")

    engineered_parts: List[pd.DataFrame] = []
    feature_columns: List[str] = []

    if use_raw_pressure:
        engineered_parts.append(df_raw.loc[:, sensor_cols].copy())
        feature_columns.extend(sensor_cols)

    if use_first_differences:
        diff_df, diff_cols = _compute_first_differences(df_raw, sensor_info=sensor_info, valid_hours=valid_hours)
        engineered_parts.append(diff_df)
        feature_columns.extend(diff_cols)

    if use_leak_window_summary:
        summary_df, summary_cols = _compute_leak_window_summary(
            df_raw,
            sensor_info=sensor_info,
            leak_start_col=leak_start_col,
            leak_duration_col=leak_duration_col,
            valid_hours=valid_hours,
        )
        engineered_parts.append(summary_df)
        feature_columns.extend(summary_cols)

    if not feature_columns:
        raise ValueError("No feature groups enabled. Set at least one of the feature flags to True.")

    tag = _feature_set_tag(
        use_first_differences=use_first_differences,
        use_leak_window_summary=use_leak_window_summary,
    )

    # Assemble final dataframe in the requested wide format
    df_features = pd.concat(engineered_parts, axis=1)
    df_out = pd.concat(
        [
            df_raw[[scenario_id_col]].reset_index(drop=True),
            df_features.reset_index(drop=True),
            df_raw[list(target_columns)].reset_index(drop=True),
        ],
        axis=1,
    )

    before = int(df_out.shape[0])
    df1 = df_out.dropna(subset=list(target_columns))
    dropped_missing_targets = before - int(df1.shape[0])

    before_feat = int(df1.shape[0])
    df2 = df1.dropna(subset=feature_columns)
    dropped_missing_features = before_feat - int(df2.shape[0])

    df_final = df2
    dropped_total = before - int(df_final.shape[0])

    return RfFeatureEngineeringResult(
        df=df_final,
        feature_columns=list(feature_columns),
        target_columns=list(target_columns),
        feature_set_tag=tag,
        dropped_rows_missing_targets=dropped_missing_targets,
        dropped_rows_missing_features=dropped_missing_features,
        dropped_rows_total=dropped_total,
        feature_columns_path=None,
    )


def run_rf_feature_engineering(
    *,
    raw_csv_path: Path,
    processed_csv_path: Path,
    feature_columns_out_path: Optional[Path],
    scenario_id_col: str,
    target_columns: Sequence[str],
    sensor_node_prefixes: Sequence[str],
    valid_hours: range,
    use_raw_pressure: bool,
    use_first_differences: bool,
    use_leak_window_summary: bool,
    leak_start_col: str,
    leak_duration_col: str,
) -> RfFeatureEngineeringResult:
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")

    df_raw = pd.read_csv(raw_csv_path, low_memory=False)
    result = build_feature_engineered_dataframe(
        df_raw,
        scenario_id_col=scenario_id_col,
        target_columns=target_columns,
        sensor_node_prefixes=sensor_node_prefixes,
        valid_hours=valid_hours,
        use_raw_pressure=use_raw_pressure,
        use_first_differences=use_first_differences,
        use_leak_window_summary=use_leak_window_summary,
        leak_start_col=leak_start_col,
        leak_duration_col=leak_duration_col,
    )

    processed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.df.to_csv(processed_csv_path, index=False)

    feature_columns_out_path_final = None
    if feature_columns_out_path is not None:
        feature_columns_out_path.parent.mkdir(parents=True, exist_ok=True)
        feature_columns_out_path.write_text("\n".join(result.feature_columns) + "\n", encoding="utf-8")
        feature_columns_out_path_final = feature_columns_out_path

    return RfFeatureEngineeringResult(
        df=result.df,
        feature_columns=result.feature_columns,
        target_columns=result.target_columns,
        feature_set_tag=result.feature_set_tag,
        dropped_rows_missing_targets=result.dropped_rows_missing_targets,
        dropped_rows_missing_features=result.dropped_rows_missing_features,
        dropped_rows_total=result.dropped_rows_total,
        feature_columns_path=feature_columns_out_path_final,
    )
