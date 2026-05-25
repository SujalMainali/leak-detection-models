from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class RfPreprocessingResult:
    df: pd.DataFrame
    feature_columns: List[str]
    target_columns: List[str]
    dropped_rows_missing_targets: int
    dropped_rows_missing_features: int
    dropped_rows_total: int


def _normalize_col(col: str) -> str:
    return re.sub(r"\s+", "_", str(col).strip().lower())


def _parse_sensor_col(
    col: str,
    allowed_prefixes: Sequence[str],
) -> Optional[Tuple[str, str, int]]:
    # Accept e.g. NODEADD_2423_Hour7 (case-insensitive)
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

    return (
        m.group("prefix"),
        m.group("node_id"),
        int(m.group("hour")),
    )


def detect_sensor_columns(
    columns: Iterable[str],
    *,
    allowed_prefixes: Sequence[str],
    valid_hours: range,
) -> List[str]:
    parsed: List[Tuple[str, str, int, str]] = []
    for c in columns:
        out = _parse_sensor_col(c, allowed_prefixes)
        if out is None:
            continue
        node_type, node_id, hour = out
        if hour not in valid_hours:
            continue
        parsed.append((node_type, node_id, hour, str(c)))

    parsed.sort(key=lambda t: (t[0], t[1], t[2]))
    return [c for *_ignore, c in parsed]


def load_raw_wide_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)


def build_rf_baseline_dataframe(
    df_raw: pd.DataFrame,
    *,
    scenario_id_col: str,
    target_columns: Sequence[str],
    sensor_node_prefixes: Sequence[str],
    valid_hours: range,
) -> RfPreprocessingResult:
    if scenario_id_col not in df_raw.columns:
        raise ValueError(f"scenario_id column not found: {scenario_id_col}")

    missing_targets = [c for c in target_columns if c not in df_raw.columns]
    if missing_targets:
        raise ValueError(f"Target columns missing from raw CSV: {missing_targets}")

    sensor_cols = detect_sensor_columns(
        df_raw.columns,
        allowed_prefixes=sensor_node_prefixes,
        valid_hours=valid_hours,
    )
    if not sensor_cols:
        raise ValueError(
            "No sensor columns detected. Check SENSOR_NODE_PREFIXES and naming scheme."
        )

    keep_cols = [scenario_id_col, *target_columns, *sensor_cols]
    df = df_raw.loc[:, keep_cols].copy()

    before = int(df.shape[0])

    # Drop rows missing any targets (RF cannot fit NaN in y)
    df1 = df.dropna(subset=list(target_columns))
    dropped_missing_targets = before - int(df1.shape[0])

    # Drop rows missing any features (classic RandomForest cannot handle NaN in X)
    before_feat = int(df1.shape[0])
    df2 = df1.dropna(subset=list(sensor_cols))
    dropped_missing_features = before_feat - int(df2.shape[0])

    df = df2
    dropped_total = before - int(df.shape[0])

    # Ensure stable ordering: scenario_id, targets, then sensors sorted by (node_type,node_id,hour)
    feature_columns = sensor_cols
    return RfPreprocessingResult(
        df=df,
        feature_columns=list(feature_columns),
        target_columns=list(target_columns),
        dropped_rows_missing_targets=dropped_missing_targets,
        dropped_rows_missing_features=dropped_missing_features,
        dropped_rows_total=dropped_total,
    )


def save_processed_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def run_rf_preprocessing(
    *,
    raw_csv_path: Path,
    processed_csv_path: Path,
    scenario_id_col: str,
    target_columns: Sequence[str],
    sensor_node_prefixes: Sequence[str],
    valid_hours: range,
) -> RfPreprocessingResult:
    df_raw = load_raw_wide_csv(raw_csv_path)
    result = build_rf_baseline_dataframe(
        df_raw,
        scenario_id_col=scenario_id_col,
        target_columns=target_columns,
        sensor_node_prefixes=sensor_node_prefixes,
        valid_hours=valid_hours,
    )
    save_processed_csv(result.df, processed_csv_path)
    return result


if __name__ == "__main__":
    from models.random_forest import config

    res = run_rf_preprocessing(
        raw_csv_path=config.RAW_CSV_PATH,
        processed_csv_path=config.PROCESSED_CSV_PATH,
        scenario_id_col=config.SCENARIO_ID_COL,
        target_columns=config.TARGET_COLUMNS,
        sensor_node_prefixes=config.SENSOR_NODE_PREFIXES,
        valid_hours=config.VALID_HOURS,
    )

    print("Saved processed dataset to:", config.PROCESSED_CSV_PATH)
    print("Rows:", int(res.df.shape[0]))
    print("Columns:", int(res.df.shape[1]))
    print("Dropped rows missing targets:", res.dropped_rows_missing_targets)
    print("Features:", len(res.feature_columns))
