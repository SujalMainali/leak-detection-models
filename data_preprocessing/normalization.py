from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class WideNormalizationMeta:
    version: int
    method: str
    normalize_features: bool
    normalize_targets: bool
    feature_columns: List[str]
    target_columns: List[str]
    fitted_on_rows: int


@dataclass(frozen=True)
class WideNormalizationArtifacts:
    feature_scaler: Optional[StandardScaler]
    target_scaler: Optional[StandardScaler]
    meta: WideNormalizationMeta


def normalization_sidecar_paths(processed_csv_path: Path) -> Tuple[Path, Path]:
    """Return (meta_json_path, scalers_joblib_path) for a processed CSV."""
    stem = processed_csv_path.stem
    meta_path = processed_csv_path.with_name(f"{stem}__normalization.json")
    scalers_path = processed_csv_path.with_name(f"{stem}__scalers.joblib")
    return meta_path, scalers_path


def load_normalization_artifacts(processed_csv_path: Path) -> WideNormalizationArtifacts:
    meta_path, scalers_path = normalization_sidecar_paths(processed_csv_path)
    if not meta_path.exists() or not scalers_path.exists():
        raise FileNotFoundError(
            f"Normalization sidecars missing for {processed_csv_path.name}. Expected: {meta_path.name} and {scalers_path.name}"
        )

    meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = WideNormalizationMeta(
        version=int(meta_obj["version"]),
        method=str(meta_obj["method"]),
        normalize_features=bool(meta_obj["normalize_features"]),
        normalize_targets=bool(meta_obj["normalize_targets"]),
        feature_columns=list(meta_obj["feature_columns"]),
        target_columns=list(meta_obj["target_columns"]),
        fitted_on_rows=int(meta_obj["fitted_on_rows"]),
    )

    scalers_obj = joblib.load(scalers_path)
    feature_scaler = scalers_obj.get("feature_scaler")
    target_scaler = scalers_obj.get("target_scaler")

    return WideNormalizationArtifacts(feature_scaler=feature_scaler, target_scaler=target_scaler, meta=meta)


def save_normalization_artifacts(processed_csv_path: Path, artifacts: WideNormalizationArtifacts) -> Tuple[Path, Path]:
    meta_path, scalers_path = normalization_sidecar_paths(processed_csv_path)
    meta_path.write_text(json.dumps(asdict(artifacts.meta), indent=2), encoding="utf-8")
    joblib.dump(
        {
            "feature_scaler": artifacts.feature_scaler,
            "target_scaler": artifacts.target_scaler,
        },
        scalers_path,
    )
    return meta_path, scalers_path


def _as_2d_float(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    return df.loc[:, list(cols)].to_numpy(dtype=float)


def fit_standard_scalers(
    train_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    normalize_features: bool,
    normalize_targets: bool,
) -> Tuple[Optional[StandardScaler], Optional[StandardScaler]]:
    feature_scaler = None
    target_scaler = None

    if normalize_features:
        feature_scaler = StandardScaler()
        feature_scaler.fit(_as_2d_float(train_df, feature_columns))

    if normalize_targets:
        target_scaler = StandardScaler()
        target_scaler.fit(_as_2d_float(train_df, target_columns))

    return feature_scaler, target_scaler


def transform_wide_dataframe(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    feature_scaler: Optional[StandardScaler],
    target_scaler: Optional[StandardScaler],
) -> pd.DataFrame:
    out = df.copy()

    if feature_scaler is not None:
        out.loc[:, list(feature_columns)] = feature_scaler.transform(_as_2d_float(out, feature_columns))

    if target_scaler is not None:
        out.loc[:, list(target_columns)] = target_scaler.transform(_as_2d_float(out, target_columns))

    return out


def inverse_transform_targets(
    y_df: pd.DataFrame,
    *,
    target_columns: Sequence[str],
    target_scaler: Optional[StandardScaler],
) -> pd.DataFrame:
    if target_scaler is None:
        return y_df.copy()

    arr = target_scaler.inverse_transform(_as_2d_float(y_df, target_columns))
    return pd.DataFrame(arr, columns=list(target_columns), index=y_df.index)


def needs_normalization_refresh(
    processed_csv_path: Path,
    *,
    normalize_features: bool,
    normalize_targets: bool,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
) -> bool:
    """Return True if processed CSV should be (re)normalized given current settings."""
    meta_path, scalers_path = normalization_sidecar_paths(processed_csv_path)
    if not processed_csv_path.exists():
        return True
    if not meta_path.exists() or not scalers_path.exists():
        return True

    try:
        artifacts = load_normalization_artifacts(processed_csv_path)
    except Exception:
        return True

    meta = artifacts.meta
    if meta.method != "standard":
        return True
    if meta.normalize_features != normalize_features:
        return True
    if meta.normalize_targets != normalize_targets:
        return True
    if list(meta.feature_columns) != list(feature_columns):
        return True
    if list(meta.target_columns) != list(target_columns):
        return True

    return False


def normalize_and_persist_processed_dataset(
    df_unnormalized: pd.DataFrame,
    *,
    processed_csv_path: Path,
    train_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    normalize_features: bool,
    normalize_targets: bool,
    method: str = "standard",
) -> Tuple[pd.DataFrame, WideNormalizationArtifacts]:
    if method != "standard":
        raise ValueError(f"Unsupported normalization method: {method}")

    feature_scaler, target_scaler = fit_standard_scalers(
        train_df,
        feature_columns=feature_columns,
        target_columns=target_columns,
        normalize_features=normalize_features,
        normalize_targets=normalize_targets,
    )

    df_norm = transform_wide_dataframe(
        df_unnormalized,
        feature_columns=feature_columns,
        target_columns=target_columns,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )

    processed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_norm.to_csv(processed_csv_path, index=False)

    artifacts = WideNormalizationArtifacts(
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        meta=WideNormalizationMeta(
            version=1,
            method=method,
            normalize_features=normalize_features,
            normalize_targets=normalize_targets,
            feature_columns=list(feature_columns),
            target_columns=list(target_columns),
            fitted_on_rows=int(train_df.shape[0]),
        ),
    )
    save_normalization_artifacts(processed_csv_path, artifacts)

    return df_norm, artifacts
