from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class SplitIdsResult:
    train_ids: pd.Series
    val_ids: pd.Series
    test_ids: pd.Series


def split_ids(
    df: pd.DataFrame,
    *,
    id_col: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_state: int,
) -> SplitIdsResult:
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"train/val/test fractions must sum to 1.0 (got {total}).")
    if id_col not in df.columns:
        raise ValueError(f"id_col not found in df: {id_col}")

    ids = df[id_col]
    # If ids are unique per row (current dataset), this is effectively a row split.
    # If not, this at least ensures the split is expressed in terms of IDs.
    unique_ids = ids.drop_duplicates().reset_index(drop=True)

    train_val_ids, test_ids = train_test_split(
        unique_ids,
        test_size=test_fraction,
        random_state=random_state,
        shuffle=True,
    )

    val_size_relative = val_fraction / max(1.0 - test_fraction, 1e-12)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size_relative,
        random_state=random_state,
        shuffle=True,
    )

    return SplitIdsResult(
        train_ids=train_ids.reset_index(drop=True),
        val_ids=val_ids.reset_index(drop=True),
        test_ids=test_ids.reset_index(drop=True),
    )


def save_split_ids(
    split_ids_res: SplitIdsResult,
    *,
    splits_dir: Path,
    id_col: str,
    prefix: str = "rf_ids",
) -> Tuple[Path, Path, Path]:
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_path = splits_dir / f"{prefix}_train.csv"
    val_path = splits_dir / f"{prefix}_val.csv"
    test_path = splits_dir / f"{prefix}_test.csv"

    split_ids_res.train_ids.to_frame(name=id_col).to_csv(train_path, index=False)
    split_ids_res.val_ids.to_frame(name=id_col).to_csv(val_path, index=False)
    split_ids_res.test_ids.to_frame(name=id_col).to_csv(test_path, index=False)

    return train_path, val_path, test_path


def load_split_ids(
    *,
    splits_dir: Path,
    id_col: str,
    prefix: str = "rf_ids",
) -> SplitIdsResult:
    train_path = splits_dir / f"{prefix}_train.csv"
    val_path = splits_dir / f"{prefix}_val.csv"
    test_path = splits_dir / f"{prefix}_test.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing split-id files under {splits_dir}. Expected: {train_path.name}, {val_path.name}, {test_path.name}"
        )

    train_ids = pd.read_csv(train_path, low_memory=False)[id_col]
    val_ids = pd.read_csv(val_path, low_memory=False)[id_col]
    test_ids = pd.read_csv(test_path, low_memory=False)[id_col]

    return SplitIdsResult(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)


def apply_split_ids(
    df: pd.DataFrame,
    *,
    id_col: str,
    split_ids_res: SplitIdsResult,
) -> SplitResult:
    if id_col not in df.columns:
        raise ValueError(f"id_col not found in df: {id_col}")

    train_df = df[df[id_col].isin(set(split_ids_res.train_ids))].copy()
    val_df = df[df[id_col].isin(set(split_ids_res.val_ids))].copy()
    test_df = df[df[id_col].isin(set(split_ids_res.test_ids))].copy()
    return SplitResult(train=train_df, val=val_df, test=test_df)
