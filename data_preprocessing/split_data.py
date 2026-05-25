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


def split_wide_dataframe(
    df: pd.DataFrame,
    *,
    id_col: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_state: int,
) -> SplitResult:
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"train/val/test fractions must sum to 1.0 (got {total})."
        )
    if id_col not in df.columns:
        raise ValueError(f"id_col not found in df: {id_col}")

    # If each row is independent (as in current wide dataset), a random split is OK.
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_fraction,
        random_state=random_state,
        shuffle=True,
    )

    # val_fraction is of total; convert to fraction of train_val
    val_size_relative = val_fraction / max(1.0 - test_fraction, 1e-12)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_relative,
        random_state=random_state,
        shuffle=True,
    )

    return SplitResult(train=train_df, val=val_df, test=test_df)


def save_splits(
    splits: SplitResult,
    *,
    splits_dir: Path,
    prefix: str = "rf",
) -> Tuple[Path, Path, Path]:
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_path = splits_dir / f"{prefix}_train.csv"
    val_path = splits_dir / f"{prefix}_val.csv"
    test_path = splits_dir / f"{prefix}_test.csv"

    splits.train.to_csv(train_path, index=False)
    splits.val.to_csv(val_path, index=False)
    splits.test.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def load_splits(
    *,
    splits_dir: Path,
    prefix: str = "rf",
) -> SplitResult:
    train_path = splits_dir / f"{prefix}_train.csv"
    val_path = splits_dir / f"{prefix}_val.csv"
    test_path = splits_dir / f"{prefix}_test.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing split files under {splits_dir}. Expected: {train_path.name}, {val_path.name}, {test_path.name}"
        )

    return SplitResult(
        train=pd.read_csv(train_path, low_memory=False),
        val=pd.read_csv(val_path, low_memory=False),
        test=pd.read_csv(test_path, low_memory=False),
    )
