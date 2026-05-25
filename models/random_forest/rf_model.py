from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


@dataclass
class TrainedRandomForestBaseline:
    feature_columns: List[str]
    target_columns: List[str]
    models: Dict[str, RandomForestRegressor]

    def predict(self, df_features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in df_features.columns]
        if missing:
            raise ValueError(f"Missing feature columns at predict time: {missing[:20]}")

        x = df_features.loc[:, self.feature_columns].to_numpy(dtype=float)
        preds: Dict[str, np.ndarray] = {}
        for target in self.target_columns:
            preds[target] = self.models[target].predict(x)
        return pd.DataFrame(preds)

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "rf_baseline.joblib"
        joblib.dump(
            {
                "feature_columns": self.feature_columns,
                "target_columns": self.target_columns,
                "models": self.models,
            },
            out_path,
        )
        return out_path

    @staticmethod
    def load(path: Path) -> "TrainedRandomForestBaseline":
        obj = joblib.load(path)
        return TrainedRandomForestBaseline(
            feature_columns=list(obj["feature_columns"]),
            target_columns=list(obj["target_columns"]),
            models=dict(obj["models"]),
        )


def train_three_random_forests(
    *,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    rf_params: dict,
) -> TrainedRandomForestBaseline:
    x = x_train.loc[:, list(feature_columns)].to_numpy(dtype=float)

    models: Dict[str, RandomForestRegressor] = {}
    for target in target_columns:
        y = y_train[target].to_numpy(dtype=float)
        model = RandomForestRegressor(**rf_params)
        model.fit(x, y)
        models[target] = model

    return TrainedRandomForestBaseline(
        feature_columns=list(feature_columns),
        target_columns=list(target_columns),
        models=models,
    )
