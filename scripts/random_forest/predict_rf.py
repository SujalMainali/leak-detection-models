from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running this script directly (adds repo root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.random_forest import config
from models.random_forest.rf_model import TrainedRandomForestBaseline


def main() -> None:
    input_csv = Path(config.PREDICT_INPUT_CSV_PATH)
    model_path = Path(config.MODEL_BUNDLE_PATH)
    output_csv = Path(config.PREDICT_OUTPUT_CSV_PATH)

    df = pd.read_csv(input_csv, low_memory=False)
    model = TrainedRandomForestBaseline.load(model_path)

    preds = model.predict(df[model.feature_columns])
    out = pd.concat([df[[config.SCENARIO_ID_COL]].reset_index(drop=True), preds], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("Wrote predictions:", output_csv)


if __name__ == "__main__":
    main()
