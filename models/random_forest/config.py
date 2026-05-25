from __future__ import annotations

from pathlib import Path

# -----------------
# Data locations
# -----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "leak_dataset_wide.csv"
PROCESSED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "leak_dataset_wide_baseline.csv"

SCENARIO_ID_COL = "scenario_id"

# Targets (baseline)
TARGET_COLUMNS = ["leak_x", "leak_y", "leak_size_lpm"]

# Sensor column detection (baseline)
SENSOR_NODE_PREFIXES = ["NODEADD", "NODE", "HOUSE_EPN"]
VALID_HOURS = range(0, 24)  # dataset uses Hour0..Hour23

# -----------------
# Splits (reproducible)
# -----------------
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "random_forest"
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.1
TEST_FRACTION = 0.2
RANDOM_STATE = 42

# -----------------
# Random Forest hyperparameters
# -----------------
N_ESTIMATORS = 500
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = "sqrt"
N_JOBS = -1

# -----------------
# Outputs
# -----------------
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models" / "random_forest"
METRICS_DIR = OUTPUTS_DIR / "metrics" / "random_forest"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions" / "random_forest"
PLOTS_DIR = OUTPUTS_DIR / "plots" / "random_forest"

# -----------------
# Script runtime toggles (so scripts run with no CLI args)
# -----------------
# If True, `scripts/random_forest/train_rf.py` will delete the processed CSV and rebuild it.
REBUILD_PROCESSED_ON_RUN = False

# If True, `scripts/random_forest/train_rf.py` will delete existing split CSVs and recreate them.
REBUILD_SPLITS_ON_RUN = False

# -----------------
# Default artifact paths (used by evaluate/predict scripts)
# -----------------
MODEL_BUNDLE_PATH = MODELS_DIR / "rf_baseline.joblib"
PREDICT_INPUT_CSV_PATH = PROCESSED_CSV_PATH
PREDICT_OUTPUT_CSV_PATH = PREDICTIONS_DIR / "predictions.csv"
