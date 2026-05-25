from __future__ import annotations

from pathlib import Path

# -----------------
# Paths
# -----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "leak_dataset_wide.csv"

# Processed datasets (shared across runs)
RF_RAW_FEATURES_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "rf_raw_features.csv"
RF_ENGINEERED_FEATURES_FS1_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "rf_engineered_features_fs1.csv"
RF_ENGINEERED_FEATURES_FS2_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "rf_engineered_features_fs2.csv"

SCENARIO_ID_COL = "scenario_id"

# Targets
TARGET_COLUMNS = ["leak_x", "leak_y", "leak_size_lpm"]

# Leak-window columns (only needed when leak-window summary features are enabled)
LEAK_START_COL = "leak_start_hr"
LEAK_DURATION_COL = "leak_duration_hr"

# Sensor column detection
SENSOR_NODE_PREFIXES = ["NODEADD", "NODE", "HOUSE_EPN"]
VALID_HOURS = range(0, 24)  # dataset uses Hour0..Hour23

# -----------------
# Splits (shared across runs)
# -----------------
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "random_forest"
SPLIT_IDS_DIR = SPLITS_DIR / "shared_ids"  # shared train/val/test IDs for fair comparisons

TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.1
TEST_FRACTION = 0.2
RANDOM_STATE = 42

# -----------------
# Outputs (per-run)
# -----------------
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Increment this integer when you want a fresh set of output artifacts.
# Processed data under data/processed and split IDs under data/splits remain shared/reused.
RUN_NUM = 1
OUTPUTS_RUN_DIR = OUTPUTS_DIR / f"run_{int(RUN_NUM)}"

# -----------------
# Script runtime toggles
# -----------------
REBUILD_PROCESSED_ON_RUN = False
REBUILD_SHARED_SPLIT_IDS_ON_RUN = False

# -----------------
# Feature engineering flags (engineered Random Forest)
# -----------------
USE_RAW_PRESSURE = True
USE_FIRST_DIFFERENCES = True
USE_LEAK_WINDOW_SUMMARY = False

# -----------------
# Normalization
# -----------------
NORMALIZATION_METHOD = "standard"  # z-score via sklearn StandardScaler
NORMALIZE_FEATURES = True
NORMALIZE_TARGETS = True

# -----------------
# Random Forest hyperparameters
# -----------------
N_ESTIMATORS = 1000
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = "sqrt"
N_JOBS = -1


def engineered_feature_set_tag() -> str:
    return "feature_set_2" if USE_LEAK_WINDOW_SUMMARY else "feature_set_1"


def engineered_processed_csv_path() -> Path:
    return RF_ENGINEERED_FEATURES_FS2_CSV_PATH if USE_LEAK_WINDOW_SUMMARY else RF_ENGINEERED_FEATURES_FS1_CSV_PATH


# -----------------
# Default artifact paths (used by evaluate/predict scripts)
# -----------------
MODEL_BUNDLE_PATH = OUTPUTS_RUN_DIR / "models" / "random_forest" / "raw_baseline" / "rf_baseline.joblib"
PREDICT_INPUT_CSV_PATH = RF_RAW_FEATURES_CSV_PATH
PREDICT_OUTPUT_CSV_PATH = OUTPUTS_RUN_DIR / "predictions" / "random_forest" / "raw_baseline" / "predictions.csv"
