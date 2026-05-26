from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this script directly (adds repo root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.random_forest import config
from models.random_forest.rf_model import TrainedRandomForestBaseline


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting. Install it in LeakEnv: `pip install matplotlib`.") from e


def _load_predictions(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(path, low_memory=False)
    target_columns = list(config.TARGET_COLUMNS)
    true_cols = [f"true_{t}" for t in target_columns]
    pred_cols = [f"pred_{t}" for t in target_columns]

    missing = [c for c in [config.SCENARIO_ID_COL, *true_cols, *pred_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")

    y_true = df.loc[:, true_cols].copy()
    y_true.columns = target_columns
    y_pred = df.loc[:, pred_cols].copy()
    y_pred.columns = target_columns
    return y_true, y_pred, target_columns


def _plot_parity_and_residuals(*, plt, y_true: pd.DataFrame, y_pred: pd.DataFrame, out_base: Path) -> None:
    target_columns = list(y_true.columns)

    parity_dir = out_base / "parity_plots"
    residual_dir = out_base / "residual_plots"
    hist_dir = out_base / "error_distributions"
    for d in [parity_dir, residual_dir, hist_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for t in target_columns:
        yt = y_true[t].to_numpy(dtype=float)
        yp = y_pred[t].to_numpy(dtype=float)
        residual = yp - yt

        lo = float(np.nanmin([np.nanmin(yt), np.nanmin(yp)]))
        hi = float(np.nanmax([np.nanmax(yt), np.nanmax(yp)]))

        # Parity
        plt.figure(figsize=(5.6, 5.6))
        plt.scatter(yt, yp, s=10, alpha=0.6)
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel(f"True {t}")
        plt.ylabel(f"Pred {t}")
        plt.title(f"Random Forest: Parity ({t})")
        plt.tight_layout()
        plt.savefig(parity_dir / f"parity__{t}.png", dpi=160)
        plt.close()

        # Residual vs true
        plt.figure(figsize=(6.2, 4.2))
        plt.scatter(yt, residual, s=10, alpha=0.6)
        plt.axhline(0.0, linestyle="--")
        plt.xlabel(f"True {t}")
        plt.ylabel("Residual (pred - true)")
        plt.title(f"Random Forest: Residuals ({t})")
        plt.tight_layout()
        plt.savefig(residual_dir / f"residuals__{t}.png", dpi=160)
        plt.close()

        # Residual distribution
        plt.figure(figsize=(6.2, 4.0))
        plt.hist(residual[np.isfinite(residual)], bins=40)
        plt.xlabel("Residual (pred - true)")
        plt.ylabel("Count")
        plt.title(f"Random Forest: Residual Distribution ({t})")
        plt.tight_layout()
        plt.savefig(hist_dir / f"residual_hist__{t}.png", dpi=160)
        plt.close()


def _plot_feature_importance(*, plt, model: TrainedRandomForestBaseline, out_base: Path, top_k: int = 25) -> None:
    out_dir = out_base / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    features = list(model.feature_columns)
    for target in model.target_columns:
        rf = model.models[target]
        if not hasattr(rf, "feature_importances_"):
            continue

        importances = np.asarray(rf.feature_importances_, dtype=float)
        if importances.shape[0] != len(features):
            continue

        idx = np.argsort(importances)[::-1][: int(top_k)]
        top_features = [features[i] for i in idx]
        top_importances = importances[idx]

        plt.figure(figsize=(10.5, 6.0))
        y = np.arange(len(top_features))
        plt.barh(y, top_importances[::-1])
        plt.yticks(y, list(reversed(top_features)))
        plt.xlabel("Feature importance")
        plt.title(f"Random Forest: Feature Importance (top {len(top_features)}) — {target}")
        plt.tight_layout()
        plt.savefig(out_dir / f"feature_importance__{target}.png", dpi=160)
        plt.close()


def main() -> None:
    plt = _require_matplotlib()

    model_path = Path(config.MODEL_BUNDLE_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    tag = model_path.parent.name
    predictions_dir = config.OUTPUTS_RUN_DIR / "predictions" / "random_forest" / tag
    plots_dir = config.OUTPUTS_RUN_DIR / "plots" / "random_forest" / tag
    plots_dir.mkdir(parents=True, exist_ok=True)

    pred_path = predictions_dir / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}. Run scripts/random_forest/predict_rf.py first.")

    y_true, y_pred, _ = _load_predictions(pred_path)
    _plot_parity_and_residuals(plt=plt, y_true=y_true, y_pred=y_pred, out_base=plots_dir)

    model = TrainedRandomForestBaseline.load(model_path)
    _plot_feature_importance(plt=plt, model=model, out_base=plots_dir)

    print("Wrote plots to:", plots_dir)


if __name__ == "__main__":
    main()
