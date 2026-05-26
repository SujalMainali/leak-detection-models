from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this script directly (adds repo root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn import config


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


def _plot_loss_curves(*, plt, history_csv: Path, out_dir: Path) -> None:
    if not history_csv.exists():
        return

    df = pd.read_csv(history_csv)
    if not {"epoch", "train_loss", "val_loss"}.issubset(df.columns):
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.2))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN (1D): Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=160)
    plt.close()


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
        plt.title(f"CNN (1D): Parity ({t})")
        plt.tight_layout()
        plt.savefig(parity_dir / f"parity__{t}.png", dpi=160)
        plt.close()

        # Residual vs true
        plt.figure(figsize=(6.2, 4.2))
        plt.scatter(yt, residual, s=10, alpha=0.6)
        plt.axhline(0.0, linestyle="--")
        plt.xlabel(f"True {t}")
        plt.ylabel("Residual (pred - true)")
        plt.title(f"CNN (1D): Residuals ({t})")
        plt.tight_layout()
        plt.savefig(residual_dir / f"residuals__{t}.png", dpi=160)
        plt.close()

        # Residual distribution
        plt.figure(figsize=(6.2, 4.0))
        plt.hist(residual[np.isfinite(residual)], bins=40)
        plt.xlabel("Residual (pred - true)")
        plt.ylabel("Count")
        plt.title(f"CNN (1D): Residual Distribution ({t})")
        plt.tight_layout()
        plt.savefig(hist_dir / f"residual_hist__{t}.png", dpi=160)
        plt.close()


def main() -> None:
    plt = _require_matplotlib()

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Loss curves
    _plot_loss_curves(
        plt=plt,
        history_csv=config.HISTORIES_DIR / "training_history.csv",
        out_dir=config.PLOTS_DIR / "loss_curves",
    )

    # Prediction-based plots
    pred_path = config.PREDICTIONS_DIR / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}. Run scripts/cnn/predict_cnn.py first.")

    y_true, y_pred, _ = _load_predictions(pred_path)
    _plot_parity_and_residuals(plt=plt, y_true=y_true, y_pred=y_pred, out_base=config.PLOTS_DIR)

    print("Wrote plots to:", config.PLOTS_DIR)


if __name__ == "__main__":
    main()
