from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401

        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for the CNN pipeline but is not installed in this environment. "
            "Install it in LeakEnv (CPU is fine): `pip install torch`."
        ) from e


@dataclass(frozen=True)
class CnnModelConfig:
    num_sensors: int
    conv_filters: Sequence[int]
    kernel_size: int
    pool_size: int
    dropout_rates: Sequence[float]
    dense_units: Sequence[int]
    use_global_avg_pool: bool
    num_outputs: int


class CnnRegressor:  # thin wrapper to match RF-style save/load
    def __init__(self, model):
        self.model = model

    def save(self, out_dir: Path) -> Path:
        torch = _require_torch()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "cnn_baseline.pt"
        torch.save(self.model.state_dict(), path)
        return path

    @staticmethod
    def load(*, model, path: Path):
        torch = _require_torch()
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        return CnnRegressor(model)


def build_cnn_model(cfg: CnnModelConfig):
    torch = _require_torch()
    import torch.nn as nn

    if len(cfg.conv_filters) != 3:
        raise ValueError("This starter CNN expects exactly 3 conv blocks (filters length=3).")
    if len(cfg.dropout_rates) != 3:
        raise ValueError("This starter CNN expects exactly 3 dropout rates (length=3).")
    if len(cfg.dense_units) != 2:
        raise ValueError("This starter CNN expects exactly 2 dense layers (dense_units length=2).")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            c1, c2, c3 = list(map(int, cfg.conv_filters))
            d1, d2 = list(map(int, cfg.dense_units))
            k = int(cfg.kernel_size)
            p = int(cfg.pool_size)

            if k % 2 == 0:
                raise ValueError("kernel_size must be odd for this starter model (so padding=k//2 preserves length).")
            pad = k // 2

            self.conv1 = nn.Conv1d(cfg.num_sensors, c1, kernel_size=k, padding=pad)
            self.bn1 = nn.BatchNorm1d(c1)
            self.drop1 = nn.Dropout(float(cfg.dropout_rates[0]))

            self.conv2 = nn.Conv1d(c1, c2, kernel_size=k, padding=pad)
            self.bn2 = nn.BatchNorm1d(c2)
            self.drop2 = nn.Dropout(float(cfg.dropout_rates[1]))
            self.pool2 = nn.MaxPool1d(kernel_size=p)

            self.conv3 = nn.Conv1d(c2, c3, kernel_size=k, padding=pad)
            self.bn3 = nn.BatchNorm1d(c3)

            self.gap = nn.AdaptiveAvgPool1d(1)

            self.fc1 = nn.Linear(c3, d1)
            self.drop3 = nn.Dropout(float(cfg.dropout_rates[2]))
            self.fc2 = nn.Linear(d1, d2)

            self.head_x = nn.Linear(d2, 1)
            self.head_y = nn.Linear(d2, 1)
            self.head_burst = nn.Linear(d2, 1)

        def forward(self, x):
            # x: (batch, time, sensors) => convert to (batch, sensors, time)
            x = x.permute(0, 2, 1)

            x = self.conv1(x)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.drop1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = torch.relu(x)
            x = self.drop2(x)
            x = self.pool2(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = torch.relu(x)

            # Global average pool over time dimension
            x = self.gap(x).squeeze(-1)

            x = torch.relu(self.fc1(x))
            x = self.drop3(x)
            x = torch.relu(self.fc2(x))

            out_x = self.head_x(x)
            out_y = self.head_y(x)
            out_b = self.head_burst(x)

            return torch.cat([out_x, out_y, out_b], dim=1)

    return Net()
