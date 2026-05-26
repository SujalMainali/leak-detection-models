from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401

        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for the CNN-2D pipeline but is not installed in this environment. "
            "Install it in LeakEnv (CPU is fine): `pip install torch`."
        ) from e


@dataclass(frozen=True)
class Cnn2dModelConfig:
    in_channels: int
    conv_filters: Sequence[int]
    kernel_sizes: Sequence[tuple[int, int]]
    pool_sizes: Sequence[tuple[int, int]]
    dropout_rates: Sequence[float]
    dense_units: Sequence[int]
    use_global_avg_pool: bool
    use_batch_norm: bool
    num_outputs: int


class Cnn2dRegressor:
    def __init__(self, model):
        self.model = model

    def save(self, out_dir: Path) -> Path:
        torch = _require_torch()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "cnn2d_baseline.pt"
        torch.save(self.model.state_dict(), path)
        return path

    @staticmethod
    def load(*, model, path: Path):
        torch = _require_torch()
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        return Cnn2dRegressor(model)


def build_cnn2d_model(cfg: Cnn2dModelConfig):
    torch = _require_torch()
    import torch.nn as nn

    if len(cfg.conv_filters) != 3:
        raise ValueError("This starter 2D CNN expects exactly 3 conv blocks (filters length=3).")
    if len(cfg.kernel_sizes) != 3:
        raise ValueError("This starter 2D CNN expects exactly 3 kernel sizes (length=3).")
    if len(cfg.pool_sizes) != 2:
        raise ValueError("This starter 2D CNN expects exactly 2 pool sizes (length=2).")
    if len(cfg.dropout_rates) != 3:
        raise ValueError("This starter 2D CNN expects exactly 3 dropout rates (length=3).")
    if len(cfg.dense_units) != 2:
        raise ValueError("This starter 2D CNN expects exactly 2 dense layers (dense_units length=2).")

    def _pad_for_kernel(k: tuple[int, int]) -> tuple[int, int]:
        kh, kw = int(k[0]), int(k[1])
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError("Kernel sizes must be odd (so padding=k//2 preserves spatial dims).")
        return (kh // 2, kw // 2)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            c1, c2, c3 = list(map(int, cfg.conv_filters))
            k1, k2, k3 = list(cfg.kernel_sizes)
            p1, p2 = list(cfg.pool_sizes)
            d1, d2 = list(map(int, cfg.dense_units))

            pad1 = _pad_for_kernel(k1)
            pad2 = _pad_for_kernel(k2)
            pad3 = _pad_for_kernel(k3)

            self.conv1 = nn.Conv2d(cfg.in_channels, c1, kernel_size=k1, padding=pad1)
            self.bn1 = nn.BatchNorm2d(c1) if cfg.use_batch_norm else nn.Identity()
            self.pool1 = nn.MaxPool2d(kernel_size=p1)
            self.drop1 = nn.Dropout(float(cfg.dropout_rates[0]))

            self.conv2 = nn.Conv2d(c1, c2, kernel_size=k2, padding=pad2)
            self.bn2 = nn.BatchNorm2d(c2) if cfg.use_batch_norm else nn.Identity()
            self.pool2 = nn.MaxPool2d(kernel_size=p2)
            self.drop2 = nn.Dropout(float(cfg.dropout_rates[1]))

            self.conv3 = nn.Conv2d(c2, c3, kernel_size=k3, padding=pad3)
            self.bn3 = nn.BatchNorm2d(c3) if cfg.use_batch_norm else nn.Identity()

            self.gap = nn.AdaptiveAvgPool2d((1, 1))

            self.fc1 = nn.Linear(c3, d1)
            self.drop3 = nn.Dropout(float(cfg.dropout_rates[2]))
            self.fc2 = nn.Linear(d1, d2)

            self.head_x = nn.Linear(d2, 1)
            self.head_y = nn.Linear(d2, 1)
            self.head_burst = nn.Linear(d2, 1)

        def forward(self, x):
            # x expected: (batch, H=24, W=21, C=1)
            # convert to (batch, C, H, W)
            x = x.permute(0, 3, 1, 2)

            x = self.conv1(x)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.pool1(x)
            x = self.drop1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = torch.relu(x)
            x = self.pool2(x)
            x = self.drop2(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = torch.relu(x)

            x = self.gap(x).view(x.shape[0], -1)

            x = torch.relu(self.fc1(x))
            x = self.drop3(x)
            x = torch.relu(self.fc2(x))

            out_x = self.head_x(x)
            out_y = self.head_y(x)
            out_b = self.head_burst(x)
            return torch.cat([out_x, out_y, out_b], dim=1)

    return Net()
