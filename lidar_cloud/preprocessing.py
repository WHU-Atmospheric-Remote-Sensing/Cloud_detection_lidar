"""The frozen preprocessing order used by the reference detector."""

import json
from pathlib import Path

import numpy as np

from .io import CHANNELS
from .weights import RESOURCE_DIR

GRID = np.arange(2048, dtype=np.float32) * np.float32(0.015)


def load_stats(stats=None):
    if isinstance(stats, dict):
        return stats
    path = Path(stats) if stats is not None else RESOURCE_DIR / "norm_stats.json"
    return json.loads(path.read_text(encoding="utf-8"))


def station_stats(stats, station):
    if station not in stats:
        raise KeyError(f"MAD statistics missing for station/instrument '{station}'. Run compute-stats first.")
    values = stats[station]
    for channel in CHANNELS[:3]:
        if channel not in values:
            raise KeyError(f"MAD statistics missing for {station}/{channel}.")
        s = values[channel]
        if not np.isfinite(s.get("median", np.nan)) or not np.isfinite(s.get("mad", np.nan)) or s["mad"] <= 0:
            raise ValueError(f"Invalid median/MAD for {station}/{channel}.")
        if not np.isfinite(s.get("K", 4)) or s.get("K", 4) <= 0:
            raise ValueError(f"Invalid K for {station}/{channel}.")
    return values


def validate_coordinates(height, times):
    height = np.asarray(height, dtype=float)
    times = np.asarray(times, dtype=float)
    if height.ndim != 1 or len(height) < 2 or not np.all(np.isfinite(height)) or np.any(np.diff(height) <= 0):
        raise ValueError("height_km must contain at least two finite, strictly increasing heights.")
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("time_utc must be nonempty, finite and strictly increasing.")
    return height, times


def prepare(signals, height, values):
    """Raw zero padding -> MAD -> range squared -> log1p -> compression later."""
    if isinstance(signals, dict):
        missing = set(CHANNELS) - set(signals)
        if missing:
            raise KeyError(f"Required channels missing: {sorted(missing)}")
        raw = np.stack([signals[name] for name in CHANNELS], axis=-1)
    else:
        raw = np.asarray(signals)
    if raw.ndim != 3 or raw.shape[1:] != (len(height), 5) or raw.shape[0] == 0:
        raise ValueError(f"Expected raw signals (T, {len(height)}, 5), got {raw.shape}.")
    # VDR is often undefined where elastic returns are weak. The reference
    # protocol replaces these ratios by zero; it does not discard those pixels.
    observed = np.all(np.isfinite(raw[:, :, :3]), axis=-1)
    raw = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    aligned = len(height) <= len(GRID) and np.allclose(height, GRID[:len(height)], rtol=0, atol=3e-6)
    if aligned:
        work = np.zeros((raw.shape[0], len(GRID), 5), dtype=np.float32)
        work[:, :len(height)] = raw
    else:
        work = np.empty((raw.shape[0], len(GRID), 5), dtype=np.float32)
        for t in range(raw.shape[0]):
            for c in range(5):
                work[t, :, c] = np.interp(GRID, height, raw[t, :, c], left=0, right=0)
    # Preserve the reference (T,C,Z) memory layout. NumPy log1p and block
    # averages can otherwise choose different SIMD reductions on interleaved
    # channels, which is enough to shift a few pixels near the threshold.
    work = work.transpose(0, 2, 1).copy()
    for c, channel in enumerate(CHANNELS):
        if c < 3:
            s = values[channel]
            work[:, c, :] = 0.5 * ((work[:, c, :] - s["median"]) / (s.get("K", 4.0) * s["mad"] + 1e-8) + 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                work[:, c, :] = np.log1p(work[:, c, :] * GRID[None, :] ** 2)
        else:
            work[:, c, :] = np.clip(work[:, c, :], 0, 1)
    work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    observed &= (height >= 0)[None, :] & (height <= float(GRID[-1]) + 3e-6)[None, :]
    return work.transpose(0, 2, 1), observed, aligned
