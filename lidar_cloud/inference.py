"""Sliding-window inference for a single profile, short input or long sequence."""

from dataclasses import dataclass, field
import warnings

import numpy as np
import torch

from .models.compression import compress_z_axis, decompress_z_predictions
from .preprocessing import GRID, load_stats, prepare, station_stats, validate_coordinates
from .weights import load_model


@dataclass
class Prediction:
    probability: np.ndarray
    mask: np.ndarray
    valid: np.ndarray
    notes: list[str] = field(default_factory=list)


class CloudDetector:
    def __init__(self, model="E3", stats=None, device="cpu", threshold=0.65, batch_size=1, checkpoint=None, max_threads=4):
        if model.upper() not in {"E1", "E2", "E3", "E4"}:
            raise ValueError("Detection models are E1--E4. R1/R2 are initialization weights; use predict_vde for E5.")
        if not (0 < threshold < 1) or int(batch_size) < 1:
            raise ValueError("threshold must be in (0, 1) and batch_size must be positive.")
        self.name = model.upper()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu.")
        torch.set_num_threads(max(1, int(max_threads)))
        self.model = load_model(self.name, self.device, checkpoint)
        self.stats = load_stats(stats)
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)

    def _segment(self, inputs):
        n = len(inputs)
        if n < 10:
            inputs = np.concatenate([inputs, np.repeat(inputs[-1:], 10 - n, axis=0)], axis=0)
        total = np.zeros((len(inputs), 2048), dtype=np.float32)
        count = np.zeros(len(inputs), dtype=np.float32)
        starts = list(range(len(inputs) - 9))
        with torch.inference_mode():
            for offset in range(0, len(starts), self.batch_size):
                group = starts[offset:offset + self.batch_size]
                compressed = []
                for start in group:
                    # Compression is kept on CPU; all released weights use this exact map.
                    window = inputs[start:start + 10].transpose(0, 2, 1).copy()
                    tensor = torch.from_numpy(window).permute(0, 2, 1)
                    part, _, _, info = compress_z_axis(tensor, GRID)
                    compressed.append(part)
                logits = self.model(torch.stack(compressed).to(self.device), task="seg")
                for local, start in enumerate(group):
                    dense = decompress_z_predictions(logits[local], info, interpolate=True)
                    probability = torch.sigmoid(dense.squeeze(-1)).cpu().numpy()
                    total[start:start + 10] += probability
                    count[start:start + 10] += 1
        return (total / count[:, None])[:n]

    def predict(self, signals, height_km, time_utc, station_id, cadence_seconds=300, gap_seconds=None):
        """Return predictions on the user's grid; missing data have mask=255."""
        height, times = validate_coordinates(height_km, time_utc)
        if not np.isfinite(cadence_seconds) or cadence_seconds <= 0:
            raise ValueError("cadence_seconds must be finite and positive.")
        gap = float(gap_seconds) if gap_seconds is not None else 1.5 * cadence_seconds
        if not np.isfinite(gap) or gap <= 0:
            raise ValueError("gap_seconds must be finite and positive.")
        inputs, observed, aligned = prepare(signals, height, station_stats(self.stats, station_id))
        if len(inputs) != len(times):
            raise ValueError("Signal time dimension does not match time_utc.")
        notes = []
        if len(times) < 10:
            message = "Fewer than 10 observed profiles: repeated last-profile padding; single-profile inference is not recommended for operations."
            notes.append(message)
            warnings.warn(message, UserWarning, stacklevel=2)
        if cadence_seconds != 300 or (len(times) > 1 and not np.allclose(np.diff(times), 300, atol=1)):
            notes.append("Temporal sampling differs from the 5-minute reference protocol, or contains gaps.")
        breaks = np.flatnonzero(np.diff(times) > gap) + 1
        probability = np.empty((len(times), len(height)), dtype=np.float32)
        for indices in np.split(np.arange(len(times)), breaks):
            dense = self._segment(inputs[indices])
            if aligned:
                probability[indices] = dense[:, :len(height)]
            else:
                for local, index in enumerate(indices):
                    probability[index] = np.interp(height, GRID, dense[local], left=np.nan, right=np.nan)
        if len(breaks):
            notes.append(f"Processed {len(breaks) + 1} time segments independently.")
        valid = observed & np.isfinite(probability)
        probability[~valid] = np.nan
        mask = np.where(valid, probability > self.threshold, False).astype(np.uint8)
        mask[~valid] = 255
        return Prediction(probability, mask, valid, notes)
