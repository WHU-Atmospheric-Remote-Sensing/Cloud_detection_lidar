"""Deterministic, station-specific Median/MAD calibration from user H5 data."""

import json
from datetime import datetime, timezone
from pathlib import Path
import warnings

import numpy as np

from .io import CHANNELS, input_files, read_observation
from .preprocessing import validate_coordinates
from .weights import sha256_file


def compute_stats(path, station, output, max_samples=1_000_000, seed=20260923, instrument_id=None):
    if int(max_samples) < 100:
        raise ValueError("max_samples must be at least 100.")
    rng = np.random.default_rng(seed)
    samples = {ch: np.empty(0) for ch in CHANNELS[:3]}
    priorities = {ch: np.empty(0) for ch in CHANNELS[:3]}
    counts = {ch: 0 for ch in CHANNELS[:3]}
    files, dates, native_ids, instrument_ids, cadences = [], set(), set(), set(), set()
    for file in input_files(path):
        obs = read_observation(file)
        validate_coordinates(obs.height_km, obs.time_utc)
        native_ids.add(obs.station_id)
        if obs.attrs.get("instrument_id") is not None:
            identifier = obs.attrs["instrument_id"]
            instrument_ids.add(identifier.decode() if isinstance(identifier, bytes) else str(identifier))
        dates.update(datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y%m%d") for t in obs.time_utc)
        if len(obs.time_utc) > 1:
            cadence = float(obs.attrs["time_avg_minutes"]) * 60 if "time_avg_minutes" in obs.attrs else float(np.median(np.diff(obs.time_utc)))
            cadences.add(round(cadence, 6))
        files.append({"file": file.name, "sha256": sha256_file(file)})
        for channel in CHANNELS[:3]:
            data = np.asarray(obs.signals[channel], dtype=float)
            values = data[np.isfinite(data) & (data > 0.0)]
            counts[channel] += int(values.size)
            # Uniform priority reservoir: deterministic and bounded by one file + cap.
            ranks = rng.random(values.size)
            merged = np.concatenate([samples[channel], values])
            merged_ranks = np.concatenate([priorities[channel], ranks])
            if len(merged) > max_samples:
                chosen = np.argpartition(merged_ranks, max_samples - 1)[:max_samples]
                merged, merged_ranks = merged[chosen], merged_ranks[chosen]
            samples[channel], priorities[channel] = merged, merged_ranks
    if len(native_ids) > 1:
        raise ValueError(f"Calibration mixes stations/instruments: {sorted(native_ids)}. Supply one instrument per run.")
    if len(instrument_ids) > 1 or instrument_id and instrument_ids and instrument_ids != {instrument_id}:
        raise ValueError(f"Calibration mixes instruments or conflicts with instrument-id: {sorted(instrument_ids)}.")
    if len(cadences) > 1:
        raise ValueError(f"Calibration mixes temporal averaging intervals: {sorted(cadences)} seconds. Use the deployment interval consistently.")
    stats = {}
    for ch, values in samples.items():
        if len(values) < 100:
            raise ValueError(f"Not enough finite positive observations for {ch}: {len(values)} (minimum 100).")
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if not np.isfinite(mad) or mad <= 0:
            raise ValueError(f"Invalid MAD for {ch}: {mad}. Check channel data and calibration period.")
        stats[ch] = {"median": median, "mad": mad, "K": 4.0}
    trial = len(dates) < 8
    if trial:
        warnings.warn("Calibration spans fewer than 8 days; treat it as a trial. Include day/night conditions.", stacklevel=2)
    payload = {station: stats, "provenance": {
        "station_id": station, "instrument_id": instrument_id or next(iter(instrument_ids), station),
        "source_instrument_ids": sorted(instrument_ids),
        "source_station_ids": sorted(native_ids), "dates": sorted(dates), "files": files,
        "cadence_seconds": sorted(cadences), "seed": seed, "max_samples_per_channel": max_samples,
        "sample_counts": {ch: len(v) for ch, v in samples.items()}, "eligible_counts": counts,
        "sampling": "uniform priority reservoir of finite positive observation-level elastic signals",
        "processing_stage": "instrument-preprocessed; before MAD, range-squared, log transform and height padding",
        "trial": trial,
    }}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
