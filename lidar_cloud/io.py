"""Read existing CARLNET or minimal user H5 files without modifying them."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import json

import h5py
import numpy as np

CHANNELS = ("1064", "355p", "532p", "vdr_355", "vdr_532")


def station_elevation_m(station, attrs):
    """Explicit site elevation, or a verified reference-site value; else AGL."""
    elevation = attrs.get("station_elevation_m")
    if elevation is None:
        path = Path(__file__).resolve().parent / "resources/stations.json"
        reference = json.loads(path.read_text(encoding="utf-8"))
        elevation = reference.get(str(station), {}).get("elevation_m")
    if elevation is not None:
        elevation = float(elevation)
        if not np.isfinite(elevation):
            raise ValueError("station_elevation_m must be finite.")
    return elevation


@dataclass
class Observation:
    signals: dict
    height_km: np.ndarray
    time_utc: np.ndarray
    station_id: str
    attrs: dict
    truth: np.ndarray | None = None
    vde_signals: tuple | None = None


def input_files(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No H5 files found in {path}")
    return files


def read_observation(path, station=None, require_truth=False, require_vde=False):
    path = Path(path)
    with h5py.File(path, "r") as h:
        attrs = dict(h.attrs)
        station_id = str(station or attrs.get("station_id", attrs.get("station", path.stem.split("_")[0])))
        if isinstance(station or attrs.get("station_id", attrs.get("station")), bytes):
            station_id = (station or attrs.get("station_id", attrs.get("station"))).decode()
        height = h["coordinates/height_km"][:]
        if "coordinates/time_utc" in h:
            times = h["coordinates/time_utc"][:].astype(np.float64)
        elif "coordinates/time_hours" in h:
            date = str(attrs.get("date", ""))
            if not re.fullmatch(r"\d{8}", date):
                match = re.search(r"(?:^|_)(\d{8})(?:_|$)", path.stem)
                if not match:
                    raise ValueError("Legacy time_hours needs a YYYYMMDD date attribute or filename.")
                date = match.group(1)
            epoch = datetime.strptime(date, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()
            hours = h["coordinates/time_hours"][:].astype(float)
            times = epoch + np.unwrap(hours * (2 * np.pi / 24)) * (24 / (2 * np.pi)) * 3600
        else:
            raise KeyError("Provide coordinates/time_utc (UTC Unix seconds).")
        signals = {}
        for channel in CHANNELS:
            prefix = "depolarization" if channel.startswith("vdr_") else "preprocessed_signals"
            key = f"{prefix}/{channel}"
            if key not in h:
                raise KeyError(f"Required model channel missing: {key}")
            signals[channel] = h[key][:]
        shape = (len(times), len(height))
        if any(array.shape != shape for array in signals.values()):
            raise ValueError("All five channels must have shape (time, height), matching coordinates.")
        truth = h["manual_annotation/cloud_mask"][:] if "manual_annotation/cloud_mask" in h else None
        if require_truth and truth is None:
            raise KeyError("Verification requires manual_annotation/cloud_mask.")
        if truth is not None and truth.shape != shape:
            raise ValueError("Manual mask shape does not match coordinates.")
        totals = []
        for nm in (355, 532):
            total_key = f"preprocessed_signals/{nm}_total"
            cross_key = f"preprocessed_signals/{nm}s"
            pgr_key = f"pgr_{nm}"
            if total_key in h:
                total = h[total_key][:].astype(float)
            elif cross_key in h and pgr_key in attrs:
                total = signals[f"{nm}p"].astype(float) + h[cross_key][:].astype(float) * float(attrs[pgr_key])
            else:
                if require_vde:
                    raise KeyError(f"E5 needs {total_key}, or {cross_key} and attribute {pgr_key}.")
                break
            if total.shape != shape:
                raise ValueError("VDE elastic channels must match the input coordinates.")
            totals.append(total)
        vde_signals = (signals["1064"].astype(float), *totals) if len(totals) == 2 else None
    return Observation(signals, height, times, station_id, attrs, truth, vde_signals)


def write_prediction(path, observation, result, metadata):
    """Write a separate output; invalid mask pixels are 255, not clear sky."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["station_id"] = observation.station_id
        h.attrs["schema_version"] = "lidar-cloud-1"
        elevation = station_elevation_m(observation.station_id, observation.attrs)
        if elevation is not None:
            h.attrs["station_elevation_m"] = elevation
        for key, value in metadata.items():
            h.attrs[key] = value
        h.create_dataset("coordinates/time_utc", data=observation.time_utc)
        h["coordinates/time_utc"].attrs["units"] = "seconds since 1970-01-01 00:00:00 UTC"
        h.create_dataset("coordinates/height_km", data=observation.height_km)
        h["coordinates/height_km"].attrs["units"] = "km above ground, after range alignment"
        for name, array in [("cloud_probability", result.probability), ("cloud_mask", result.mask), ("valid_observation", result.valid.astype(np.uint8))]:
            ds = h.create_dataset(f"prediction/{name}", data=array, compression="gzip", shuffle=True)
            if name == "cloud_mask":
                ds.attrs["values"] = "0=clear, 1=cloud, 255=invalid or outside supported range"
