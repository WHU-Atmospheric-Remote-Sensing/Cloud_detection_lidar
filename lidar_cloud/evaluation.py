"""Micro pixel metrics and one-to-one physical cloud-boundary matching."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .inference import CloudDetector, Prediction
from .io import input_files, read_observation
from .weights import sha256_file

COUNT_KEYS = ("tp", "fp", "fn", "tn")


def counts(gt, pred, valid=None):
    gt, pred = np.asarray(gt, dtype=bool), np.asarray(pred, dtype=bool)
    keep = np.ones(gt.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return {"tp": int(np.count_nonzero(keep & gt & pred)), "fp": int(np.count_nonzero(keep & ~gt & pred)),
            "fn": int(np.count_nonzero(keep & gt & ~pred)), "tn": int(np.count_nonzero(keep & ~gt & ~pred))}


def metrics(c):
    tp, fp, fn, tn = (c[k] for k in COUNT_KEYS)
    # Preserve the reference evaluator's zero-denominator convention.
    return {"iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "accuracy": (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0}


def extract_layers(mask, height, minimum_thickness_m=90.0):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    dz = float(np.median(np.diff(height))) * 1000
    result = []
    for group in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        base = float(height[group[0]]) * 1000
        top = float(height[group[-1]]) * 1000 + dz
        if top - base + 1e-6 >= minimum_thickness_m:
            result.append({"base_m": base, "top_m": top})
    return result


def physical_matches(gt, pred):
    if not gt or not pred:
        return {}
    cost = np.full((len(gt), len(pred)), 1e12)
    for i, a in enumerate(gt):
        for j, b in enumerate(pred):
            overlap = max(0.0, min(a["top_m"], b["top_m"]) - max(a["base_m"], b["base_m"]))
            gap = max(0.0, a["base_m"] - b["top_m"], b["base_m"] - a["top_m"])
            center = abs((a["base_m"] + a["top_m"] - b["base_m"] - b["top_m"]) / 2)
            if overlap > 0:
                union = max(a["top_m"], b["top_m"]) - min(a["base_m"], b["base_m"])
                cost[i, j] = (1 - overlap / union) * 1e6 + center
            elif gap <= 300:
                cost[i, j] = 2e6 + gap * 1000 + center
    rows, cols = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(rows, cols) if cost[i, j] < 1e11}


def boundary_statistics(points, manual_count):
    result = {"manual_layer_count": manual_count, "matched_layer_count": len(points),
              "detection_rate": len(points) / manual_count if manual_count else 0.0}
    for name in ("base", "top"):
        error = np.array([p[f"{name}_error_m"] for p in points])
        result[f"{name}_mae_m"] = float(np.mean(np.abs(error))) if len(error) else None
        result[f"{name}_bias_m"] = float(np.mean(error)) if len(error) else None
        result[f"{name}_rmse_m"] = float(np.sqrt(np.mean(error ** 2))) if len(error) else None
    return result


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_layers(path, result, observation):
    rows = []
    for i in range(len(result.mask)):
        for j, layer in enumerate(extract_layers(result.mask[i] == 1, observation.height_km)):
            rows.append({"station_id": observation.station_id, "profile": i, "time_utc": observation.time_utc[i],
                         "layer": j + 1, "cloud_base_km_agl": layer["base_m"] / 1000,
                         "detectable_cloud_top_km_agl": layer["top_m"] / 1000})
    if rows:
        write_csv(path, rows)
    else:
        Path(path).write_text("station_id,profile,time_utc,layer,cloud_base_km_agl,detectable_cloud_top_km_agl\n", encoding="utf-8")


def predict_vde(obs, reference_protocol=False):
    from .vde import generate_mask
    from .preprocessing import validate_coordinates
    validate_coordinates(obs.height_km, obs.time_utc)
    if obs.vde_signals is None:
        raise ValueError("E5 requires total 355/532 elastic signals or cross channels with PGR.")
    height = obs.height_km
    bg = (height >= 17) & (height <= 18)
    if np.count_nonzero(bg) < 10:
        raise ValueError("Paper E5 requires measured 17--18 km background data; zero padding cannot replace it.")
    if len(obs.time_utc) < 5:
        raise ValueError("Paper E5 requires at least five observed profiles for temporal support.")
    mask = generate_mask(*obs.vde_signals, height, obs.time_utc)
    valid = np.all(np.isfinite(np.stack(obs.vde_signals, axis=-1)), axis=-1)
    # Paper metrics retain every reference bin; operational outputs mark the
    # algorithm's unsupported height interval as invalid.
    if not reference_protocol:
        valid &= ((height >= 0.4) & (height <= 18.0))[None, :]
    probability = mask.astype(np.float32)
    probability[~valid] = np.nan
    mask = mask.astype(np.uint8)
    mask[~valid] = 255
    return Prediction(probability, mask, valid, ["E5 probability is a binary decision (0/1), not a calibrated neural probability."])


def verify(data, output, models=("E1", "E2", "E3", "E4", "E5"), stats=None, device="cpu", batch_size=1, protocol="paper"):
    """Recompute outputs. Paper protocol preserves the original file-order windows."""
    files = input_files(data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(data) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if manifest:
        expected = {row["file"]: row["sha256"] for row in manifest["files"]}
        if len(expected) != len(manifest["files"]) or any(
            Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name
            or not (Path(data)/name).resolve().is_relative_to(Path(data).resolve())
            for name in expected
        ):
            raise ValueError("Dataset manifest must contain unique relative paths inside its directory.")
        if {f.relative_to(data).as_posix() for f in files} != set(expected):
            raise ValueError("Dataset file list differs from its manifest.")
        for file in files:
            if sha256_file(file) != expected[file.relative_to(data).as_posix()]:
                raise ValueError(f"Dataset checksum mismatch: {file.name}")
    is_full = bool(manifest and manifest.get("role") == "full_test" and len(files) == 84)
    totals = {model: dict.fromkeys(COUNT_KEYS, 0) for model in models}
    detail, boundary_points = [], {model: [] for model in models}
    manual_count = {model: 0 for model in models}
    comparison_rows = []
    reference_check_dir = None
    # Optional private cache comparison used during release acceptance, never required by users.
    import os
    if os.environ.get("LIDAR_CLOUD_REFERENCE_CACHE"):
        reference_check_dir = Path(os.environ["LIDAR_CLOUD_REFERENCE_CACHE"])
    for model in models:
        detector = None if model == "E5" else CloudDetector(model, stats, device, batch_size=batch_size)
        for i, file in enumerate(files):
            obs = read_observation(file, require_truth=True, require_vde=model == "E5")
            if model == "E5":
                result = predict_vde(obs, reference_protocol=protocol == "paper")
            else:
                result = detector.predict(obs.signals, obs.height_km, obs.time_utc, obs.station_id,
                                          gap_seconds=1e12 if protocol == "paper" else None)
            pred = result.mask == 1
            c = counts(obs.truth, pred, result.valid)
            for key in COUNT_KEYS:
                totals[model][key] += c[key]
            detail.append({"method": model, "file": file.name, "station_id": obs.station_id, **c, **metrics(c),
                           "mask_sha256": hashlib.sha256(pred.astype(np.uint8).tobytes(order="C")).hexdigest()})
            for profile in range(len(pred)):
                gt_layers = extract_layers((obs.truth[profile] > 0) & result.valid[profile], obs.height_km)
                pred_layers = extract_layers(pred[profile], obs.height_km)
                manual_count[model] += len(gt_layers)
                for a, b in physical_matches(gt_layers, pred_layers).items():
                    boundary_points[model].append({"method": model, "file": file.name, "profile": profile,
                        "base_error_m": pred_layers[b]["base_m"] - gt_layers[a]["base_m"],
                        "top_error_m": pred_layers[b]["top_m"] - gt_layers[a]["top_m"]})
            if reference_check_dir:
                import h5py
                cached = (Path(os.environ["LIDAR_CLOUD_REFERENCE_VDE"]) / file.name if model == "E5" and os.environ.get("LIDAR_CLOUD_REFERENCE_VDE")
                          else reference_check_dir / ("masks" if model == "E5" else f"predictions/{model}") / file.name)
                if cached.exists():
                    with h5py.File(cached, "r") as h:
                        ref = h["quality_control/cloud_mask_vde" if model == "E5" else "prediction/cloud_mask"][:].astype(bool)
                    comparison_rows.append({"method": model, "file": file.name, "different_pixels": int(np.count_nonzero(pred != ref))})
            if (i + 1) % 10 == 0 or i == len(files) - 1:
                print(f"{model}: {i + 1}/{len(files)}", flush=True)
        del detector
    pixel_rows = [{"method": model, "n_files": len(files), **totals[model], **metrics(totals[model])} for model in models]
    boundary_rows = [{"method": model, **boundary_statistics(boundary_points[model], manual_count[model])} for model in models]
    write_csv(output / "pixel_metrics.csv", pixel_rows)
    write_csv(output / "pixel_metrics_by_file.csv", detail)
    write_csv(output / "boundary_metrics.csv", boundary_rows)
    write_csv(output / "boundary_points.csv", [p for points in boundary_points.values() for p in points])
    if comparison_rows:
        write_csv(output / "reference_mask_comparison.csv", comparison_rows)
    report = {"dataset_role": "full_test" if is_full else "subset", "n_files": len(files), "protocol": protocol,
              "threshold": 0.65, "aggregation": "global micro pixel counts", "invalid_pixels": "excluded",
              "boundary_min_thickness_m": 90, "matching_max_gap_m": 300, "pixel_metrics": pixel_rows,
              "boundary_metrics": boundary_rows}
    ref_path = Path(__file__).resolve().parent / "resources/reference_pixel_metrics.json"
    if is_full and protocol == "paper" and ref_path.exists():
        ref = {row["method"]: row for row in json.loads(ref_path.read_text(encoding="utf-8"))}
        errors = {row["method"]: max(abs(row[key] - float(ref[row["method"]][key])) for key in ("iou", "f1", "precision", "recall")) for row in pixel_rows}
        report["max_abs_metric_difference"] = errors
        report["reference_tolerance"] = 0.002
        report["passed"] = all(value <= 0.002 for value in errors.values())
        boundary_reference = json.loads((ref_path.parent / "reference_boundary_metrics.json").read_text(encoding="utf-8"))
        measured = {row["method"]: row for row in boundary_rows}
        boundary_errors = {
            row["method"]: max(abs(measured[row["method"]][key] - float(row[key])) for key in ("base_mae_m", "top_mae_m"))
            for row in boundary_reference if row["method"] in measured
        }
        match_counts_equal = all(measured[row["method"]]["matched_layer_count"] == int(row["matched_layer_count"])
                                 for row in boundary_reference if row["method"] in measured)
        report["max_abs_boundary_mae_difference_m"] = boundary_errors
        report["boundary_mae_tolerance_m"] = 0.1
        report["reference_matched_layer_counts_equal"] = match_counts_equal
        report["passed"] &= match_counts_equal and all(value <= 0.1 for value in boundary_errors.values())
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    from .visualization import plot_verification
    plot_verification(report, output / "metrics.png")
    return report
