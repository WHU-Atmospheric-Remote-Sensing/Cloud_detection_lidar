"""Reproduction of the paper E5: mean-SNR3, run5, minimum 120 m.

The detection kernels are extracted without numerical changes from the original
validated implementation. No optical-retrieval or model prediction is required.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d, binary_opening

MIN_HEIGHT_KM = 0.4
INPUT_CHANNEL_NM = 1064
MAX_HEIGHT_KM = 18.0
BACKGROUND_RANGE_KM = (17.0, 18.0)
NOISE_K = 3.0
MIN_LAYER_DEPTH_KM = 0.045
T_MIN = 4.0
D_MAX = -4.0
LAYER_SNR_MIN = 3.0
LAYER_SNR_MIN_EXCLUSIVE = 3.0
LAYER_SNR_STATISTIC = "mean"
MIN_BLOB_PIXELS = 1
MINIMUM_BLOB_PIXELS = 1
MINIMUM_CONSECUTIVE_PROFILES = 5
MAX_TIME_GAP_SECONDS = 1200.0
MAX_VERTICAL_GAP_KM = 0.300
CANDIDATE_MIN_THICKNESS_KM = 0.045
OUTPUT_MIN_THICKNESS_KM = 0.120
ALGORITHM_VERSION = "zhao2014_sdp_vde_1064_mean_snr3_run5_min120m_v1"

@dataclass
class VDEProfileResult:
    cloud_mask: np.ndarray
    particle_mask: np.ndarray
    pn: np.ndarray
    baseline: np.ndarray
    noise_sd: float
    n_particle_layers: int
    n_cloud_layers: int
    layer_diagnostics: list[dict]

def moving_average_edge(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return x.astype(float, copy=True)
    return uniform_filter1d(np.asarray(x, float), size=window, mode='nearest')

def contiguous_groups(mask: np.ndarray) -> list[np.ndarray]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    return list(np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1))

def layer_depth_meets_minimum(n_bins: int, dz_km: float, minimum_km: float) -> bool:
    return n_bins > 0 and n_bins * dz_km + 1e-12 >= minimum_km

def zhao2014_profile(signal: np.ndarray, height_km: np.ndarray, min_layer_depth_km: float=MIN_LAYER_DEPTH_KM) -> VDEProfileResult:
    z_all = np.asarray(height_km, float)
    s_all = np.asarray(signal, float)
    out_cloud = np.zeros(z_all.size, bool)
    out_particle = np.zeros(z_all.size, bool)
    out_pn = np.full(z_all.size, np.nan)
    out_baseline = np.full(z_all.size, np.nan)
    work = (z_all >= MIN_HEIGHT_KM) & (z_all <= MAX_HEIGHT_KM) & np.isfinite(s_all)
    bg = (z_all >= BACKGROUND_RANGE_KM[0]) & (z_all <= BACKGROUND_RANGE_KM[1]) & np.isfinite(s_all)
    if work.sum() < 20 or bg.sum() < 10:
        return VDEProfileResult(out_cloud, out_particle, out_pn, out_baseline, np.nan, 0, 0, [])
    z = z_all[work]
    background = float(np.nanmean(s_all[bg]))
    noise_sd = float(np.nanstd(s_all[bg] - background, ddof=1))
    if not np.isfinite(noise_sd) or noise_sd <= 1e-12:
        return VDEProfileResult(out_cloud, out_particle, out_pn, out_baseline, noise_sd, 0, 0, [])
    p = s_all[work] - background
    dz = float(np.nanmedian(np.diff(z)))
    smooth_window = 3 if dz >= 0.03 - 1e-06 else max(1, int(round(0.06 / dz)))
    ps = moving_average_edge(p, smooth_window)
    noise_threshold = NOISE_K * noise_sd
    pd1 = ps.copy()
    for i in range(1, pd1.size):
        if abs(pd1[i] - pd1[i - 1]) < noise_threshold:
            pd1[i] = pd1[i - 1]
    pd2 = ps.copy()
    for i in range(pd2.size - 2, -1, -1):
        if abs(pd2[i] - pd2[i + 1]) < noise_threshold:
            pd2[i] = pd2[i + 1]
    pd_signal = 0.5 * (pd1 + pd2)
    order = np.argsort(pd_signal, kind='stable')
    rs = pd_signal[order]
    n = rs.size
    pe = np.arange(1, n + 1, dtype=float) / n
    for i in range(1, n):
        if rs[i] == rs[i - 1]:
            pe[i] = pe[i - 1]
    mi, ma = (float(rs[0]), float(rs[-1]))
    y = pe * (ma - mi) + mi
    pn = np.empty_like(y)
    pn[order] = y
    baseline = ma + (mi - ma) * (z - z[0]) / (z[-1] - z[0])
    particle = pn > baseline
    min_layer_depth_km = float(min_layer_depth_km)
    if not np.isfinite(min_layer_depth_km) or min_layer_depth_km <= 0:
        raise ValueError('min_layer_depth_km must be positive and finite')
    min_bins = max(1, int(np.ceil(min_layer_depth_km / dz - 1e-12)))
    cloud = np.zeros_like(particle)
    valid_particle = np.zeros_like(particle)
    n_particle = 0
    n_cloud = 0
    layer_diagnostics = []
    for group in contiguous_groups(particle):
        if group.size < min_bins:
            continue
        depth = z[group[-1]] - z[group[0]] + dz
        if not layer_depth_meets_minimum(group.size, dz, min_layer_depth_km):
            continue
        n_particle += 1
        valid_particle[group] = True
        positive_floor = max(noise_sd * 0.05, np.finfo(float).tiny)
        q = np.maximum(ps[group], positive_floor) * np.square(z[group])
        log_q = np.log(np.maximum(q, np.finfo(float).tiny))
        if group.size >= 3:
            f_slope = np.gradient(log_q, z[group])
        else:
            f_slope = np.diff(log_q) / np.diff(z[group])
        if not np.any(np.isfinite(f_slope)):
            continue
        t_value = float(np.nanmax(f_slope))
        d_value = float(np.nanmin(f_slope))
        t_threshold = 3.0 if z[group[0]] < 3.0 else 1.5
        classified_cloud = t_value > t_threshold or d_value < -7.0
        layer_diagnostics.append({'base_km': float(z[group[0]]), 'top_km': float(z[group[-1]] + dz), 'depth_km': float(depth), 'T': t_value, 'D': d_value, 't_threshold': t_threshold, 'classified_cloud': classified_cloud})
        if classified_cloud:
            cloud[group] = True
            n_cloud += 1
    out_cloud[work] = cloud
    out_particle[work] = valid_particle
    out_pn[work] = pn
    out_baseline[work] = baseline
    return VDEProfileResult(out_cloud, out_particle, out_pn, out_baseline, noise_sd, n_particle, n_cloud, layer_diagnostics)

def direct_elastic_snr(signals, height_km):
    primary = (height_km >= 25.0) & (height_km <= 30.0)
    fallback = (height_km >= 18.0) & (height_km <= 20.0)
    normalized = []
    for signal in signals:
        if not np.isfinite(np.nanmedian(signal)) or np.nanmedian(signal) <= 1e-06:
            continue
        with np.errstate(invalid='ignore'):
            noise = np.sqrt(np.nanmean(np.square(signal[:, primary]), axis=1))
            fallback_noise = np.sqrt(np.nanmean(np.square(signal[:, fallback]), axis=1))
        bad = ~np.isfinite(noise) | (noise <= 1e-12)
        noise[bad] = fallback_noise[bad]
        noise[~np.isfinite(noise) | (noise <= 1e-12)] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized.append(signal / noise[:, None])
    if not normalized:
        raise ValueError('Invalid VDE signal or coordinate shape; temporal QC needs at least two profiles.')
    with np.errstate(invalid='ignore'):
        return np.nanmean(np.stack(normalized, axis=0), axis=0)

def layers(mask, height_km, snr_profile=None):
    indices = np.flatnonzero(mask)
    if not indices.size:
        return []
    dz = float(np.nanmedian(np.diff(height_km)))
    result = []
    for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        values = np.asarray(snr_profile[group], float) if snr_profile is not None else np.array([], float)
        values = values[np.isfinite(values)]
        result.append({'first': int(group[0]), 'last': int(group[-1]), 'base_km': float(height_km[group[0]]), 'top_km': float(height_km[group[-1]] + dz), 'cloud_mean_snr': float(np.mean(values)) if values.size else np.nan})
    return result

def _profile_layers(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [(int(group[0]), int(group[-1])) for group in groups]

def filter_min_blob_pixels(mask: np.ndarray, height_km: np.ndarray, unix_time: np.ndarray, minimum_pixels: int=MIN_BLOB_PIXELS) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    height_km = np.asarray(height_km, dtype=float)
    unix_time = np.asarray(unix_time, dtype=float)
    if mask.ndim != 2 or mask.shape[1] != height_km.size or mask.shape[0] != unix_time.size:
        raise ValueError('minimum-blob filter coordinate shape mismatch')
    if minimum_pixels <= 0:
        raise ValueError('minimum_pixels must be positive')
    dz = float(np.nanmedian(np.diff(height_km)))
    gap_bins = max(1, int(np.ceil(MAX_VERTICAL_GAP_KM / dz)))
    nodes: list[tuple[int, int, int]] = []
    by_profile: list[list[int]] = [[] for _ in range(mask.shape[0])]
    for profile in range(mask.shape[0]):
        for first, last in _profile_layers(mask[profile]):
            node_index = len(nodes)
            nodes.append((profile, first, last))
            by_profile[profile].append(node_index)
    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = (find(a), find(b))
        if ra != rb:
            parent[rb] = ra
    for profile in range(mask.shape[0] - 1):
        dt = float(unix_time[profile + 1] - unix_time[profile])
        if not np.isfinite(dt) or dt < 0 or dt > MAX_TIME_GAP_SECONDS + 1e-06:
            continue
        for left in by_profile[profile]:
            _, a0, a1 = nodes[left]
            for right in by_profile[profile + 1]:
                _, b0, b1 = nodes[right]
                vertical_gap = max(0, b0 - a1 - 1, a0 - b1 - 1)
                if vertical_gap <= gap_bins:
                    union(left, right)
    component_pixels: dict[int, int] = {}
    for node_index, (_, first, last) in enumerate(nodes):
        root = find(node_index)
        component_pixels[root] = component_pixels.get(root, 0) + last - first + 1
    output = np.zeros_like(mask, dtype=bool)
    for node_index, (profile, first, last) in enumerate(nodes):
        if component_pixels[find(node_index)] >= minimum_pixels:
            output[profile, first:last + 1] = True
    return output.astype(np.uint8)

def filter_min_profile_run(mask: np.ndarray, height_km: np.ndarray, unix_time: np.ndarray, minimum_profiles: int, max_time_gap_seconds: float=MAX_TIME_GAP_SECONDS, max_vertical_gap_km: float=MAX_VERTICAL_GAP_KM) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    height_km = np.asarray(height_km, dtype=float)
    unix_time = np.asarray(unix_time, dtype=float)
    if mask.ndim != 2 or mask.shape != (unix_time.size, height_km.size):
        raise ValueError('profile-run filter coordinate shape mismatch')
    if minimum_profiles <= 0:
        raise ValueError('minimum_profiles must be positive')
    dz = float(np.nanmedian(np.diff(height_km)))
    radius = max(1, int(np.ceil(max_vertical_gap_km / dz - 1e-12)))
    expanded = maximum_filter1d(mask.astype(np.uint8), size=2 * radius + 1, axis=-1, mode='constant') > 0
    opened = np.zeros_like(mask, dtype=bool)
    breaks = np.flatnonzero(~np.isfinite(np.diff(unix_time)) | (np.diff(unix_time) < 0) | (np.diff(unix_time) > max_time_gap_seconds + 1e-06)) + 1
    for segment in np.split(np.arange(mask.shape[0]), breaks):
        if segment.size < minimum_profiles:
            continue
        opened[segment] = binary_opening(expanded[segment], structure=np.ones((minimum_profiles, 1), dtype=bool))
    return (mask & opened).astype(np.uint8)

def filter_min_vertical_layer_thickness(mask: np.ndarray, height_km: np.ndarray, minimum_thickness_km: float) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    height_km = np.asarray(height_km, dtype=float)
    if mask.ndim != 2 or mask.shape[1] != height_km.size:
        raise ValueError('vertical-thickness filter coordinate shape mismatch')
    dz = float(np.nanmedian(np.diff(height_km)))
    if not np.isfinite(minimum_thickness_km) or minimum_thickness_km <= 0:
        raise ValueError('minimum_thickness_km must be positive and finite')
    output = np.zeros_like(mask, dtype=bool)
    for profile in range(mask.shape[0]):
        for first, last in _profile_layers(mask[profile]):
            thickness = (last - first + 1) * dz
            if thickness + 1e-12 >= minimum_thickness_km:
                output[profile, first:last + 1] = True
    return output.astype(np.uint8)

def corrected_vde_mask(signal1064: np.ndarray, height_km: np.ndarray, elastic_snr: np.ndarray, unix_time: np.ndarray, layer_snr_min: float | None=None, layer_snr_statistic: str | None=None, minimum_blob_pixels: int=MIN_BLOB_PIXELS, candidate_min_thickness_km: float | None=None) -> np.ndarray:
    signal1064 = np.asarray(signal1064, dtype=float)
    height_km = np.asarray(height_km, dtype=float)
    elastic_snr = np.asarray(elastic_snr, dtype=float)
    unix_time = np.asarray(unix_time, dtype=float)
    if signal1064.ndim != 2 or signal1064.shape != elastic_snr.shape:
        raise ValueError('Invalid VDE signal or coordinate shape; temporal QC needs at least two profiles.')
    if signal1064.shape[1] != height_km.size:
        raise ValueError('Invalid VDE signal or coordinate shape; temporal QC needs at least two profiles.')
    if signal1064.shape[0] != unix_time.size:
        raise ValueError('Invalid VDE signal or coordinate shape; temporal QC needs at least two profiles.')
    if signal1064.shape[0] < 2:
        raise ValueError('Invalid VDE signal or coordinate shape; temporal QC needs at least two profiles.')
    if layer_snr_min is None:
        layer_snr_min = LAYER_SNR_MIN
    if layer_snr_statistic is None:
        layer_snr_statistic = LAYER_SNR_STATISTIC
    if layer_snr_statistic not in {'mean', 'median'}:
        raise ValueError("layer_snr_statistic must be 'mean' or 'median'")
    preliminary = np.zeros_like(signal1064, dtype=bool)
    for i in range(signal1064.shape[0]):
        if candidate_min_thickness_km is None:
            result = zhao2014_profile(signal1064[i], height_km)
        else:
            result = zhao2014_profile(signal1064[i], height_km, min_layer_depth_km=candidate_min_thickness_km)
        for layer in result.layer_diagnostics:
            if not (layer['T'] > T_MIN and layer['D'] < D_MAX):
                continue
            first = int(np.searchsorted(height_km, layer['base_km'], side='left'))
            last = int(np.searchsorted(height_km, layer['top_km'], side='left')) - 1
            if last < first:
                continue
            values = elastic_snr[i, first:last + 1]
            values = values[np.isfinite(values)]
            if values.size:
                statistic = float(np.mean(values) if layer_snr_statistic == 'mean' else np.median(values))
            else:
                statistic = np.nan
            if np.isfinite(statistic) and statistic > layer_snr_min:
                preliminary[i, first:last + 1] = True
    output = np.zeros_like(preliminary)
    dz = float(np.nanmedian(np.diff(height_km)))
    vertical_bins = max(1, int(np.ceil(MAX_VERTICAL_GAP_KM / dz)))
    n_time, n_height = preliminary.shape
    for i in range(n_time):
        for layer in layers(preliminary[i], height_km):
            first, last = (int(layer['first']), int(layer['last']))
            lo = max(0, first - vertical_bins - 1)
            hi = min(n_height, last + vertical_bins + 2)
            supported = False
            for j in (i - 1, i + 1):
                if not 0 <= j < n_time:
                    continue
                dt = abs(float(unix_time[j] - unix_time[i]))
                if np.isfinite(dt) and dt <= MAX_TIME_GAP_SECONDS:
                    if np.any(preliminary[j, lo:hi]):
                        supported = True
                        break
            if supported:
                output[i, first:last + 1] = True
    return filter_min_blob_pixels(output, height_km, unix_time, minimum_pixels=minimum_blob_pixels)

def generate_mask(s1064: np.ndarray, s355: np.ndarray, s532: np.ndarray, height_km: np.ndarray, unix_seconds: np.ndarray, output_min_thickness_km: float | None=OUTPUT_MIN_THICKNESS_KM) -> np.ndarray:
    elastic_snr = direct_elastic_snr([s355, s532, s1064], height_km)
    candidate = corrected_vde_mask(s1064, height_km, elastic_snr, unix_seconds, layer_snr_min=LAYER_SNR_MIN_EXCLUSIVE, layer_snr_statistic=LAYER_SNR_STATISTIC, minimum_blob_pixels=MINIMUM_BLOB_PIXELS, candidate_min_thickness_km=CANDIDATE_MIN_THICKNESS_KM)
    run5 = filter_min_profile_run(candidate, height_km, unix_seconds,
                                  minimum_profiles=MINIMUM_CONSECUTIVE_PROFILES,
                                  max_time_gap_seconds=MAX_TIME_GAP_SECONDS,
                                  max_vertical_gap_km=MAX_VERTICAL_GAP_KM).astype(np.uint8)
    if output_min_thickness_km is None:
        return run5
    return filter_min_vertical_layer_thickness(run5, height_km,
                                               minimum_thickness_km=output_min_thickness_km).astype(np.uint8)

def mask_attributes() -> dict[str, object]:
    return {'algorithm_version': ALGORITHM_VERSION, 'input_channel_nm': INPUT_CHANNEL_NM, 'T_min_exclusive': 4.0, 'D_max_exclusive': -4.0, 'candidate_min_thickness_m': CANDIDATE_MIN_THICKNESS_KM * 1000.0, 'output_min_layer_thickness_m': OUTPUT_MIN_THICKNESS_KM * 1000.0, 'layer_elastic_snr_statistic': LAYER_SNR_STATISTIC, 'layer_elastic_snr_min_exclusive': LAYER_SNR_MIN_EXCLUSIVE, 'elastic_snr_channels': '355_total,532_total,1064; valid-channel mean', 'elastic_snr_noise': 'per-profile RMS background 25-30 km; fallback 18-20 km', 'minimum_consecutive_profiles': MINIMUM_CONSECUTIVE_PROFILES, 'max_time_gap_min': MAX_TIME_GAP_SECONDS / 60.0, 'max_vertical_gap_km': MAX_VERTICAL_GAP_KM, 'minimum_time_height_blob_pixels': MINIMUM_BLOB_PIXELS, 'additional_statistical_thickness_filter': 'False'}
