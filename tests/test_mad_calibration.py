"""Independent calibration oracles, including the capped multi-file path."""
from pathlib import Path
import h5py
import numpy as np
import pytest
from lidar_cloud.io import CHANNELS
from lidar_cloud.statistics import compute_stats
from lidar_cloud.preprocessing import GRID, prepare


def observation(path, day=0, instrument='receiver-A', cadence=300):
    rng = np.random.default_rng(day)
    arrays = {}
    with h5py.File(path, 'w') as h:
        h.attrs.update(station_id='SITE', instrument_id=instrument, time_avg_minutes=cadence/60)
        h.create_dataset('coordinates/time_utc', data=1735689600 + day*86400 + np.arange(5)*cadence)
        h.create_dataset('coordinates/height_km', data=GRID[:160])
        for c in CHANNELS:
            raw = rng.lognormal(size=(5, 160))
            raw.flat[:5] = [0, -1, np.nan, np.inf, -np.inf]
            arrays[c] = raw
            group = 'depolarization' if c.startswith('vdr_') else 'preprocessed_signals'
            h.create_dataset(f'{group}/{c}', data=raw)
    return arrays


@pytest.mark.parametrize('cap', [101, 10000])
def test_multi_day_stats_against_independent_full_sort(cap, tmp_path):
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    all_values = {c: [] for c in CHANNELS[:3]}
    all_ranks = {c: [] for c in CHANNELS[:3]}
    rng = np.random.default_rng(123)
    for day in range(8):
        arrays = observation(inputs/f'{day:02}.h5', day)
        for c in CHANNELS[:3]:
            raw = arrays[c]
            values = raw[np.isfinite(raw) & (raw > 0)]
            all_values[c].extend(values)
            all_ranks[c].extend(rng.random(len(values)))
    a = compute_stats(inputs, 'SITE', tmp_path/'a.json', cap, 123)
    b = compute_stats(inputs, 'SITE', tmp_path/'b.json', cap, 123)
    assert a == b and not a['provenance']['trial']
    assert a['provenance']['instrument_id'] == 'receiver-A'
    for c in CHANNELS[:3]:
        selected = np.argsort(all_ranks[c])[:cap]
        sample = np.asarray(all_values[c])[selected]
        median = np.median(sample)
        assert a['SITE'][c] == {'median': float(median), 'mad': float(np.median(np.abs(sample-median))), 'K': 4.0}
        assert a['provenance']['eligible_counts'][c] == 8*795
        assert a['provenance']['sample_counts'][c] == min(cap, 8*795)


@pytest.mark.parametrize('error', ['instruments', 'cadence', 'coordinates', 'zero_mad', 'insufficient'])
def test_reject_unsafe_calibration(error, tmp_path):
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    observation(inputs/'a.h5')
    observation(inputs/'b.h5', 1, instrument='receiver-B' if error == 'instruments' else 'receiver-A',
                cadence=600 if error == 'cadence' else 300)
    if error in {'coordinates', 'zero_mad', 'insufficient'}:
        for file in inputs.glob('*.h5'):
            with h5py.File(file, 'r+') as h:
                if error == 'coordinates':
                    h['coordinates/time_utc'][1] = h['coordinates/time_utc'][0]
                else:
                    h['preprocessed_signals/1064'][:] = 1 if error == 'zero_mad' else 0
    with pytest.raises(ValueError):
        compute_stats(inputs, 'SITE', tmp_path/'out.json')
    assert not (tmp_path/'out.json').exists()


def test_raw_padding_and_unclipped_normalization_against_formula():
    raw = np.full((2, 100, 5), 50, dtype=np.float32)
    raw[:, :, 3:] = [-1, 2]
    values = {c: {'median': 2, 'mad': 1, 'K': 4} for c in CHANNELS[:3]}
    work, valid, aligned = prepare(raw, GRID[:100], values)
    full_raw = np.zeros((2, 2048), dtype=np.float32)
    full_raw[:, :100] = 50
    expected = np.log1p(.5*((full_raw-2)/(4+1e-8)+1)*GRID[None, :]**2)
    for index in range(3):
        np.testing.assert_array_equal(work[:, :, index], expected)
    assert work[0, 99, 0] > np.log1p(GRID[99]**2)  # elastic values are not clipped
    assert work[0, 200, 0] > 0  # raw-zero padding does not become feature-zero
    assert np.all(work[:, :, 3] == 0) and np.all(work[:, :100, 4] == 1)
    assert valid.shape == (2, 100) and valid.all() and aligned
