import json
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import pytest

from lidar_cloud import CloudDetector, load_backbone, load_model
from lidar_cloud.evaluation import counts, extract_layers, metrics, physical_matches
from lidar_cloud.evaluation import predict_vde
from lidar_cloud.io import CHANNELS, read_observation
from lidar_cloud.preprocessing import GRID, station_stats
from lidar_cloud.statistics import compute_stats

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def obs():
    return read_observation(next((ROOT / "data/examples").glob("53845*.h5")))


@pytest.fixture(scope="module")
def detector():
    return CloudDetector(device="cpu")


@pytest.mark.parametrize("n", [1, 3, 9, 10, 11, 20])
def test_real_profile_counts(n, detector, obs):
    with pytest.warns(UserWarning) if n < 10 else nullcontext():
        out = detector.predict({c: x[:n] for c, x in obs.signals.items()}, obs.height_km, obs.time_utc[:n], obs.station_id)
    assert out.probability.shape == (n, 2048)
    assert np.all(np.isfinite(out.probability))
    assert set(np.unique(out.mask)).issubset({0, 1})


@pytest.mark.parametrize("z", [512, 800, 1024, 1501])
def test_short_and_nondivisible_heights(z, detector, obs):
    out = detector.predict({c: x[:10, :z] for c, x in obs.signals.items()}, obs.height_km[:z], obs.time_utc[:10], obs.station_id)
    assert out.mask.shape == (10, z) and out.valid.all()


def test_regrid_and_outside_range(detector, obs):
    heights = np.arange(2200) * 0.03
    raw = np.stack([np.array([np.interp(heights, obs.height_km, row, left=0, right=0) for row in obs.signals[c][:10]]) for c in CHANNELS], axis=-1)
    out = detector.predict(raw, heights, obs.time_utc[:10], obs.station_id)
    assert out.mask.shape == (10, 2200)
    assert np.all(out.mask[:, heights > GRID[-1] + 3e-6] == 255)
    assert np.all(np.isnan(out.probability[:, heights > GRID[-1] + 3e-6]))


def test_time_gap_cannot_mix_context(detector, obs):
    signals = {c: x[:20] for c, x in obs.signals.items()}
    times = obs.time_utc[:20].copy()
    times[10:] += 3600
    combined = detector.predict(signals, obs.height_km, times, obs.station_id)
    left = detector.predict({c: x[:10] for c, x in signals.items()}, obs.height_km, times[:10], obs.station_id)
    right = detector.predict({c: x[10:] for c, x in signals.items()}, obs.height_km, times[10:], obs.station_id)
    np.testing.assert_array_equal(combined.probability, np.concatenate([left.probability, right.probability]))


def test_missing_and_nonfinite(detector, obs):
    signals = {c: x[:10].copy() for c, x in obs.signals.items()}
    signals['1064'][2, 20] = np.nan
    signals['355p'][3, 21] = np.inf
    out = detector.predict(signals, obs.height_km, obs.time_utc[:10], obs.station_id)
    assert not out.valid[2, 20] and out.mask[2, 20] == 255
    assert not out.valid[3, 21] and out.mask[3, 21] == 255
    del signals['vdr_355']
    with pytest.raises(KeyError, match='Required channels'):
        detector.predict(signals, obs.height_km, obs.time_utc[:10], obs.station_id)
    with pytest.raises(KeyError, match='MAD statistics missing'):
        detector.predict(obs.signals, obs.height_km, obs.time_utc, 'UNKNOWN')
    with pytest.raises(ValueError, match='Invalid median/MAD'):
        station_stats({'BAD': {c: {'median': 1, 'mad': 0} for c in CHANNELS[:3]}}, 'BAD')


def test_hungarian_is_one_to_one():
    gt = [{'base_m': 1000, 'top_m': 1600}, {'base_m': 1800, 'top_m': 2100}]
    pred = [{'base_m': 1100, 'top_m': 2050}]
    matches = physical_matches(gt, pred)
    assert len(matches) == 1 and len(set(matches.values())) == 1
    assert physical_matches(gt, [{'base_m': 3000, 'top_m': 3500}]) == {}
    layers = extract_layers(np.array([0, 1, 1, 1, 1, 1, 1, 0]), np.arange(8) * .015)
    assert len(layers) == 1 and layers[0]['top_m'] == pytest.approx(105)


def test_micro_aggregation_is_not_macro():
    a = counts(np.array([1]), np.array([1]))
    b = counts(np.zeros(99), np.ones(99))
    summed = {k: a[k] + b[k] for k in a}
    assert metrics(summed)['precision'] == pytest.approx(.01)


def test_stats_repeatable_and_matches_direct_median(tmp_path, obs):
    file = tmp_path / 'station.h5'
    with h5py.File(file, 'w') as h:
        h.attrs['station_id'] = 'CUSTOM'
        h.attrs['date'] = '20250101'
        h.create_dataset('coordinates/time_utc', data=obs.time_utc[:3])
        h.create_dataset('coordinates/height_km', data=obs.height_km[:150])
        for c in CHANNELS:
            prefix = 'depolarization' if c.startswith('vdr_') else 'preprocessed_signals'
            h.create_dataset(f'{prefix}/{c}', data=obs.signals[c][:3, :150])
    with pytest.warns(UserWarning):
        a = compute_stats(file, 'CUSTOM', tmp_path/'a.json')
    with pytest.warns(UserWarning):
        b = compute_stats(file, 'CUSTOM', tmp_path/'b.json')
    assert a == b
    values = obs.signals['1064'][:3, :150]
    values = values[np.isfinite(values) & (values > 0)]
    assert a['CUSTOM']['1064']['median'] == float(np.median(values))
    assert a['CUSTOM']['1064']['mad'] == float(np.median(np.abs(values - np.median(values))))


def test_pretrained_weights_are_reusable():
    r1 = load_model('R1')
    assert r1.recon_head is not None
    backbone = load_backbone('R2')
    assert backbone.out_ch == 48


@pytest.mark.parametrize('name,checkpoint,digest', [
    ('E1', 'checkpoints_stage12_wo_stage3_include/stage3_epoch_80.pth',
     '6e2e1544c5068c263c9814ac45469829db35fe0b90a094455af1e55352a3c2c9'),
    ('E2', 'checkpoints_stage12_wo_stage3_exclude/stage3_epoch_70.pth',
     '400b93f0cb0f67c9df542384e5644e920eab66e073d4b9ce82ce0d03d91b7374'),
])
def test_detector_numbering_matches_manuscript(name, checkpoint, digest):
    from lidar_cloud.weights import RESOURCE_DIR, model_manifest, sha256_file
    metadata = model_manifest()[name]
    assert metadata['paper_id'] == name
    assert metadata['source_checkpoint'] == checkpoint
    assert metadata['file'] == f'weights/{name}.pth'
    assert metadata['sha256'] == sha256_file(RESOURCE_DIR/metadata['file']) == digest


def test_vde_operational_limits(obs):
    full = read_observation(next((ROOT / "data/examples").glob("53845*.h5")), require_vde=True)
    full.time_utc = full.time_utc[:10]
    full.signals = {c: x[:10] for c, x in full.signals.items()}
    full.vde_signals = tuple(x[:10] for x in full.vde_signals)
    result = predict_vde(full)
    outside = (full.height_km < .4) | (full.height_km > 18)
    assert np.all(result.mask[:, outside] == 255)
    assert np.all(np.isnan(result.probability[:, outside]))
    paper = predict_vde(full, reference_protocol=True)
    assert np.all(paper.mask[:, outside] == 0)


def test_vde_run_support_and_minimum_thickness():
    from lidar_cloud.vde import filter_min_profile_run, filter_min_vertical_layer_thickness
    height = np.arange(100) * .015
    times = np.arange(8) * 300.0
    candidate = np.zeros((8, 100), dtype=bool)
    candidate[:4, 30:40] = True
    assert not filter_min_profile_run(candidate, height, times, 5).any()
    candidate[:5, 30:40] = True
    assert filter_min_profile_run(candidate, height, times, 5).sum() == 50
    thin = np.zeros_like(candidate)
    thin[:, 30:37] = True
    assert not filter_min_vertical_layer_thickness(thin, height, .120).any()
    thin[:, 30:38] = True
    assert filter_min_vertical_layer_thickness(thin, height, .120).sum() == 64


def test_plot_elevation_is_explicit_and_never_guessed():
    from lidar_cloud.io import station_elevation_m
    assert station_elevation_m('51628', {}) == 1107.1
    assert station_elevation_m('UNKNOWN', {}) is None
    assert station_elevation_m('51628', {'station_elevation_m': 230}) == 230
    with pytest.raises(ValueError, match='must be finite'):
        station_elevation_m('UNKNOWN', {'station_elevation_m': np.nan})


def test_paper_plot_one_visible_gate_does_not_crop_predictions(tmp_path):
    from lidar_cloud.inference import Prediction
    from lidar_cloud.io import Observation
    from lidar_cloud.visualization import plot_prediction
    height = np.array([9.995, 10.010])
    times = 1735689600 + np.arange(3) * 300
    mask = np.ones((3, 2), dtype=np.uint8)
    result = Prediction(mask.astype(np.float32), mask.copy(), np.ones_like(mask, dtype=bool))
    observation = Observation({'1064': np.ones((3, 2))}, height, times, 'UNKNOWN', {})
    plot_prediction(observation, result, tmp_path / 'one_visible_gate.png')
    np.testing.assert_array_equal(result.mask, mask)
    assert result.probability.shape == (3, 2)


def test_plot_time_cells_leave_gaps_blank():
    from lidar_cloud.visualization import _time_cells
    times = np.array([1735689600, 1735689900, 1735693500], dtype=float)
    centers, segments = _time_cells(times, 300, 450)
    assert len(segments) == 2
    assert segments[0][1][-1] < segments[1][1][0]
    assert (segments[1][1][-1] - segments[1][1][0]) * 86400 == pytest.approx(300, abs=1e-5)
    assert segments[1][1][0] < centers[-1] < segments[1][1][-1]


@pytest.mark.parametrize('style', ['paper', 'overview'])
def test_every_time_height_panel_stops_at_ten_km(style, tmp_path, monkeypatch):
    from PIL import Image
    from lidar_cloud import visualization
    from lidar_cloud.io import Observation
    from lidar_cloud.inference import Prediction
    made = []
    create = visualization._figure
    def capture(size):
        figure = create(size)
        made.append(figure)
        return figure
    monkeypatch.setattr(visualization, '_figure', capture)
    height = np.arange(1200)*.015  # measured cloud extends above the display ceiling
    times = 1735689600 + np.arange(10)*300
    mask = np.ones((10, 1200), dtype=np.uint8)
    obs = Observation({'1064': np.ones_like(mask, dtype=float)}, height, times, '58847', {})
    result = Prediction(mask.astype(np.float32), mask.copy(), mask.astype(bool))
    output = visualization.plot_prediction(obs, result, tmp_path/f'{style}.png', {'model': 'E3'}, style=style)
    panels = [axis for axis in made[0].axes if axis.get_ylabel().startswith(('Altitude', 'Height'))]
    assert panels and all(axis.get_ylim()[1] == 10 for axis in panels)
    with Image.open(output) as png:
        assert float(png.info['HeightCeilingKm']) == 10 and png.info['PlotStyle'] == style
    np.testing.assert_array_equal(result.mask, mask)


@pytest.mark.parametrize("n", [1, 10])
def test_saved_prediction_plot_handles_slice_and_rejects_wrong_time(n, tmp_path, obs, detector):
    from lidar_cloud.io import Observation, write_prediction
    from lidar_cloud.visualization import plot_saved_prediction
    from matplotlib.image import imread
    original = next((ROOT / "data/examples").glob("53845*.h5"))
    start = 20
    subset = Observation({c: x[start:start+n] for c, x in obs.signals.items()}, obs.height_km,
                         obs.time_utc[start:start+n], obs.station_id, obs.attrs, obs.truth[start:start+n])
    with pytest.warns(UserWarning) if n < 10 else nullcontext():
        result = detector.predict(subset.signals, subset.height_km, subset.time_utc, subset.station_id)
    file = tmp_path / "prediction.h5"
    write_prediction(file, subset, result, {"model": "E3", "threshold": .65, "source_file": original.name})
    png = plot_saved_prediction(file, tmp_path / "prediction.png", original)
    pixels = imread(png)
    assert pixels.ndim == 3 and pixels.shape[0] > 500 and pixels.shape[1] > 500
    with h5py.File(file, 'r+') as h:
        h['coordinates/time_utc'][:] += 1
    with pytest.raises(ValueError, match="timestamps do not match"):
        plot_saved_prediction(file, tmp_path / "wrong.png", original)
