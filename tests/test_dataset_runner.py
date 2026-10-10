"""Fail before inference for corrupt/incomplete data, and detect mask changes."""
import importlib.util
import json
from pathlib import Path
import pytest
from lidar_cloud.weights import sha256_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dataset_runner', ROOT/'scripts/run_test_dataset.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def dataset(path, role='full_test'):
    path.mkdir()
    files = []
    for name in ['a.h5', 'b.h5']:
        file = path/name
        file.write_bytes(name.encode())
        files.append({'file': name, 'bytes': file.stat().st_size, 'sha256': sha256_file(file)})
    manifest = path/'manifest.json'
    manifest.write_text(json.dumps({'role': role, 'n_files': 2, 'files': files}), encoding='utf-8')
    return {'expected_files': 2, 'manifest_sha256': sha256_file(manifest)}


def test_runner_integrity_accepts_complete_data(tmp_path):
    data = tmp_path/'data'
    expected = dataset(data)
    assert runner.validate_dataset(data, expected)['files_checked'] == 2


def test_runner_accepts_published_data_subdirectory(tmp_path):
    data = tmp_path/'TEST_DATASET'
    expected = dataset(data)
    (data/'data').mkdir()
    manifest = json.loads((data/'manifest.json').read_text())
    for row in manifest['files']:
        (data/row['file']).rename(data/'data'/row['file'])
        row['file'] = 'data/'+row['file']
    (data/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    expected['manifest_sha256'] = sha256_file(data/'manifest.json')
    assert runner.validate_dataset(data, expected)['files_checked'] == 2


@pytest.mark.parametrize('unsafe', ['../a.h5', '/a.h5', 'C:/a.h5', 'data/../a.h5', 'data\\a.h5'])
def test_runner_rejects_unsafe_manifest_paths(unsafe, tmp_path):
    data = tmp_path/'TEST_DATASET'
    expected = dataset(data)
    manifest = json.loads((data/'manifest.json').read_text())
    manifest['files'][0]['file'] = unsafe
    (data/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    expected['manifest_sha256'] = sha256_file(data/'manifest.json')
    with pytest.raises(ValueError, match='relative H5 paths'):
        runner.validate_dataset(data, expected)


@pytest.mark.parametrize('damage', ['corrupt', 'missing', 'extra', 'manifest', 'subset'])
def test_runner_rejects_changed_dataset(damage, tmp_path):
    data = tmp_path/'data'
    expected = dataset(data, role='subset' if damage == 'subset' else 'full_test')
    if damage == 'corrupt':
        (data/'a.h5').write_bytes(b'xxxx')  # same length, different content
    elif damage == 'missing':
        (data/'a.h5').unlink()
    elif damage == 'extra':
        (data/'c.h5').write_bytes(b'extra')
    elif damage == 'manifest':
        with (data/'manifest.json').open('a', encoding='utf-8') as stream:
            stream.write(' ')
    with pytest.raises(ValueError):
        runner.validate_dataset(data, expected)


def test_original_mask_comparison_detects_changed_mask():
    original = [{'method': 'E1', 'file': 'a.h5', 'mask_sha256': 'aaa'},
                {'method': 'E2', 'file': 'a.h5', 'mask_sha256': 'bbb'}]
    assert runner.compare_masks(original, original)['exact']
    altered = [dict(row) for row in original]
    altered[1]['mask_sha256'] = 'ccc'
    result = runner.compare_masks(altered, original)
    assert not result['exact'] and result['different_masks'] == [{'model': 'E2', 'file': 'a.h5'}]
    with pytest.raises(ValueError, match='file lists differ'):
        runner.compare_masks(altered[:1], original)
    with pytest.raises(ValueError, match='Duplicate'):
        runner.compare_masks([original[0], original[0]], original)


@pytest.mark.parametrize('strict,exit_code', [(True, 1), (False, 0)])
def test_runner_exit_code_requires_exact_masks_in_strict_mode(strict, exit_code, tmp_path, monkeypatch):
    from lidar_cloud.evaluation import write_csv
    from lidar_cloud.weights import RESOURCE_DIR
    data, repository, output = tmp_path/'data', tmp_path/'repo', tmp_path/'output'
    expected = dataset(data)
    expected['frozen_stats_sha256'] = sha256_file(RESOURCE_DIR/'norm_stats.json')
    reference_dir = repository/'results/full_test'
    reference_dir.mkdir(parents=True)
    (repository/'results/test_dataset.json').write_text(json.dumps(expected), encoding='utf-8')
    row = {'method': 'E1', 'tp': 1, 'fp': 0, 'fn': 0, 'tn': 1, 'iou': 1., 'f1': 1., 'precision': 1., 'recall': 1.}
    report = {'passed': True, 'pixel_metrics': [row], 'max_abs_metric_difference': {'E1': 0.},
              'max_abs_boundary_mae_difference_m': {'E1': 0.}, 'reference_tolerance': .002, 'boundary_mae_tolerance_m': .1}
    (reference_dir/'verification.json').write_text(json.dumps(report), encoding='utf-8')
    reference = [{'method': 'E1', 'file': name, 'mask_sha256': 'original'} for name in ['a.h5', 'b.h5']]
    write_csv(reference_dir/'reference_mask_hashes.csv', reference)
    def inference_stub(data, output, **kwargs):
        altered = [dict(item) for item in reference]
        altered[1]['mask_sha256'] = 'changed-pixels-with-identical-confusion-counts'
        write_csv(output/'pixel_metrics_by_file.csv', altered)
        return report
    monkeypatch.setattr(runner, 'ROOT', repository)
    monkeypatch.setattr(runner, 'verify', inference_stub)
    command = ['--data', str(data), '--output', str(output)] + (['--strict'] if strict else [])
    assert runner.main(command) == exit_code
    summary = json.loads((output/'reproduction.json').read_text())
    assert summary['passed'] is not strict and summary['exact_original_pixel_counts']
    assert not summary['original_mask_comparison']['exact']
