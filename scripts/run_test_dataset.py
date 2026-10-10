"""Reproduce the full paper test with integrity checks and optional exact masks."""
import argparse
import csv
import json
from pathlib import Path
import sys

from lidar_cloud.evaluation import verify
from lidar_cloud.weights import model_manifest, sha256_file

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('E1', 'E2', 'E3', 'E4', 'E5')


def validate_dataset(data, expected):
    data = Path(data).resolve()
    manifest_path = data/'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Missing {manifest_path}. Extract the complete TEST_DATASET download.')
    if sha256_file(manifest_path) != expected['manifest_sha256']:
        raise ValueError('Dataset manifest checksum differs from the released test set.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = manifest.get('files', [])
    names = [row['file'] for row in entries]
    if (manifest.get('role') != 'full_test' or manifest.get('n_files') != expected['expected_files']
            or len(names) != expected['expected_files'] or len(set(names)) != len(names)):
        raise ValueError('Expected the complete, uniquely indexed full test set; subsets cannot reproduce full-test accuracy.')
    if any(Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name
           or Path(name).suffix != '.h5' or not (data/name).resolve().is_relative_to(data)
           for name in names):
        raise ValueError('Manifest entries must be relative H5 paths inside TEST_DATASET.')
    actual = {path.relative_to(data).as_posix() for path in data.rglob('*.h5')}
    if actual != set(names):
        raise ValueError(f'Test file list differs: missing={sorted(set(names)-actual)}, extra={sorted(actual-set(names))}.')
    for row in entries:
        path = data/row['file']
        if path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
            raise ValueError(f'Dataset checksum/size mismatch: {row["file"]}.')
    return {'passed': True, 'files_checked': len(names), 'manifest_sha256': expected['manifest_sha256']}


def compare_masks(measured, reference):
    def indexed(rows):
        result = {}
        for row in rows:
            key = (row['method'], row['file'])
            if key in result:
                raise ValueError(f'Duplicate prediction reference: {key}.')
            result[key] = row['mask_sha256']
        return result
    actual, expected = indexed(measured), indexed(reference)
    if actual.keys() != expected.keys():
        raise ValueError('Prediction/reference file lists differ.')
    differences = [{'model': model, 'file': file} for (model, file) in sorted(expected)
                   if actual[(model, file)] != expected[(model, file)]]
    return {'comparisons': len(expected), 'exact': not differences, 'different_masks': differences,
            'encoding': 'SHA-256 of (T,Z) cloud boolean mask cast to uint8, C order'}


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT.parent/'TEST_DATASET', help='Extracted complete test directory.')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/verification/full_test')
    parser.add_argument('--device', default='cpu', help='cpu (default) or cuda.')
    parser.add_argument('--check-only', action='store_true', help='Check all 84 file checksums without inference.')
    parser.add_argument('--strict', action='store_true', help='Require all 420 prediction masks to match original code exactly.')
    args = parser.parse_args(argv)
    try:
        data, output = args.data.resolve(), args.output.resolve()
        if output == data or output.is_relative_to(data):
            raise ValueError('Output must be outside TEST_DATASET to keep the published observations unchanged.')
        expected = json.loads((ROOT/'results/test_dataset.json').read_text(encoding='utf-8'))
        integrity = validate_dataset(data, expected)
        from lidar_cloud.weights import RESOURCE_DIR
        if sha256_file(RESOURCE_DIR/'norm_stats.json') != expected['frozen_stats_sha256']:
            raise ValueError('Bundled station MAD statistics differ from the reference.')
        output.mkdir(parents=True, exist_ok=True)
        (output/'dataset_check.json').write_text(json.dumps(integrity, indent=2)+'\n', encoding='utf-8')
        print(f'Integrity PASS: {integrity["files_checked"]} H5 checksums verified.', flush=True)
        if args.check_only:
            return 0
        print('Running E1–E5: frozen station MAD, threshold 0.65, batch 1, paper window protocol.', flush=True)
        report = verify(data, output, models=METHODS, device=args.device, batch_size=1, protocol='paper')
        masks = compare_masks(read_csv(output/'pixel_metrics_by_file.csv'),
                              read_csv(ROOT/'results/full_test/reference_mask_hashes.csv'))
        references = json.loads((ROOT/'results/full_test/verification.json').read_text(encoding='utf-8'))
        original_counts = {r['method']: r for r in references['pixel_metrics']}
        counts_equal = all(all(row[k] == original_counts[row['method']][k] for k in ('tp', 'fp', 'fn', 'tn'))
                           for row in report['pixel_metrics'])
        exact = masks['exact'] and counts_equal
        passed = bool(report.get('passed')) and (exact or not args.strict)
        summary = {'passed': passed, 'strict': args.strict, 'device': args.device, 'integrity': integrity,
                   'exact_original_pixel_counts': counts_equal, 'original_mask_comparison': masks,
                   'max_abs_pixel_metric_difference': report['max_abs_metric_difference'],
                   'max_abs_boundary_mae_difference_m': report['max_abs_boundary_mae_difference_m'],
                   'pixel_metric_tolerance': report['reference_tolerance'],
                   'boundary_mae_tolerance_m': report['boundary_mae_tolerance_m'],
                   'models': {name: model_manifest()[name]['sha256'] for name in METHODS if name != 'E5'},
                   'protocol': 'paper', 'threshold': .65, 'batch_size': 1,
                   'frozen_stats_sha256': expected['frozen_stats_sha256']}
        (output/'reproduction.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
        for row in report['pixel_metrics']:
            print(f'{row["method"]}: IoU={row["iou"]:.6f}, F1={row["f1"]:.6f}, precision={row["precision"]:.6f}, recall={row["recall"]:.6f}')
        match = 'exact original masks/counts' if exact else 'masks differ; inspect metric differences in reproduction.json'
        print(f'{"PASS" if passed else "FAIL"}: {match}. Results: {output}')
        return 0 if passed else 1
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
