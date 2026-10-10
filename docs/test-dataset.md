# Full-test data and reproduction

The independent test contains **84 labeled site-days**, separate from the training data and algorithm repository. Its local publication folder is `../TEST_DATASET`; the ZIP and ZIP checksum are beside it. No database has been selected or upload performed. [Package checksums](../results/test_dataset.json).

## Download layout

After downloading `lidar-cloud-test-v1.zip`, verify its SHA-256 against the download metadata, then extract it into the repository's parent folder:

```bash
python -m zipfile -e /path/to/lidar-cloud-test-v1.zip ..
```

The archive creates this compact folder:

```text
TEST_DATASET/
  data/          84 files: <station_id>_<YYYYMMDD>.h5
  results/       pixel_metrics.csv, boundary_metrics.csv, metrics.png
  README.md      Usage, H5 fields and attribution
  manifest.json  Data-file dates, sizes and SHA-256
  SHA256SUMS     Package-file checksums
```

For example, `data/51628_20250106.h5` is Aksu on 6 January 2025 (UTC). The three repository examples use the same naming scheme and identical H5 bytes. The data-package results are concise summaries; detailed per-file metrics and 420 original mask hashes remain in this algorithm repository. H5 field descriptions are incorporated into the dataset's single README.

## Run from the repository

Install the package once, then run:

```bash
python -m pip install .
python scripts/run_test_dataset.py --device cuda --strict
```

For another location, add `--data /path/to/TEST_DATASET`. `--check-only` checks all 84 data files without inference. CPU is supported by omitting `--device cuda`. The script fixes the published weights, frozen station statistics, threshold 0.65, batch size 1 and historical paper window ordering; it does not estimate new MAD statistics from the test set.

By default, outputs are under `outputs/verification/full_test/`. `metrics.png` shows the measured scores; `verification.json` and CSV files provide numeric results; `reproduction.json` reports the integrity checks, metric differences and exact-mask comparisons. The data directory remains unchanged.

## What counts as a pass

**`--strict` requires all 420 mask hashes and all five global confusion matrices to match the original code exactly.** Missing/corrupt data, changed MAD statistics or changed predictions return a nonzero exit code. Hashes are computed from each `(T,Z)` binary cloud mask cast to uint8 in C order; they come from the original E1–E4 prediction caches and updated E5/VDE masks.

Without `--strict`, the script accepts the reference metric tolerance (maximum difference 0.002 for IoU/F1/precision/recall), while still explicitly reporting any mask differences. Boundary evaluation requires identical original matched-layer counts for E1/E3/E5 and MAE differences at most 0.1 m. The current same-device acceptance run has exact masks and zero pixel-metric differences; boundary MAE differences reach 0.023 m because coordinate-related floating-point matching ties can affect assignments. Different devices or scientific-library versions may change pixels close to the threshold.

Models are named exactly as the manuscript: E1, E2, E3, E4 and E5. A three-day example verification is a subset, not evidence of full-test reproduction.

## Inspect a day

```bash
python -m lidar_cloud predict --input ../TEST_DATASET/data/51628_20250106.h5 --output outputs/example_day
```

The output contains one H5, one layer-height CSV and one PNG. This deployment command segments time gaps; the full-test runner uses the reference paper protocol. Renaming the released files or changing their contents makes the integrity check fail.
