# Reference results and validation

| Folder/file | Contents |
|---|---|
| [Full test](full_test/metrics.png) | 84-file E1?E5 [pixel scores](full_test/pixel_metrics.csv), [boundary scores](full_test/boundary_metrics.csv), [verification](full_test/verification.json) and [strict reproduction](full_test/reproduction.json) |
| [Example subset](examples/metrics.png) | Three-day E1?E5 [pixel scores](examples/pixel_metrics.csv), [boundary scores](examples/boundary_metrics.csv) and [verification](examples/verification.json) |
| [Visual examples](visualization/README.md) | Three executed E3 curtains below 10 km and the paper's annotation-efficiency figure |
| [Validation summary](validation.json) | Consolidated weight, probability, MAD, short-input, publication-data and installation checks |
| [Test-package metadata](test_dataset.json) | ZIP SHA-256, trusted data-manifest SHA-256 and frozen MAD checksum |

The full-test CUDA run used batch size 1, threshold 0.65 for E1?E4 and the paper file-order protocol. All **420 masks** and all five global confusion matrices equal the original code: **zero pixel-metric difference**. The [original mask hashes](full_test/reference_mask_hashes.csv) remain available for the [strict runner](../scripts/run_test_dataset.py). E5 uses fixed physical rules, not the neural probability threshold.

All six exported weights equal their original parameters. Twelve E1?E4 same-device probability comparisons have maximum error zero. The 51 inference/calibration/runner checks passed in source and installed environments. Different devices, batch sizes and libraries may affect predictions near the threshold.

Boundary evaluation uses 90 m minimum layers and one-to-one matching with a 300 m maximum nonoverlap gap. E1/E3/E5 matched-layer counts equal the archived reference. Their largest MAE difference is **0.023 m**, caused by coordinate-related floating-point assignment ties. The CSVs report executable rerun values; the README's published boundary values remain labeled separately. [Paper correspondence](../docs/paper-correspondence.md).

MAD was recomputed using eight days at each test station. Nine channel estimates match an independent sampling oracle exactly; all 45 preprocessing comparisons match the original code. The maximum sampling deviation from all finite positive values is 0.675%. Historical Changsha/Beijing JSON values were checked, but their original calibration periods were not rerun.

## Short inputs

[Aggregate comparisons](padding_summary.csv) cover nine days, three stations and 21 distinct clear/cloud windows (588 comparisons). The table reports mask disagreement with complete 10-profile/2048-gate context, **not an accuracy estimate**, using real observed pixels only.

| Input | Released padding | Mask disagreement |
|---|---|---:|
| 1 profile | Repeat last | 0.481% |
| 3 profiles | Repeat last | 0.272% |
| 5 profiles | Repeat last | 0.220% |
| 9 profiles | Repeat last | 0.048% |
| 512 height gates | Raw zero | 1.668% |
| 800 height gates | Raw zero | 0.076% |
| 1024 height gates | Raw zero | 0.025% |
| 1501 height gates | Raw zero | 0.000% |

Some reflection/raw-median alternatives performed better in individual scenarios. Padding cannot recover missing context; prefer at least ten real profiles and validate locally. Detailed historical audit tables and build environments are archived outside the user repository; fresh `verify` runs generate their own per-file diagnostics under `outputs/`.
