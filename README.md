# Cloud-layer detection from ground-based lidar

User release of **A Self-Supervised and Distillation-Guided Framework for Robust Cloud-Layer Identification in Multi-Channel Raman Lidar Networks**. Five lidar channels produce cloud probabilities, masks, cloud bases and **detectable cloud tops**. Default detector: **E3**, threshold **0.65**.

## Install and run

Python 3.10+. Run from this repository; CPU is supported. Add `--device cuda` to use a GPU.

```bash
python -m pip install .
python -m lidar_cloud predict --input data/examples --output outputs/examples
```

Each day produces **one PNG, one layer-height CSV and one prediction H5**, named `<station_id>_<YYYYMMDD>_<model>`. Input files are unchanged. The three labeled examples cover Aksu, Yan'an and Fuzhou; see the [data guide](data/README.md).

![Executed E3 example: Fuzhou 2025-01-04](results/visualization/58847_20250104_E3.png)

Plots show the 1064 nm signal up to **10 km**, with solid model contours and dotted manual labels. Saved predictions retain the full observed height range. [All example figures](results/visualization/README.md).

For 10 profiles, select one H5 with `--profiles 0:10`; omit the option for the whole sequence. Single profiles are supported but not recommended. [Input/output guide](docs/input-format.md) / [Python example](examples/python_usage.py).

## Verify results

```bash
python -m lidar_cloud verify --data data/examples --output outputs/verification
```

Open `outputs/verification/metrics.png`. The accompanying CSV/JSON files report E1-E5 scores for the **three-day subset**, not the full test.

Place the separate 84-file `TEST_DATASET` beside this repository, then run:

```bash
python scripts/run_test_dataset.py --device cuda --strict
```

Strict mode checks the data and all **420 prediction masks** against the original implementation. Outputs are in `outputs/verification/full_test/`; open `metrics.png` and `reproduction.json`. Omit `--device cuda` for CPU; use `--check-only` for integrity only. [Full-test instructions](docs/test-dataset.md) / [Archive checksum](results/test_dataset.json).

## Deploy at your station

### Non-2048 range gates and other resolutions

**E1-E4 accept non-2048 inputs, including 512, 1024 and 1501 gates, and resolutions such as 30 m. Use the same prediction command; no manual padding or gate-count option is needed.**

Store the five channels as `(T,Z)` arrays and provide `coordinates/height_km` with exactly `Z` finite, strictly increasing values. Use the **actual range-aligned height above ground in km**, including the instrument's starting range. With a zero starting range, 1024 gates at 15 m cover about 15.3 km, while 1024 gates at 30 m cover about 30.7 km. See the [minimum H5 layout](docs/input-format.md).

| Input grid | Automatic handling |
|---|---|
| A shorter prefix of the 15 m reference grid | Pad raw signals with zero before normalization |
| Different spacing, starting range or gate count | Interpolate raw signals onto the internal 15 m / 2048-gate grid |
| Outside the model's 0-30.705 km range | Return NaN probability and mask 255 (invalid) |

Predictions return on **your original grid**, with shape `(T,Z)`. Do not pad transformed features or include synthetic values in MAD calibration. Truncated heights and interpolation can affect cloud boundaries; validate a locally labeled subset. Plots stop at 10 km while numerical outputs retain valid observed heights. E5/VDE additionally requires real 17-18 km background observations, which padding cannot replace.

### Estimate and freeze station MAD

Prepare the five channels after instrument corrections and temporal averaging, **before MAD, range-squared correction or logarithms**. Use at least eight representative calibration days, including day and night, from your own station/instrument:

```bash
python -m lidar_cloud compute-stats --input calibration_h5 --station MY-LIDAR --instrument-id receiver-01 --output outputs/station_stats/my_station.json
python -m lidar_cloud predict --input measurements_h5 --output outputs/my_station --station MY-LIDAR --stats outputs/station_stats/my_station.json
```

- Keep calibration and inference consistent in instrument, signal preprocessing, temporal averaging and observed height range. Compute statistics from **finite positive measured elastic values only**; exclude padding.
- MAD is a **fixed scalar per station/instrument and elastic channel** (`1064`, `355p`, `532p`). Reuse the JSON across profiles and windows; do not recompute it for each input or borrow another instrument's values. Missing or invalid MAD raises an error.
- Elastic normalization is `0.5 * ((signal - median) / (4 * MAD + 1e-8) + 1)`, without clipping, followed by `log1p(normalized_signal * range_km**2)`. VDR channels are clipped to `[0,1]` and receive no MAD, range-squared or log transform. **Pass observation-level signals to the detector; it applies these transformations itself.**

The same commands support 10 profiles and long sequences; use `--profiles 0:10` for ten profiles from a file. Check a locally annotated subset before operational use. The [station deployment guide](docs/station-deployment.md) covers validation and Changsha/Beijing reference instruments.

## Pretrained weights for reuse

**R1/R2 are reusable initialization weights; E1-E4 are final detectors, including the staged E3/E4 models.** All six checkpoints contain model parameters only, about 17.33 MiB each. E5 is the reproduced VDE baseline and has no neural weights. Training code, training data and optimizer states are outside this user release.

```python
from lidar_cloud import load_model, load_backbone
pretrained = load_model("R1")    # includes reconstruction head
backbone = load_backbone("R2")  # attach your own task head
```

[Weight sources and checksums](lidar_cloud/resources/models.json) / [Python usage](examples/README.md) / [Paper annotation-efficiency figure](results/visualization/annotation_efficiency.png).

## Reference results

E1-E5 follow the manuscript's result order. The pixel values below match Table 2 and the original full-test implementation.

| Model | Configuration | IoU | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| E1 | Direct supervised | 0.865 | 0.928 | 0.909 | 0.947 |
| E2 | Direct supervised | 0.867 | 0.929 | 0.927 | 0.931 |
| **E3** | **Primary staged detector** | **0.881** | **0.937** | **0.925** | **0.949** |
| E4 | Staged comparison detector | 0.877 | 0.935 | 0.924 | 0.945 |
| E5 | Reproduced VDE | 0.540 | 0.702 | 0.639 | 0.778 |

Rounded reference boundary values are shown below; E2 is an additional evaluation.

| Model | Cloud-base MAE (m) | Detectable-top MAE (m) | Layer detection |
|---|---:|---:|---:|
| E1 | 55.3 | 94.0 | 94.1% |
| E2 | 63.4 | 98.2 | 94.0% |
| E3 | 59.1 | 82.2 | 93.9% |
| E5 | 123.6 | 250.7 | 86.0% |

Full-precision rerun scores are in [measured results](results/README.md). Boundary-MAE differences from archived values are at most 0.023 m due to floating-point matching ties. See [reference checks and release conventions](docs/paper-correspondence.md).

## Repository guide

| Location | Purpose |
|---|---|
| `lidar_cloud/` | Inference, calibration, evaluation, visualization and six parameter-only weights |
| `data/` | Three labeled example days and their manifest |
| `docs/` | Input format, station deployment, full-test and reference-check guides |
| `examples/` | Python usage and historical instrument-statistics configurations |
| `scripts/` | Strict full-test runner |
| `results/` | Reference scores, figures and one consolidated validation summary |
| `tests/` | Inference/calibration/reproduction checks |
| `outputs/` | Local generated predictions, scores and statistics; excluded from Git |

Cite the manuscript title above and release version 1.0.0; the final DOI is pending. [MIT License](LICENSE) applies to code and released weights. CARLNET observation-data terms are supplied separately by the data owner.
