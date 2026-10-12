# Input and output

The detector consumes observation-level **elastic signals and volume depolarization ratios (VDR)**, not backscatter/extinction retrievals. Complete the instrument's background subtraction, dead-time/gain corrections, range alignment and temporal averaging first. Do **not** apply range-squared correction, logarithms, MAD or z-score before calling the detector.

## Minimum H5

| Dataset | Shape | Meaning |
|---|---|---|
| `coordinates/time_utc` | `(T,)` | Strictly increasing UTC Unix seconds |
| `coordinates/height_km` | `(Z,)` | Strictly increasing range-aligned height above ground, km |
| `preprocessed_signals/1064` | `(T,Z)` | 1064 nm elastic signal |
| `preprocessed_signals/355p` | `(T,Z)` | 355 nm parallel-polarization elastic signal |
| `preprocessed_signals/532p` | `(T,Z)` | 532 nm parallel-polarization elastic signal |
| `depolarization/vdr_355` | `(T,Z)` | Calibrated 355 nm VDR |
| `depolarization/vdr_532` | `(T,Z)` | Calibrated 532 nm VDR |

Set root attributes `station_id` and `date` (`YYYYMMDD`). Optionally provide `station_elevation_m` for paper-style altitude plots; model inputs and saved height coordinates remain above ground. `--station` can explicitly select a station/instrument key. Verification additionally requires `manual_annotation/cloud_mask` with the same `(T,Z)` shape and binary labels. Legacy `coordinates/time_hours` is supported when a UTC date is available. The bundled data use the supported `station` attribute for the station ID and contain both UTC coordinate forms.

Released observations use `<station_id>_<YYYYMMDD>.h5`, for example `51628_20250106.h5`; predictions append the model ID, such as `51628_20250106_E3.h5`, `.csv` and `.png`. The full test stores these observations under `TEST_DATASET/data/`; the three repository examples use the same names and field layout.

For calibration, also record `instrument_id` and `time_avg_minutes` when known. Use one instrument and averaging interval per calibration directory; the tool rejects conflicting explicit metadata.

The released channel names match the actual training input. Do not replace `355p`/`532p` with total elastic signals while retaining their existing MAD statistics. VDE uses total elastic signals separately: supply `preprocessed_signals/355_total` and `532_total`, or `355s`/`532s` with root attributes `pgr_355`/`pgr_532`. Total = parallel + PGR × cross. The bundled data contain the required cross channels and calibration attributes.

## Python arrays

```python
from lidar_cloud import CloudDetector

detector = CloudDetector(model="E3", stats="my_station.json", device="cpu")
# signals: dict of the five arrays (T,Z), or array (T,Z,5)
# Array channel order: 1064, 355p, 532p, vdr_355, vdr_532
result = detector.predict(signals, height_km, time_utc, station_id="MY-LIDAR")
probability, mask, valid = result.probability, result.mask, result.valid
```

See [python_usage.py](../examples/python_usage.py) for a runnable 10-profile example and a minimal H5 writer.

## Sampling and padding

The reference sampling is 5 min, 15 m, and 2048 height bins. The internal vertical compression keeps bins 0–799, averages bins 800–1599 by four and the remaining bins by eight (1056 elements). Raw elastic signals undergo station MAD, then `log1p(normalized_signal × range_km²)`; VDR is clipped to `[0,1]`.

Normalization uses fixed station/instrument statistics: `0.5 * ((signal - median) / (4 * MAD + 1e-8) + 1)`, without elastic clipping. See the [station calibration guide](station-deployment.md).

Use the default batch size one when comparing individual probabilities with the original implementation. Larger GPU batches can introduce small floating-point differences near the threshold; complete test metrics remain subject to the documented reproduction tolerance.

10 profiles run directly; longer inputs use overlapping 10-profile windows with stride one. Logits are restored to the original vertical grid before sigmoid; overlapping probabilities are averaged. For 1–9 profiles, the last profile is repeated to reach 10 and padding predictions are discarded. A single profile is supported but **not recommended for operational decisions**.

Short heights are padded with **raw signal zero**, before normalization. Zero in raw units and zero in transformed model features are different. Nonstandard height grids are interpolated to the model grid and results are returned on the input grid. Heights beyond 30.705 km are invalid. Interpolation and padding do not establish equivalent detection accuracy. A boundary reaching the measured upper limit may be truncated.

Nonfinite elastic observations are replaced for computation and marked invalid in the output. Undefined VDR is replaced by zero, as in the original model; it does not discard an otherwise observed elastic pixel. A missing entire channel is an error. Inspect your input quality; replacing undefined ratios cannot create polarization information.

Deployment splits gaps longer than 1.5 × `--cadence-seconds` (default 300 s). Use `--gap-seconds` to specify the permitted spacing explicitly. For non-5-minute observations, set the cadence to the actual averaging interval. No missing timestamps are invented. Paper verification preserves the historical file-order windows, including missing intervals, to reproduce the reported estimates.

## Output

Each input produces separate `<input>_<model>.h5`, `.csv` and `.png` files. The H5 preserves the input coordinates and stores:

- `prediction/cloud_probability`: float32; invalid pixels are NaN.
- `prediction/cloud_mask`: uint8; 0 = clear, 1 = cloud, **255 = invalid**.
- `prediction/valid_observation`: uint8 validity indicator.

Layers are contiguous cloud-positive regions at least 90 m thick. The CSV reports cloud base and **detectable** cloud top above ground; cloud top is the upper edge of the final positive range bin. It need not be the true cloud top when the lidar signal is attenuated. The neural mask receives no additional closing/smoothing that would change the reference pixel scores. E5 reports a binary rule decision as its probability field; it is not a calibrated probability.

Inputs are never overwritten. Do not interpret mask 255 or a padded/unobserved interval as clear sky.

## Visualization

`predict` creates a **paper-style PNG by default**, matching the manuscript's 1064 nm curtain: `turbo` palette, `log1p(max(signal,0) × range_km²)` background and black cloud-mask contours with white halos. The natural logarithm and percentile color scaling follow the original paper plotting script; the colorbar retains its `log(P·R²)` notation. Solid contours show the selected model; dotted contours show **manual annotation** when available. The plot displays all detected components.

The display ceiling is **10 km**. With a known station elevation, the axis is altitude above sea level, as in the paper. The three reference-site elevations are documented in [station metadata](../lidar_cloud/resources/stations.json); your own `station_elevation_m` attribute takes precedence. Without an elevation the axis is explicitly height above ground. This choice affects only display: H5/CSV coordinates and evaluation remain above ground. White space below the site's elevation is unobserved ground-relative range, not clear atmosphere.

Use `--plot-style overview` for diagnostic signal/probability/mask panels in above-ground coordinates. `--plot-height-km` changes the display ceiling without cropping inference or evaluation; `--no-plots` disables automatic PNGs. Single-profile images use a cell of `--cadence-seconds` width and contain only the real profile. Invalid observations are gray and time gaps remain blank. Mask contours may include thinner components than the CSV's 90 m layer minimum; a contour clipped at the display ceiling does not establish a physical cloud top.

To plot existing predictions without loading model weights:

```bash
python -m lidar_cloud plot --input outputs/my_station --output outputs/figures
```

Add `--observations measurements_h5` to include source signal and manual labels. The reader matches the original filename, timestamps and heights, including `--profiles` slices; it rejects mismatched observations. Without this option, the PNG displays the saved binary mask, not a reconstructed signal. To avoid duplicate folders, redraw into the same output directory. All figures are standalone PNGs and work on a headless server. Time-height PNG metadata records `HeightCeilingKm`, `HeightReference` and `PlotStyle` for checking the display settings.

Paper E5 needs at least five real profiles and measured 17–18 km background data. Its operational working height range is 0.4–18 km; other heights are marked invalid. The paper's evaluation counts the full reference grid, including bins outside that working range, using the original binary E5 mask. `verify --protocol paper` preserves that convention.
