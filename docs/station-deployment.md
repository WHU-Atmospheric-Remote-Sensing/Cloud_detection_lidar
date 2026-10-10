# Deploy at your lidar station

## 1. Prepare observation-level inputs

Organize your own measurements in the [minimum H5 format](input-format.md). Use one station/instrument key for each detector configuration. Two lidars at the same geographical station can have very different response scales and require separate statistics.

Complete instrument corrections and UTC/range alignment, then use the same temporal averaging for calibration and inference. The model does its own range-squared correction and log transform. Consistency here matters more than the absolute signal units. VDR requires correct polarization gain calibration; MAD does not repair an incorrect VDR or range origin.

## 2. Estimate and freeze your MAD statistics

Collect at least 8 representative calibration days covering daytime, nighttime and varying cloud conditions. Prefer a period separate from data you will use to assess deployment accuracy. Point `--input` to **one instrument's** calibration files:

```bash
python -m lidar_cloud compute-stats --input calibration_h5 --station MY-LIDAR --instrument-id receiver-01 --output outputs/station_stats/my_station.json
```

For each of `1064`, `355p` and `532p`, the tool computes scalar median and MAD from finite positive observed signals. The default deterministic reservoir retains at most 1,000,000 values per channel (`--seed 20260923`). It records input checksums, dates, sampling interval and sample counts. Statistics from fewer than eight days are marked as a trial. Nonpositive/invalid signals are excluded; all-zero/constant data or insufficient samples raise an error.

Files from different station IDs, explicit instrument IDs or averaging intervals are rejected. Set `instrument_id` in each input H5 when available; absent metadata cannot distinguish two receivers sharing the same station ID, so keep their calibration directories separate. Coordinates must be finite and strictly increasing.

The elastic normalization is `(signal - median) / (4 × MAD + 1e-8)`, followed by `(value + 1) / 2`. It is **not clipped to [0,1]**; the original implementation leaves elastic values unclipped. Do not substitute per-height statistics or a Gaussian z-score. Never include synthetic padding in the calibration data. The calibration tool reads the measurement grid directly, before resampling/padding.

MAD is per station/instrument and channel; it is not recomputed for every 10-profile inference window. Freeze and reuse the JSON until the detector response or preprocessing changes, then recalibrate and check results again.

**Current PDF discrepancy:** Appendix A1 describes per-sample normalization with `1.4826 × MAD + 1e-6`. The actual checkpoint code uses frozen station statistics and `4 × MAD + 1e-8`, then `(value + 1) / 2`, without elastic clipping. This release preserves that executable implementation: the full-test pixel results reproduce exactly. Substituting the appendix formula changes the checkpoint's input distribution.

## 3. Run and inspect

```bash
python -m lidar_cloud predict --input measurements_h5 --output outputs/my_station --station MY-LIDAR --stats outputs/station_stats/my_station.json --model E3
```

Add `--device cuda --batch-size 16` for GPU processing. For 10-minute averaged data, add `--cadence-seconds 600`; this changes the physical duration of each 10-profile window and is an adaptation, not the reference evaluation protocol. Unknown stations never fall back to another station's MAD values.

Open the automatically generated paper-style PNG beside each H5/CSV. It displays the 1064 nm curtain and outlined cloud masks up to 10 km. Add the correct `station_elevation_m` to your H5 attributes for altitude above sea level; otherwise the axis is height above ground. Inspect low clouds, weak high clouds, aerosol scenes, daytime noise and gaps. `--plot-style overview` provides probability/mask diagnostic panels; `--plot-height-km` changes the display ceiling. Supply a small, independently annotated subset using your station/instrument ID in its H5 attributes, then measure actual skill:

```bash
python -m lidar_cloud verify --data labeled_h5 --stats outputs/station_stats/my_station.json --models E3 --protocol deployment --output outputs/local_validation
```

The validation directory includes `metrics.png` with measured scores. Successful execution alone does not demonstrate the paper's cross-site accuracy.

## Changsha and Beijing examples

[Reference statistics](../examples/station_stats/) preserve historical scalar MAD configurations for different instruments. They are not defaults for your instrument.

| File | Key | Calibration context |
|---|---|---|
| `changsha.json` | `S001-CS` | Single-day, 10-minute compatibility trial; original sample-selection policy is not recorded |
| `beijing_kjb.json` | `S002-BJ` | KJB instrument, eight selected days, 10-minute averaging |
| `beijing_wxzk.json` | `S002-BJ` | Separate WXZK instrument, eight selected days; different gain and dead-time settings |

The Beijing files share a historical station key but describe different instruments. Keep them separate, or give the instruments distinct keys when constructing your own configuration. Their provenance records explain range offsets, gain factors and preprocessing; copy the **procedure**, not those instrument constants. None of these files establishes detection accuracy equivalent to the independent CARLNET test set.

## Executed MAD checks

[Consolidated validation](../results/validation.json) records eight-day checks at each test station: nine capped channel estimates match an independent sampling oracle exactly, and all 45 preprocessing comparisons match the original code, including raw-zero height padding. Maximum sampling deviation from all finite positive values is 0.675%.

The historical Changsha/Beijing JSON values match their source configurations; their original raw-data periods were not recomputed here. The generic positive-value estimator follows the recorded Beijing procedure, using a bounded uniform-priority sampler instead of its historical random-choice sampler. New estimates need not be bit-identical to historical statistics. The generated `outputs/station_stats/51628.json`, `53845.json` and `58847.json` are **audit examples**; bundled frozen CARLNET statistics remain the defaults for paper reproduction.
