# Example data

Three labeled CARLNET site-days, selected from the separate 84-file independent test. Files are identical to their counterparts in `TEST_DATASET/data/`; no training observations are included.

| File | Station | UTC date | Profiles |
|---|---|---|---:|
| `examples/51628_20250106.h5` | Aksu | 2025-01-06 | 200 |
| `examples/53845_20250101.h5` | Yan'an | 2025-01-01 | 254 |
| `examples/58847_20250104.h5` | Fuzhou | 2025-01-04 | 283 |

`examples/manifest.json` records dates, file sizes and SHA-256. Each H5 contains five model channels, VDE cross-polarization channels/PGR, time/height coordinates and manual cloud labels. Arrays are `(time,height)`, with 2048 gates at 15 m and nominal 5-minute sampling; missing intervals remain missing. Signals precede MAD, range-squared correction and logarithms. The [input guide](../docs/input-format.md) defines the fields.

Run from the repository:

```bash
# Three whole-day examples; default model E3
python -m lidar_cloud predict --input data/examples --output outputs/examples
# Ten real profiles from one day
python -m lidar_cloud predict --input data/examples/51628_20250106.h5 --profiles 0:10 --output outputs/ten_profiles
# All five methods on the example subset
python -m lidar_cloud verify --data data/examples --output outputs/verification
```

Predictions use `<station_id>_<YYYYMMDD>_<model>.png/.csv/.h5`. Open the PNG first; it stops at 10 km while H5/CSV retain observed heights. Verification creates `metrics.png` and numeric summaries. **Subset scores are not full-test scores.** For the full 84-file data, use the [test-dataset guide](../docs/test-dataset.md). For your own instrument, use [station deployment](../docs/station-deployment.md).
