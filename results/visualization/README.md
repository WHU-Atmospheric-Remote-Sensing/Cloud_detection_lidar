# Executed examples

These are the same three labeled days provided in `data/examples`. After installation, run `python -m lidar_cloud predict --input data/examples --output outputs/examples` from the repository.

| Site | Day (UTC) | Profiles | Figure |
|---|---|---:|---|
| Aksu (51628) | 2025-01-06 | 200 | [PNG](51628_20250106_E3.png) |
| Yan'an (53845) | 2025-01-01 | 254 | [PNG](53845_20250101_E3.png) |
| Fuzhou (58847) | 2025-01-04 | 283 | [PNG](58847_20250104_E3.png) |

All figures use E3, threshold 0.65, and the paper's turbo 1064 nm curtain with black outlines and white halos. Solid outlines are E3; dotted outlines are manual labels. The vertical axis stops at **10 km above sea level**. Each PNG records the limit in its metadata. Saved H5/CSV retain the full measured height range.

Open `outputs/examples/*.png` for the local images; their same-stem `.csv` and `.h5` files contain layer heights and pixel products. `outputs/verification/metrics.png` summarizes the **three-day subset**, and `outputs/station_stats/` contains separately labeled calibration audits. Older duplicated visualization/smoke-test folders are outside the user output tree.

[Annotation efficiency](annotation_efficiency.png) is the paper's pretraining/data-size figure. The three curtains above are new release examples using its display conventions, rather than reproduced Fig. 6 panels. [Validation summary](../validation.json) records the execution checks.
