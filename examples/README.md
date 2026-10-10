# Python and instrument examples

## Run ten profiles

After installation, run from the repository:

```bash
python examples/python_usage.py --input data/examples/58847_20250104.h5 --profiles 10 --output outputs/python_example
```

The [script](python_usage.py) demonstrates `CloudDetector.predict`, a minimum H5 writer, and saving H5/CSV/PNG. Its signal dictionary contains five `(T,Z)` arrays. Array inputs use `(T,Z,5)` in the order `1064, 355p, 532p, vdr_355, vdr_532`. Instrument preprocessing comes first; the detector applies its own MAD/range/log processing.

## Reuse initialization weights

```python
from lidar_cloud import load_model, load_backbone
pretrained = load_model("R1")
backbone = load_backbone("R2")
# Normalized/compressed backbone input: (B,T,1056,5)
# Backbone features: (B,48,T,1056)
```

These calls load parameters; implementing another task head or training procedure is left to the consuming algorithm. [Checkpoint metadata](../lidar_cloud/resources/models.json) gives stages, source filenames and SHA-256 without asserting an unverified complete training chain.

`station_stats/changsha.json`, `beijing_kjb.json` and `beijing_wxzk.json` preserve historical statistics for **three distinct instruments**. They illustrate configuration structure and are not defaults for a new station. The [deployment guide](../docs/station-deployment.md) explains their periods, instrument keys and limitations.
