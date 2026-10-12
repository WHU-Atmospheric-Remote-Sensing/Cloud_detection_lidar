# Reference results and release conventions

This summary records the executed release checks and the conventions needed to use and verify the results.

| Type | Meaning |
|---|---|
| **Results agreement** | The README's IoU/F1/precision/recall values match the reference results at their displayed precision. The full-test rerun reproduces all **420 prediction masks** and all five global confusion matrices from the original implementation exactly, across **84 files**. |
| **Numerical and usage conventions** | Reference tables show rounded values; CSV files retain full precision. Boundary-MAE differences from the full-precision archived results are at most **0.023 m**, due to floating-point matching ties. Example plots stop at **10 km**, while saved predictions and evaluation retain the full valid height range. Short inputs use the documented padding and require local validation. |

See the [strict reproduction report](../results/full_test/reproduction.json), [measured boundary scores](../results/full_test/boundary_metrics.csv), and [input/output guide](input-format.md). Station calibration is described in the [deployment guide](station-deployment.md).
