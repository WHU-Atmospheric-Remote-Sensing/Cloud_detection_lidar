# README and manuscript correspondence

Checked against `AMT_en_v3_NOCHANGE.pdf`, the original checkpoint/index records and the executed release results. Page numbers below are the PDF's printed page numbers. The PDF is unchanged. E1–E5 follow the order of the five model/result rows in Table 2.

## Results that agree

All 20 IoU/F1/precision/recall entries in the README match Table 2 (p. 19) at its displayed precision. The full-test release rerun matches the original executable results exactly: 84 files, 420 model/file masks and 42,395,648 evaluated pixels per method, with zero differing masks or pixel metrics. See the [strict reproduction report](../results/full_test/reproduction.json).

The five input channels, 15 m grid, 2048-to-1056 vertical compression, 10-profile input, inference stride 1 and fixed final threshold 0.65 agree with Appendix A1/A3/A6 (pp. 28–32). Cloud tops mean detectable tops, consistent with Section 4.2.2 (pp. 19–20); they do not imply recovery above complete attenuation.

## Differences and their reasons

| Item | PDF | Release and reason |
|---|---|---|
| Elastic MAD normalization | A1, p. 28: per-sample statistics, `(x-median)/(1.4826*MAD+1e-6)` | Original executable preprocessing uses frozen scalar station/channel statistics, `(x-median)/(4*MAD+1e-8)`, then `(value+1)/2`, without elastic clipping. The release retains the checkpoint's actual input distribution to reproduce its predictions. The implementation also clips VDR to [0,1]; the PDF calls these native ratios without specifying this guard. |
| E1/E2 test-site label provenance | Table 2, p. 19: first supervised row excludes labels, second includes labels | The numerically corresponding original indices/checkpoints record the reverse: E1 comes from the `include` experiment, E2 from `exclude`. Numbering and values follow the paper; source records remain faithful. This conflict already exists between the manuscript and original experiment records, rather than being introduced by export. |
| E4 site lineage | Table 2 describes no test-site pretraining observations and some test-site fine-tuning labels | The exported checkpoint and result correspondence are verified, but its complete training-site history is not established by available records. A directory name alone cannot establish that history; the release does not assert it. |
| Boundary MAE | Table 3, p. 21 reports rounded archived results | Release boundary extraction uses above-ground coordinates; the archived calculation adds site elevation. Floating-point ties in Hungarian assignment can choose a different pair among equal-cost candidates. Matched-layer counts agree, while MAE differs by at most 0.023 m. The README labels paper values separately from the actual rerun CSV. |
| E5 versus Stage-2 teacher | Section 2.6, p. 11 averages VDE soft outputs from three wavelengths for distillation | Released E5 is the archived final evaluation baseline using 1064 nm candidates, mean elastic SNR > 3, five-profile support and 120 m final thickness. These exact baseline settings are not specified in the PDF. E5 is not an export of teacher scores thresholded at 0.4; that training threshold has another purpose (A6, p. 32). |
| Example plots | Fig. 6, p. 23 compares pretrained solid contours with direct-supervised dotted contours, within 0–6 km and a selected time interval | Release examples compare E3 solid contours with manual dotted contours, show the available day and stop at 10 km. Case dates were replaced at the user's request. The display conventions are reused, but the plots are labeled as release examples, not reproduced Fig. 6 panels. Full-height predictions/evaluation are retained. |
| Deployment additions | The reference experiment uses 10 real profiles on its fixed grid; irregular-input handling is not fully specified | Repeat-last time padding, raw-zero height padding, coordinate resampling, gap splitting and new-instrument MAD estimation are user deployment extensions. Their checks are documented; shorter inputs are not claimed equivalent to complete context or to achieve paper accuracy at a new site. |
| Release scope | Code/data availability, p. 33 plans training and inference code and weights upon publication | This authorized stage contains inference/evaluation, weights and independent test data. Training code/data and radiosonde figure reproduction are deferred. No database or DOI is claimed to exist. |

The distinction between the distillation teacher and the evaluation baseline also resolves an ambiguity within the PDF: Section 2.6 says VDE is not used in performance evaluation, while Section 4 and Tables 2–3 evaluate a VDE baseline. The release interprets the former as referring to the soft teacher product; E5 reproduces the independently evaluated baseline.

## Published versus rerun boundary values

| Model | PDF base MAE (m) | Rerun base MAE (m) | PDF top MAE (m) | Rerun top MAE (m) |
|---|---:|---:|---:|---:|
| E1 | 55.3 | 55.353020 | 94.0 | 93.982223 |
| E3 | 59.1 | 59.062500 | 82.2 | 82.252956 |
| E5 | 123.6 | 123.562344 | 250.7 | 250.680122 |

The 0.023 m comparison uses the full-precision archived values, not the already rounded PDF numbers. E1 base and E3 top round to 55.4 and 82.3 in the release rerun. Detection rates round to the same published values: 94.1%, 93.9% and 86.0%. E2/E4 boundary metrics are additional release evaluations, absent from Table 3. See [boundary CSV](../results/full_test/boundary_metrics.csv), [archived boundary references](../lidar_cloud/resources/reference_boundary_metrics.json) and [validation notes](../results/README.md).

The 90 m minimum layer thickness used by the evaluation is an original-code setting; the PDF describes the 300 m one-to-one physical matching rule without stating this minimum. Both are fixed in the reproduction script.

## Manuscript items to resolve

Before claiming that every methodological detail matches the executable release, reconcile Appendix A1 with the actual frozen-statistics preprocessing, reconcile the first two Table 2 provenance labels with the experiment indices, establish E4's training-site lineage, and distinguish the VDE teacher from the evaluated baseline in Section 2.6. The present audit establishes the discrepancies; it does not determine why those manuscript/record differences originally arose.
