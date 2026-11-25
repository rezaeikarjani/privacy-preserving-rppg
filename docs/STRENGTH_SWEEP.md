# Transformation-strength sweep

A parameter sweep changes one setting systematically while keeping other experimental choices fixed. Here we test whether physiological preservation depends on how much we blur or pixelate the image. This is a single-recording exploration, not a validated privacy/utility frontier.

## Fixed choices
- Same local video crop (source seconds 15-120), decoded at 30 FPS and width 480.
- Same original-face tracking coordinates and three skin patches, no redetection on transformed frames.
- Same independent POS and simplified CHROM implementations and signal postprocessing.
- Same 30-second evaluation windows with 15-second steps and 2-second edge trim.
- Same quality settings (peak ratio 1.2, 0.15-Hz competitor separation, 0.05-Hz band-edge warning).
- No watch data enter this experiment, no HR correction and no parameter optimization.

## Transformation settings
Blur sigma / face size: 0.025, 0.05, 0.10. Larger sigma gives more smoothing.
Pixelation grid width: 32, 16, 8 cells. Fewer cells give larger blocks.

These values were fixed before running the sweep, but the earlier baseline at sigma 0.10 and grid 8 was already inspected. This is not a preregistered independent experiment. Settings do not quantify identity privacy. A more degraded image is not necessarily anonymous.

## Measurements
- Whole-recording HR and its absolute change from original.
- Whole-waveform signed zero-lag Pearson correlation with original.
- Mean and maximum absolute window-HR change from original (not watch-reference error).
- Number of quality-flagged windows, out of five overlapping windows per variant/method.

Interpret metrics together: a good aggregate HR can coexist with waveform distortion and an individual-window failure. High correlation may preserve a shared artifact. A quality warning is exploratory and cannot tell whether a specific estimate is wrong.

## Integrity checks
The input video SHA256 must match the baseline. Every decoded frame's original ROI RGB must match its cached baseline values. Newly recomputed original BVPs must match cached waveforms. Source baseline result, tracking, window and signal files are hash checked before/after and left unchanged. Already evaluated sigma 0.10 and grid 8 reuse baseline RGB traces; other settings are computed from decoded video frames with exactly the same transform function. No source or transformed faces are exported by this sweep.

A separate controlled-texture smoke check confirmed that increasing blur sigma and coarsening pixelation reduced Laplacian spatial variation. That check verifies transform ordering only; it is not a biometric privacy measurement.

## Reproduce
```bash
export MPLCONFIGDIR=/tmp/rppg-mpl
python -m scripts.strength_sweep
```
Configuration: `configs/strength_sweep.json`. Results stay in `results/private/strength_sweep/`: summary.csv, windows.csv, strength_sweep.png, cached traces/BVP and manifest.json. Window times in this sweep are relative to the analysis crop; add 15 seconds to obtain original video times.

The consolidated report PDF summarizes the extensions; these addenda retain full protocol details. The next defensible validation is to freeze these rules and evaluate independent recordings with synchronized contact PPG under the dataset's usage conditions. Report all settings, not just the one with favourable values.

## Actual results
All 3,149 frames passed original RGB reproduction checks; baseline hashes remained unchanged. Sweep runtime was 52.6 seconds on this host (including cached-trace reuse; not a general speed benchmark).

POS blur at sigma fractions 0.025 / 0.05 / 0.10: correlations 0.977 / 0.959 / 0.932; mean absolute window HR changes 0.176 / 0.352 / 4.570 BPM; quality warnings 0 / 0 / 1 of five windows. The strongest blur repeats the known prior failure, rather than independently discovering a new one.

CHROM blur: correlations 0.977 / 0.948 / 0.892; mean changes 0.176 / 0.264 / 0.439 BPM; warnings 0 / 0 / 1.

POS pixelation grids 32 / 16 / 8: correlations 0.967 / 0.924 / 0.818; mean window HR changes 0.264 / 0.352 / 0.264 BPM; no quality flags. CHROM correlations 0.977 / 0.931 / 0.832; mean changes 0.088 / 0.264 / 0.264 BPM; no quality flags.

Interpretation: within this recording, more image degradation reduced waveform similarity, but HR changes did not always increase monotonically. Stable peak frequency does not imply unchanged waveform. We cannot choose a best privacy/utility setting without identity measurements and independent validation.
