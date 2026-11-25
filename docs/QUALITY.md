# Exploratory spectral-quality addendum

This extends the baseline; the current consolidated technical_report.pdf now summarizes the extension. The consolidated PDF now summarizes this extension; this addendum retains full protocol details. Current project suite: 22 passing tests (13 earlier + 9 quality tests).

## Question
Can a simple spectrum-based rule warn when a dominant-peak HR estimate is ambiguous? This does not estimate a probability of accuracy, decide which candidate is physiological, correct HR, or measure identity privacy.

## Fixed implementation and configuration
`src/quality.py` uses the same Welch spectrum as the unchanged HR baseline. It computes strongest-peak power / strongest distinct competing-peak power in 0.7-3 Hz. Competitors must be local maxima at least 0.15 Hz apart to avoid treating adjacent zero-padded bins as separate peaks. A ratio below 1.2 marks competing-peak ambiguity. A dominant peak within 0.05 Hz of a search-band edge is separately flagged. Flat, nonfinite, short and non-1D signals are invalid and flagged; invalid mask controls are reported separately from valid windows. No-rival ratio is infinity, which is NOT proof of a physiological signal.

Settings live in `configs/quality.json`. Ratios 1.1, 1.2 and 1.5 are evaluated as sensitivity analysis. These are heuristics chosen AFTER seeing this recording's failure, not preregistered or independently validated thresholds. The flag function accepts no watch HR. This avoids direct label-based correction, but does not eliminate post-hoc selection bias. Independent synchronized-reference recordings are still needed.

## Actual results at ratio threshold 1.2
Each non-mask method/variant has five overlapping windows. POS: original 0/5 flagged, blur 1/5, pixelation 0/5. CHROM: original 0/5, blur 1/5, pixelation 0/5. All five mask windows per method are invalid (not reliable signals).

The POS blur window at source-video seconds 32-62 retains its baseline 44.824 BPM estimate. Peak ratio 1.034545 triggers ambiguity; the dominant frequency also lies near the lower band edge. The competing candidate is around 64.6 BPM. CHROM blur in the same interval has a baseline 65.039 BPM estimate with ratio 1.101486 and is flagged for ambiguity despite agreeing approximately with watch values. Thus a flag must not be interpreted as a proven error.

The runner recomputes baseline HR for each cached window, requires equality with stored HR, and checks SHA256 hashes of original BVP/window/results files before and after. Baseline estimates/files remain unchanged. No low-quality windows are removed from the original metrics and no post-rejection accuracy improvement is claimed.

## Reproduce
Run from repository root using the verified Anaconda Python or installed requirements:
```bash
python -m scripts.evaluate_quality --directory results/private/garmin_15_120 --source-offset-s 15
python -m scripts.evaluate_quality --directory results/synthetic
python -m pytest -q
```
Outputs: `quality_windows.csv` (individual peaks/flags), `quality_summary.csv` (threshold sensitivity), `quality_manifest.json` (settings/hashes), `quality_flags.png` (all original estimates with warning markers). The real recording's outputs remain private.

## Limitations and next evaluation
Harmonics or genuine low HR near a band edge may produce a warning. A single strong motion/light artifact can evade the flag; a clear spectrum does not guarantee accuracy. Spectral resolution depends on the Welch segment length. One recording, overlapping windows and sparse watch samples cannot validate detection accuracy, precision/recall or rejection benefits. Synthetic tests verify intended behaviour, not human physiological validity.

Next: fix the quality rules before new data, evaluate a public dataset with synchronized contact PPG under its usage conditions, and report both error and retained-window coverage if rejection is later tested. Report sensitivity results without selecting whichever threshold looks best on this participant.
