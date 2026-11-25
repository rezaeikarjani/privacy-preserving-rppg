# Contact-reference audit — 12 November 2026

## Purpose and scope
Subject3's fixed contact-PPG spectral HR differs substantially from its device-reported HR. This post-hoc audit examines the reference using pulse intervals, waveform shape and two spectral methods. All settings are retained. No diagnostic is selected because it makes an rPPG estimate look better, and no existing error table, trace or reference is overwritten.

Configuration: configs/reference_audit.json. Subjects1/4 are comparison recordings, not independent validation of the new audit method. The audit uses no new sensor, ECG or manual expert pulse annotations, so it cannot establish definitive physiological truth.

## Fixed diagnostics
- Existing contact processing: timestamp-based linear interpolation, 0.7-3 Hz zero-phase Butterworth and Welch with 10-second segments, 4096 FFT points.
- Shape sensitivity check: 0.4-5 Hz filter, positive/negative peaks, prominence 0.25/0.5/1.0 times signal SD, minimum peak distance 1/3 second. All 12 settings per window are reported. These are post-hoc pulse-detection heuristics, not validated ground truth.
- Pulse summaries: median-interval HR  = 60/median(inter-pulse interval); span-count HR  = 60*(number of intervals)/(time between first and last detected pulses). These are different estimands and need not agree for varying/detected-irregular pulse timing.
- Native-timestamp Lomb-Scargle on linearly detrended raw PPG. It differs from the filtered/resampled, segmented Welch estimator in more than one respect. Comparison does not isolate interpolation as a cause.
- Device HR: timestamp-weighted mean, without changing its timing or treating it as unquestionable truth.
- No lag search, signal stretching, rPPG-label-based tuning, polarity selection or exclusion of unfavourable windows.

## Actual subject3 measurements
| Reference summary | First window (about 2-32 s) | Second window (about 17-47 s) |
|---|---:|---:|
| Existing Welch dominant peak | 87.43 BPM | 84.84 BPM |
| Next Welch peak | 100.85 BPM | 104.31 BPM |
| Native-time Lomb-Scargle strongest peak | 101.40 BPM | 89.82 BPM |
| Median detected pulse-interval HR | 98.49 BPM | 98.49 BPM |
| Span-count HR across all audit settings | 94.28-94.61 BPM | 95.89-96.17 BPM |
| Device HR mean | 100.95 BPM | 104.56 BPM |

All 12 pulse-detection settings in each window return the same median 98.49 BPM, but that agreement is only sensitivity within these heuristics, not proof of physiological accuracy. The common shape-band positive prominence 0.5 setting detects 47 pulses per window. Detected intervals span approximately 0.541-0.778 s and have median 0.609 s; interval IQR is about0.102/0.093 s. These are detected timing variations, not a diagnosis or proof of their physiological cause.

The first Welch spectrum has near-equal peaks around 87 and 101 BPM; the second also has both 85 and 104 BPM components. Different estimators and pulse summaries favor different components/statistics. A strongest spectral peak, median instantaneous rate, count-per-span and a device's internally processed HR are not interchangeable definitions of average HR. Rate/amplitude variation, sampling/interpolation, artifacts or device processing could contribute; causes are not established.

## Comparison recordings and timing
For the displayed shape-band positive prominence 0.5 setting, subject1 median-interval HR is 103.29/109.74 BPM, Welch 106.31/106.74 and device 104.80/109.71. Subject4 values are 111.04,111.47 and111.57 BPM. Thus some disagreement also exists without the large subject3 split; no method is automatically perfect.

Reference/video duration endpoints differ by less than 0.15 s across these recordings. We found no obvious gross global timing-scale error, but endpoint agreement does not certify every timestamp or remove within-recording timing uncertainty. Reference maximum gaps, FPS and durations are in timing.csv. Subject3's reference file had already been byte-checked against its official source.

## Consequences for our claims
Keep the original benchmark errors labelled **error versus a fixed contact-PPG spectral-reference estimate**, not clinical HR accuracy. The warning confusion counts also remain conditional on that reference. Do not silently replace it with 98.49 or device HR, exclude subject3, change the band to favor another peak or claim the algorithm is now more accurate.

Example: subject3 YuNet POS blur in the second window estimates 99.55 BPM. Its difference is 14.72 BPM from the original spectral reference, about5.01 from device HR and1.06 from the median pulse-interval diagnostic. Those alternatives illustrate reference/estimand sensitivity, not a licence to select the lowest error or prove which number is correct. Other failures, including Haar POS estimates near48 BPM, remain concerns against all these summaries; the audit does not explain away all pipeline failures.

For a stronger future reference, predefine the target HR statistic, validate pulse detection with appropriate independent evidence/manual expert annotation and report reference quality. Without adjudication, retain explicit reference uncertainty and avoid ranking methods as clinically accurate.

## Reproduction and output
```bash
python -m scripts.audit_reference
python -m pytest -q
```
Outputs under results/private/reference_audit: pulse_estimates.csv (all settings), pulse_intervals.csv, spectral_peaks.csv, timing.csv, subject*_audit.png and manifest.json. SHA256 checks verify that pre-existing baseline/error/trace files remain unchanged. New synthetic tests check known pulse rate, invalid flat signals and the difference between median-interval and span-count rate; they do not validate these methods on human PPG.
