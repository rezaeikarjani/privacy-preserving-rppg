# Expanded independent-recording test — completed 12 November 2026

## Selection, access and frozen protocol
Subject3 and subject4 were selected in numeric order before their outcomes (configs/expanded_validation.json). Initial automated downloads were blocked by official Google Drive quota; those logs are retained as history. The user then placed the prescribed official vid.avi files locally. Both videos and corresponding reference files were verified as readable; subject3's reference was byte-checked against the official source. Their source hashes are saved. This access issue is resolved, not a continuing algorithm/data blocker.

The original strict Haar/LK baseline AND a separate automatic YuNet variant ran on both new recordings. YuNet threshold 0.9 and model revision 48b10fa505aeb51fbd127fc60a5c660040e823b7 were recorded before new HR outcomes. No manual target region was needed on either new participant. The earlier subject1 manual fallback and later YuNet variant remain separate development-condition results. Signal methods, quality threshold1.2, 0.7-3 Hz band, blur sigma 0.10, pixelation grid 8 and windows were unchanged; protocol hashes match.

Subject3: 1801 frames at 29.548271 FPS, about 61 s, two overlapping 30-second windows. Subject4: 1368 frames at 29.609362 FPS, about 46 s, one 30-second window. Thus there are only THREE distinct new temporal intervals from TWO new people. Applying two methods and three transformations yields18 correlated assessments per detector, not 18 independent participants or recordings.

## All measured HR MAE values
MAE below is conditional on the preselected contact-PPG spectral reference, not unquestionable clinical truth. Mask outputs are invalid and excluded from numeric error, not assigned zero.

| Subject | Detector | Method | Original | Blur | Pixelation | Windows per version |
|---|---|---|---:|---:|---:|---:|
| 3 | Haar/LK | POS | 37.657 | 25.970 | 15.366 | 2 |
| 3 | Haar/LK | CHROM | 0.000 | 0.433 | 14.500 | 2 |
| 3 | YuNet | POS | 1.948 | 8.657 | 8.440 | 2 |
| 3 | YuNet | CHROM | 1.082 | 0.866 | 7.142 | 2 |
| 4 | Haar/LK | POS | 0.434 | 0.434 | 0.434 | 1 |
| 4 | Haar/LK | CHROM | 0.867 | 0.434 | 0.434 | 1 |
| 4 | YuNet | POS | 0.434 | 0.434 | 0.434 | 1 |
| 4 | YuNet | CHROM | 0.434 | 0.434 | 0.434 | 1 |

Units:BPM. Matching spectral bins can yield zero MAE without establishing perfect accuracy. Detector/ROI geometry differs; do not select whichever detector looks best per person. Both detector conditions are fully reported. A geometry spot check placed intended patches on facial skin, but is not exhaustive per-frame ROI validation or proof of error causation.

## Reference uncertainty exposed
On subject3 the contact-PPG spectral references are approximately 87.43 and 84.84 BPM; device-reported timestamp-weighted HR is approximately 100.95 and 104.56 BPM. The first contact-PPG spectrum also triggers our ambiguity rule. The second does not, despite disagreement with device HR. The correct interpretation is unresolved reference uncertainty, not an automatic decision that either reference is true.

Both references remain in all_windows.csv and reference_diagnostics.csv. We did not switch to the reference that makes the method look better, retune the band or remove the ambiguous window. Error-event counts below are deviations from the fixed spectral-reference estimate and may not equal genuine physiological errors. Additional reference adjudication is needed.

## Fixed quality-rule outcomes
An error event is absolute HR difference >5 BPM against the fixed reference, selected before outcomes.

| Detector | TP | FP | FN | TN | Descriptive precision | Descriptive recall |
|---|---:|---:|---:|---:|---:|---:|
| Haar/LK | 4 | 5 | 3 | 6 | 0.444 | 0.571 |
| YuNet | 2 | 6 | 1 | 9 | 0.250 | 0.667 |

These are raw small-sample, correlated-assessment counts, NOT validated classifier performance or population estimates. Six of the 18 assessments per detector share the same ambiguous reference interval; they are not six independent questionable references. The warning misses some large reference deviations and flags some near-reference estimates. A strong single artifact can escape a peak-ratio rule. No post-result tuning or claimed quality improvement was made.

## Reproduce
```bash
python -m scripts.validate_external --dataset-dir data/public/ubfc/subject3 --output results/private/external_ubfc_subject3_haar
python -m scripts.validate_external --dataset-dir data/public/ubfc/subject3 --output results/private/external_ubfc_subject3_yunet --detector yunet
python -m scripts.validate_external --dataset-dir data/public/ubfc/subject4 --output results/private/external_ubfc_subject4_haar
python -m scripts.validate_external --dataset-dir data/public/ubfc/subject4 --output results/private/external_ubfc_subject4_yunet --detector yunet
python -m scripts.summarize_expanded
```
Results are separate per run. Consolidated outputs: results/private/expanded_validation/per_subject_summary.csv, all_windows.csv, quality_confusion_descriptive.csv, reference_diagnostics.csv, expanded_mae.png and manifest.json. Strict original failure on subject1 remains in its historical log. No raw faces, videos or physiological files were uploaded/published. This is a small test across three public identities in total, not the complete UBFC benchmark or a clinical validation.
