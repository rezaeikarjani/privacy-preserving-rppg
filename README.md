# Privacy-Preserving rPPG Baseline from Facial Video

A CPU portfolio asking whether simple facial transformations preserve rPPG while reducing model-specific facial similarity. **A research prototype, not clinically validated or proven anonymous video.** Developed with AI assistance.

## Current evidence
- Independent POS and simplified CHROM; approximate forehead/cheek ROI processing; blur, pixelation and black-mask control.
- Synthetic 72-BPM fixture, private recording with sparse Garmin HR, and three official UBFC recordings with contact PPG.
- Actual isolated pyVHR-cpu 1.0.2 methods and rPPG-Toolbox classical POS comparison. Not full-framework training/dataset benchmarks.
- Signal-quality warnings and a fixed transformation-strength sweep; baseline estimates remain unchanged.
- Original automatic Haar/LK fails on a shirt false-positive in UBFC subject1. A separately labelled manual-region run completes. A new automatic YuNet per-frame detector completes all 1,547 frames, but was tested after that recording was inspected; broader detector validation remains pending.
- SFace similarity and closed-set ranking were measured across three dataset identities with temporally separated frames from each recording. Impostors are present, but independent capture sessions and calibrated verification accuracy are absent.
- Subjects3/4 were preselected; their official videos were obtained after an initial download-quota block. Both automatic detector conditions completed on both new recordings. See docs/EXPANDED_VALIDATION.md for all outcomes and reference uncertainty.
- **43 project tests pass.** No upload/publication of raw video, participant images or embeddings.

## Structure
| Path | Purpose |
|---|---|
| src/signals.py | Independent classical signal extraction, filtering and HR estimation |
| src/video.py | Original strict Haar/LK geometry, skin patches and transforms |
| src/quality.py | Exploratory competing-peak/band-edge warnings |
| src/reference.py | Timestamped contact-reference integrity/interpolation |
| src/face_models.py | CPU YuNet detector and embedding normalization |
| scripts/ | Reproduction, diagnostics, upstream comparisons and report generation |
| configs/ | Fixed settings, protocol hashes and model-demo protocol |
| tests/ | Signal, failure, timing, tracking-policy and embedding checks |
| docs/ | Consolidated report, learning notes and experiment protocols |
| results/synthetic/ | Engineering-fixture outputs |
| results/private/ | Human/dataset results, traces and aggregate plots; ignored by git |
| data/ and vendor/ | Inputs and downloaded dependencies/models; ignored by git |

## Install and verify
Tested host: macOS arm64, Anaconda Python 3.13.5; NumPy2.1.3, SciPy1.15.3, pandas2.2.3, Matplotlib3.10.0, OpenCV4.12.0, pytest8.3.4. No CUDA/CSC required.
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MPLCONFIGDIR=/tmp/rppg-mpl
python scripts/make_synthetic.py
python -m src.run --config configs/synthetic.json
python -m pytest -q
```
Requirements reflect installed/tested versions; a fresh clean-environment installation is not certified. Optional upstream/model-download dependencies are separate from the base requirements. `environment-linux.yml` is an untested Conda bootstrap, not a solved lockfile.

## Private facial video
Use only your own or explicitly consented single-person video; constant-frame-rate 30 FPS, stable diffuse light, stationary camera, visible forehead/cheeks, no beauty filter. Without synchronized reference, results measure consistency, not accuracy. Put input under data/private and edit configs/consented.json:
```bash
python -m src.run --config configs/consented.json
```
The current generic runner uses uniform frame spacing. The private HEVC recording was explicitly resampled with FFmpeg to 30 FPS at width 480 before analysis; that limitation is documented. Its Garmin device supplies only 13 sparse HR samples, not raw PPG. Watch files include location; exported analysis tables omit it. Full reproduction details remain in docs/EXTERNAL_VALIDATION.md and experiment manifests.

## Public contact-reference experiment
The official UBFC research source and accompanying sharing record were checked before downloading subject1. Raw dataset redistribution is not authorized here. See docs/EXTERNAL_VALIDATION.md.
```bash
pip install -r requirements-external.txt
python scripts/download_ubfc_sample.py
# Reproduces the strict automatic Haar failure on subject1:
python -m scripts.validate_external
# Explicit manual-region fallback, kept separate:
python -m scripts.validate_external --output results/private/external_ubfc_subject1_roi --target-region 250 140 220 210
# New automatic detector variant, no manual region:
python scripts/download_identity_models.py
python -m scripts.validate_external --output results/private/external_ubfc_subject1_yunet --detector yunet
```
Automatic YuNet HR MAE vs contact-PPG-derived spectral HR, only two overlapping windows:

| Method | Original | Blur | Pixelation |
|---|---:|---:|---:|
| POS | 0.643 BPM | 0.429 BPM | 1.072 BPM |
| CHROM | 0.643 BPM | 0.429 BPM | 0.857 BPM |

For subject1, the earlier manual-region run gave 0.429/0.429/0.857 BPM for both methods. Geometry differs, so results are not interchangeable. On subject1 no non-mask error exceeds 5 BPM, so warning recall is undefined there. New subjects3/4 contain much larger deviations and warnings include both misses and false alarms. One subject/two overlapping windows do not establish general or clinical accuracy. Reference HR is itself a spectral estimate. Timing checks remove one exact duplicate, reject conflicting duplicates, avoid extrapolation/lag fitting and report coverage.

## Signal preservation and quality
```bash
python -m scripts.evaluate_quality --directory results/private/garmin_15_120 --source-offset-s 15
python -m scripts.strength_sweep
```
In the private recording, stronger blur reduced whole-waveform similarity. At sigma fraction 0.10, POS switched from a peak near 64.6 to 44.8 BPM in one window; smaller blur settings did not show that switch. Pixelation reduced waveform similarity while HR changes stayed small. The warning uses competing-peak ratio<1.2 and band-edge checks, chosen after inspecting that recording. It does not correct HR or certify accuracy. See docs/QUALITY.md and docs/STRENGTH_SWEEP.md for full settings, sensitivity analysis and measured results.

## Identity-similarity and small-cohort experiments
```bash
python scripts/download_identity_models.py
python -m scripts.identity_demo
# After prescribed public videos and original tracking outputs exist:
python -m scripts.identity_cohort
```
Official CPU YuNet/SFace models are pinned to OpenCV Zoo revision 48b10fa505aeb51fbd127fc60a5c660040e823b7; hashes and licenses are retained. The original one-person demo remains separate. The new three-person experiment enrolls each identity from original frames at 5/10/15 s and tests probes at 30/35/40 s of the SAME recording, with the other two identities serving as impostors.

| Variant | Mean genuine cosine | Unique own-template rank1 | Re-detection |
|---|---:|---:|---:|
| Original | 0.984 | 9/9 | 9/9 |
| Blur | 0.125 | 6/9 | 0/9 |
| Pixelation | 0.202 | 8/9 | 0/9 |
| Black mask | 0.146 | 6/9 | 0/9 |

Original landmarks align transformed probes, an explicit assisted condition. Ranking forces a choice among only three original templates without rejection, even if all similarities are low. Ties are not successes. The masked-control matches highlight possible context/geometry or model/gallery bias, not proven facial recovery through a mask. These are not anonymity percentages, recognition probabilities, independent-session validation or general recognition accuracy. No face crops or embedding vectors are saved. See docs/IDENTITY_COHORT.md and the earlier docs/IDENTITY_DEMO.md. Do not mix Haar/assisted identity geometry with the separate YuNet rPPG geometry as a matched privacy/utility frontier.

## Upstream methods
```bash
pip install --target vendor/python-probe pyvhr-cpu==1.0.2
python -m scripts.compare_pyvhr_isolated
git clone https://github.com/ubicomplab/rPPG-Toolbox.git vendor/rPPG-Toolbox
git -C vendor/rPPG-Toolbox checkout b7500b848f84ad7f86e277b4612563b69f4f88f9
OPENBLAS_NUM_THREADS=1 python -m scripts.compare_toolbox --toolbox vendor/rPPG-Toolbox --traces results/private/garmin_15_120/rgb_traces.npz --output results/private/toolbox_comparison --numpy2-compat
```
The isolated pyVHR loader requires the tested scientific/Torch/scikit-learn stack and bypasses optional root video/GUI imports; it executes unmodified CPU methods, not full Pipeline. A target-directory MediaPipe trial brought incompatible optional dependencies; do not prepend vendor/python-probe to global PYTHONPATH. System packages were not replaced. Toolbox's explicit compatibility alias restores removed np.mat via np.asmatrix; upstream source remains unchanged. Internal filters differ. No full framework or neural training claim.

## Evaluation limits and research context
Classical extraction averages three approximate skin patches, not learned skin segmentation. Offline Butterworth filtering covers 0.7-3 Hz; 4096-point Welch zero padding adds no physiological resolution. Pearson is signed zero-lag, without hidden sign/lag fitting. The reference-band power ratio is a spectral concentration surrogate, not ground-truth physiological SNR. High waveform correlation can preserve an artifact. Black-mask signals are invalid, not successful low-error estimates.

Blur and pixelation do not guarantee biometric anonymity. Reusing original geometry is an assisted utility experiment; background/hair/clothes may remain identifying. All same-session embedding and single-recording results are limited. Full requested-paper critical reading remains incomplete; research context is based on accessible primary abstracts/author records. See docs/REFERENCES.md for Savic & Zhao, Wei/Yu/Zhao, PhysFormer and Contrast-Phys+. This portfolio does not reproduce their learned methods.

## Report and remaining work
- docs/technical_report.pdf: consolidated three-page report updated 12 November 2026.
- docs/EXPANDED_VALIDATION.md: additional-data access and outcomes.

Selected videos are now processed. Next: adjudicate reference uncertainty, test additional challenging independent data, and add independent-session enrollment with larger impostor cohorts. This public repository contains code, configurations, protocols, synthetic outputs and aggregate results only. Raw videos, dataset files, watch data, traces, face crops, embeddings and run logs are excluded by `.gitignore` and are not distributed.

## Expanded independent results (new participants, fixed rules)
MAE below is against the fixed contact-PPG spectral reference; subject3 reference is itself uncertain (spectral/device HR disagreement; first contact spectrum ambiguous). No detector is selected post hoc to show the best outcome.

| Subject | Detector | Method | Original | Blur | Pixelation |
|---|---|---|---:|---:|---:|
| 3 | Haar/LK | POS | 37.657 | 25.970 | 15.366 |
| 3 | Haar/LK | CHROM | 0.000 | 0.433 | 14.500 |
| 3 | YuNet | POS | 1.948 | 8.657 | 8.440 |
| 3 | YuNet | CHROM | 1.082 | 0.866 | 7.142 |
| 4 | Haar/LK | POS | 0.434 | 0.434 | 0.434 |
| 4 | Haar/LK | CHROM | 0.867 | 0.434 | 0.434 |
| 4 | YuNet | POS | 0.434 | 0.434 | 0.434 |
| 4 | YuNet | CHROM | 0.434 | 0.434 | 0.434 |

Units BPM; subject3 has two overlapping windows, subject4 one. Fixed warning counts (against >5-BPM spectral-reference deviation) are Haar TP4/FP5/FN3/TN6 and YuNet TP2/FP6/FN1/TN9. These are 18 correlated assessments per detector from only three temporal intervals, not validated classifier precision/recall. Every outcome, reference warning and mask invalidity is retained. See docs/EXPANDED_VALIDATION.md.

## Contact-reference audit
A post-hoc audit now compares pulse intervals, filtered Welch peaks, raw native-time Lomb-Scargle and device HR without changing baseline results. Subject3 has competing spectral components and differing HR summaries: Welch about 85-87 BPM, median detected intervals 98.49, span-count about 94-96, device 101-105. This exposes reference/estimand uncertainty, not a new ground truth or improved accuracy claim. [Full audit and all settings](docs/REFERENCE_AUDIT.md). Run `python -m scripts.audit_reference`; baseline hashes are checked unchanged. Current tests: 43 pass.
