# Independent-recording validation: UBFC-rPPG Dataset 2

## Data selection and usage
Official author page: https://sites.google.com/view/ybenezeth/ubfcrppg . It describes synchronized webcam video and CMS50E contact pulse-oximeter PPG, shared for research. Official Google Drive folder: https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk .

The accompanying Agreement.xlsx marks subject1 as agreeing to data sharing and showing images. The readme specifically restricts publishing face images for subjects21/27; we do not use those recordings and export no participant pictures. Author material was downloaded locally under data/public/ubfc and is ignored by git. Research access is not a blanket unrestricted redistribution license. No raw dataset, new face image or video is published/uploaded.

Citation: S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri and J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography," Pattern Recognition Letters, 2017.

LGI-PPGI was also considered. Its official repository https://github.com/partofthestars/LGI-PPGI-DB currently warns that sessions are unavailable for download. We therefore chose the accessible official UBFC source, not an unofficial mirror.

Subject1 was selected as the first available subject in numeric order, before seeing estimation results. This is an initial independent-recording feasibility evaluation, not a full UBFC benchmark or population study. It is independent from the personal recording used to design the quality heuristic.

## Frozen protocol and timing
The experiment settings and SHA256 of src/signals.py, src/quality.py, src/video.py and configs/quality.json were stored in configs/external_validation_protocol.json before external outcomes. The original version proposed device-reported HR as primary. Before subject signal access, the primary was amended to spectral HR from contact PPG because the official readme says the authors evaluate against contact-PPG-derived HR rather than the sensor's HR. Original protocol version is retained as external_validation_protocol_v1.json.

Fixed methods: local POS and simplified CHROM; original/blur sigma 0.10/pixelation grid 8/black mask; full video at original metadata FPS; original geometry shared between versions; 0.7-3 Hz band; 2-second boundary trim; 30-second windows with 15-second steps. Quality threshold 1.2, minimum competitor separation 0.15 Hz and band-edge warning 0.05 Hz remain unchanged. No lag search, label-based HR correction, time stretching or new threshold fitting.

Reference format (author readme): line1 contact PPG, line2 device HR, line3 timestamps in seconds. Timestamp origin is normalized to zero. Exact duplicate timestamp/value tuples may be removed, but backwards timestamps or conflicting values at equal times cause failure. Subject1 contains one exact duplicate at 16.750 seconds; both PPG and HR values are identical. Removal does not average or change measurements. This handling was added after the integrity check stopped on that duplicate; it does not alter the frozen rPPG/quality methods.

Contact PPG is linearly resampled using its timestamps to the uniform video timeline; no extrapolation or interpolation across gaps larger than 0.5 s. The same fixed filter and Welch estimator derive its reference HR. Device HR is a secondary timestamp-weighted mean. First dataset-relative times are aligned; metadata FPS and measured contact duration are reported, not stretched to match. Windows without reference coverage are marked, not silently scored. Waveform zero-lag correlation is descriptive because sensor sites, pulse transit time and residual synchronization error can shift phase.

## Evaluation
Report every window and method/variant. HR MAE and RMSE are against contact-PPG spectral HR; they are limited research estimates, not clinical validation. Classify error >5 BPM as an exploratory error event (fixed before outcomes) and report warning TP/FP/FN/TN, precision and recall only where denominators exist. Invalid estimates are counted separately. If retained-window error is reported, also report retained fraction and all-window error. It is a selected subset, not automatic evidence of improved accuracy.

Overlapping windows from one participant are not independent subjects. The quality heuristic remains post-hoc relative to the original personal recording. A single external recording cannot validate a general-purpose confidence score, clinical accuracy, or identity privacy.

## Reproduction
With working network and sufficient storage (one uncompressed video is about 1.43 GB):
```bash
pip install -r requirements.txt -r requirements-external.txt
python scripts/download_ubfc_sample.py
python -m scripts.validate_external
python -m pytest -q
```
In this session gdown was installed under vendor/gdown-probe and used with that directory on PYTHONPATH only for downloading. No global dependency stack was replaced. The downloader skips existing final filenames; inspect manifest hashes before reuse. Failed partial downloads do not count as completed inputs.

Default outputs: results/private/external_ubfc_subject1/config.json, manifest.json, validation_manifest.json, results.csv, windows.csv, reference_windows.csv, reference_summary.csv, reference_comparison.png and BVP/RGB caches. Keep these local until aggregate-output review and explicit publication authorization.

## Actual outcome: strict failure and assisted fallback
The strict original detector aborts on frame0 with "Multiple faces". Inspection shows the actual face at approximately [286,183,147,147] and a false detection on the patterned shirt. The strict failure log is kept locally. This is a real end-to-end limitation and is not counted as successful automatic validation.

Before seeing HR results, a fixed manual target region [250,140,220,210] (x,y,width,height in the 640x480 image) was specified in configs/external_roi_assistance.json. It filters detection centres, rather than fabricating a detected face box. Multiple detections within that region still trigger failure; missing detections are not replaced by a manual box. `src/run.py` gained an optional tracker factory; the default strict tracker and frozen signal/quality implementations remain unchanged. Report the fallback as **ROI-assisted external evaluation**, not full frozen-pipeline generalization.

Reproduce the strict failure with the default command. Reproduce the assisted result with:
```bash
python -m scripts.validate_external --output results/private/external_ubfc_subject1_roi --target-region 250 140 220 210
```
To recalculate reference comparison without re-extracting:
```bash
python -m scripts.validate_external --output results/private/external_ubfc_subject1_roi --target-region 250 140 220 210 --reuse-extraction
```
The evaluator asserts that its window indexing reproduces stored baseline HR, including integer edge-trim rounding at noninteger FPS. Reuse corrects public/private source-kind metadata without changing numeric estimates.

The recording has 1,547 video frames at 29.264106 FPS; last frame time 52.829 s. The reference has 1,547 entries, reduced to 1,546 unique timestamps after one exact duplicate; contact duration 52.691 s and maximum gap 0.223 s. No lag fitting or duration stretching was applied. Two overlapping windows are approximately [1.982,31.950] and [16.949,46.918] seconds. Both reference windows are covered and their contact-PPG spectra do not trigger the fixed ambiguity rule.

Actual HR MAE vs contact-PPG-derived spectral HR: POS original 0.429 BPM, blur 0.429, pixelation 0.857; CHROM original 0.429, blur 0.429, pixelation 0.857. Both black-mask windows per method are invalid and have undefined error. All six non-mask method/variant combinations retain 2/2 windows and generate no warnings. No error exceeds the fixed 5-BPM event threshold, so warning precision and recall are undefined, not 100%. This recording therefore cannot validate error detection. The spectral reference is an estimate, not perfect clinical truth.

Current tests: 30 passing, including reference format/timing, duplicate handling and target-region filtering. The consolidated three-page technical_report.pdf summarizes the extensions; this addendum retains detailed protocol and results. Further independent participants and challenging recordings are needed before generalization or clinical claims.
