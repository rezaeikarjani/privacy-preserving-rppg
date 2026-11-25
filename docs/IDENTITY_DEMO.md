# Descriptive face-embedding experiment

## Actual scope
One publicly shared UBFC Dataset2 participant, one capture session. This is a model-specific embedding-similarity demonstration, NOT verification/identification accuracy, anonymous-video certification or a validated privacy guarantee. No real-world name is inferred. No subject's image or embedding vector is exported or uploaded.

OpenCV CPU runs official OpenCV Zoo YuNet (2023mar ONNX, MIT license) and SFace (2021dec ONNX, Apache 2.0 license), pinned at repository revision 48b10fa505aeb51fbd127fc60a5c660040e823b7. Original model URLs, byte sizes and SHA256 are in vendor/opencv_models/models.json and copied to the experiment manifest. Source: https://github.com/opencv/opencv_zoo ; API: https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html . Upstream license files are retained locally. No model training or CUDA was used.

## Enrollment and probes
Settings were fixed before reading scores in configs/identity_demo.json. Original frames at 5, 10 and 15 seconds produce normalized 128-dimensional SFace embeddings. Their mean is normalized into one enrollment template. Probe frames are at 30, 35 and 40 seconds. Enrollment and probes are different frames and temporally separated, but they are NOT independent recording sessions; pose, background, camera and appearance are shared.

Original/blur sigma0.10/pixelation grid8/black mask use the same original Haar-region geometry as the earlier ROI-assisted rPPG experiment. This is not the new YuNet geometry used for automatic physiological validation, so do not combine those two runs into a matched privacy/utility curve.

Two tracks are separate:
1. Assisted similarity uses landmarks detected on each original probe to align every transformed version. This gives the evaluation geometry unavailable from the de-identified video alone; it is an explicit assisted/side-information condition. Embeddings are computed even when transformed-face detection fails.
2. End-to-end re-detection attempts YuNet separately on each transformed frame at fixed score threshold 0.9. Failure counts are reported, not converted into a similarity score of zero or proof of anonymity.

## Actual measurements
Three probes per variant. Mean cosine similarity to the original enrollment template:

| Variant | Mean assisted cosine | Re-detection successes |
|---|---:|---:|
| Original | 0.985913 | 3/3 |
| Blur | 0.192941 | 0/3 |
| Pixelation | 0.169987 | 0/3 |
| Black mask | 0.075534 | 0/3 |

Missing end-to-end similarities are undefined, not zero. Similarity decreased for this model in this narrow within-recording experiment. These values are NOT percentages, recognition probabilities or anonymity levels. No verification threshold was borrowed from another benchmark. There are no impostor identities, so false-match rates, verification accuracy and re-identification success cannot be estimated. A different recognition model, identity cohort, separate enrollment video or attacker may behave differently. Same-session original similarity is expected to be high and should not be generalized.

## Reproduce locally
```bash
python scripts/download_identity_models.py
python -m scripts.identity_demo
```
The model download helper uses the pinned tested official revision and records hashes. Compare hashes before future comparisons. Outputs under results/private/identity_demo contain probe_scores.csv, summary.csv, embedding_similarity.png and manifest.json. No embedding vectors or face crops are saved.

## Automatic detector improvement, separate experiment
`python -m scripts.validate_external --output results/private/external_ubfc_subject1_yunet --detector yunet` completed automatic per-frame detection on all 1547 subject1 frames. No manual search region was used. Results versus contact-PPG-derived spectral HR over two overlapping windows:

| Method | Original MAE | Blur MAE | Pixelation MAE |
|---|---:|---:|---:|
| POS | 0.643 BPM | 0.429 BPM | 1.072 BPM |
| CHROM | 0.643 BPM | 0.429 BPM | 0.857 BPM |

No non-mask >5-BPM errors or warnings occurred. This detector variant was introduced after inspecting the original shirt false-positive, so it is an exploratory engineering improvement on a known sample, not held-out validation. Default strict Haar behaviour and its failure remain reproducible. YuNet performs independent detections per frame, without temporal tracking/smoothing. Face geometry differs, so these numeric results need not equal the Haar-assisted results.
