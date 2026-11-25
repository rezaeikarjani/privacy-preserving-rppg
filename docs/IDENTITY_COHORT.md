# Small three-person closed-set linkability experiment

## Scope and protocol
Official research-shared UBFC Dataset2 subject1, subject3 and subject4 provide three anonymous dataset labels, not real-world names. This extends the earlier one-person SFace similarity demo. Settings were stored in configs/identity_cohort.json before subject3/4 embedding outcomes. Official pinned CPU YuNet/SFace models, threshold 0.9 and hashes are unchanged.

For each identity, original frames at 5/10/15 s form a normalized average enrollment template; probes at 30/35/40 s come from the SAME recording. All nine enrollment/probe frames per role are different in time, but there is NO independent capture session. All three enrollment frames per identity were required; all nine original probes were detected successfully. Face transforms use earlier original Haar geometry, with subject1's explicitly manual target region and strict automatic Haar for subjects3/4. This mixed geometry and same-session setup limit comparison. No user-private face or additional unconsented identity is used.

Original landmarks align every transformed probe; this is an assisted, side-information condition, not transformed-only deployment. We separately attempt re-detection. No face images or embedding vectors are saved; only scores and anonymous labels remain local.

## Measurements
For each probe we compare cosine similarity to its own template (genuine) and both other templates (impostors). Unique rank1 means the genuine score is STRICTLY greater than both impostor scores; ties are not successes. There is no verification/rejection threshold: the closed-set system is forced to rank three candidates even when all scores are low. Rank1 is not calibrated recognition probability or a general-world re-identification rate.

| Variant | Mean genuine cosine | Mean strongest-impostor cosine | Unique own-template rank1 | Re-detection |
|---|---:|---:|---:|---:|
| Original | 0.983510 | 0.059498 | 9/9 | 9/9 |
| Blur | 0.125149 | 0.088520 | 6/9 | 0/9 |
| Pixelation | 0.201605 | 0.104585 | 8/9 | 0/9 |
| Black mask | 0.146126 | 0.051974 | 6/9 | 0/9 |

Lower genuine similarity did not consistently remove own-template ranking in this setup. The black-mask 6/9 result is an additional warning about forced ranking, residual context/geometry and model/gallery bias; it is NOT proof of facial biometric recovery through a mask. We have not isolated which nuisance information contributes. Detection failure does not imply anonymity, and missing end-to-end embedding scores are not zero.

Three correlated probes per identity and only two impostors per comparison cannot establish anonymity, false-match rates, reliable identity recognition, or population privacy. Different identities, models, image conditions and independent-session enrollment may change outcomes. Do not combine these geometry-assisted identity scores with the separate automatic YuNet rPPG run as a matched privacy/utility frontier.

## Reproduce
After the prescribed videos and the Haar/assisted tracking outputs exist:
```bash
python -m scripts.identity_cohort
python -m pytest -q
```
Outputs: results/private/identity_cohort/probe_scores.csv, summary.csv, by_subject.csv, cohort_linkability.png and manifest.json. Original one-person results remain separately reproducible using scripts/identity_demo.py. This cohort is a descriptive model-specific experiment, not proven anonymization or independent-session privacy validation.
