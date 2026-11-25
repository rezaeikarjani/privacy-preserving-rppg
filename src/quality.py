"""Exploratory spectral ambiguity indicator, not an accuracy/confidence probability.
No reference HR or identity information is accepted. Original estimates are retained.
"""
import numpy as np
from scipy.signal import find_peaks
from .signals import spectrum


def spectral_quality(bvp, fps, band=(0.7, 3.0), ratio_threshold=1.2,
                     min_separation_hz=0.15, edge_margin_hz=0.05):
    """Flag similar, distinct PSD peaks or a dominant peak near the search edge.

    Ratio = strongest PSD peak / strongest distinct competing PSD peak.
    min_separation_hz avoids counting adjacent zero-padded FFT bins as rivals.
    Fixed heuristic choices are exploratory and require independent validation.
    Harmonics may trigger a flag; a single strong artifact may evade it.
    """
    x = np.asarray(bvp, dtype=float)
    if not np.isfinite(fps) or not 0 < band[0] < band[1] < fps/2:
        raise ValueError('Invalid sampling frequency or band')
    if ratio_threshold <= 1 or min_separation_hz <= 0 or edge_margin_hz < 0:
        raise ValueError('Invalid quality settings')
    empty = dict(valid=False, flagged=True, ambiguous=False, near_band_edge=False,
                 status='invalid_signal', dominant_bpm=float('nan'),
                 competing_bpm=float('nan'), peak_ratio=float('nan'))
    if x.ndim != 1 or len(x) < int(8*fps) or not np.isfinite(x).all() or np.std(x) < 1e-10:
        return empty
    f, p = spectrum(x, fps)
    take=(f >= band[0]) & (f <= band[1]); f,p=f[take],p[take]
    top=int(np.argmax(p))
    # Match the baseline maximum-bin choice, including cardiac-band boundaries.
    peaks,_=find_peaks(np.r_[-np.inf,p,-np.inf]); peaks=peaks-1
    competitors=[int(i) for i in peaks if abs(f[i]-f[top]) >= min_separation_hz]
    rival=max(competitors,key=lambda i:p[i]) if competitors else None
    ratio=float(p[top]/p[rival]) if rival is not None and p[rival]>0 else float('inf')
    ambiguous=ratio < ratio_threshold
    edge=bool(min(f[top]-band[0],band[1]-f[top]) <= edge_margin_hz)
    status='band_edge_peak' if edge else ('competing_peaks' if ambiguous else 'no_ambiguity_detected')
    return dict(valid=True, flagged=bool(ambiguous or edge), ambiguous=bool(ambiguous),
                near_band_edge=edge, status=status, dominant_bpm=float(60*f[top]),
                competing_bpm=float(60*f[rival]) if rival is not None else float('nan'),
                peak_ratio=ratio)
