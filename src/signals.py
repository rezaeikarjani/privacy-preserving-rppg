"""Independent classical implementations; optional upstream backends never silently fall back."""
import importlib
import numpy as np
from scipy import signal


def classical(rgb, fps, method="POS", backend="numpy"):
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("RGB must be finite [frames,3]")
    if backend == "pyvhr":
        module = importlib.import_module("pyVHR.BVP.methods")
        fn = getattr(module, "cpu_" + method)
        kwargs = {"fps": float(fps)} if method == "POS" else {}
        return np.asarray(fn(rgb.T[None].astype(np.float32), **kwargs)).reshape(-1)
    if backend != "numpy":
        raise ValueError("Unknown backend")
    if method not in ("POS", "CHROM"):
        raise ValueError("Unknown method")
    # 1.6-second temporal normalization and overlap-add.
    length = int(np.ceil(1.6 * fps))
    if len(rgb) < length:
        raise ValueError("Video too short for extraction")
    out, weights = np.zeros(len(rgb)), np.zeros(len(rgb))
    for start in range(len(rgb) - length + 1):
        c = rgb[start:start+length]
        if np.any(c.mean(axis=0) < 1e-6):
            continue
        c = c / c.mean(axis=0) - 1
        if method == "POS":
            x, y = c[:,1] - c[:,2], -2*c[:,0] + c[:,1] + c[:,2]
            h = x + np.std(x) / max(np.std(y), 1e-12) * y
        else:
            x, y = 3*c[:,0] - 2*c[:,1], 1.5*c[:,0] + c[:,1] - 1.5*c[:,2]
            h = x - np.std(x) / max(np.std(y), 1e-12) * y
        out[start:start+length] += h - h.mean()
        weights[start:start+length] += 1
    return out / np.maximum(weights, 1)


def clean(bvp, fps, band=(0.7, 3.0)):
    if not 0 < band[0] < band[1] < fps/2:
        raise ValueError("Invalid band or frame rate")
    if len(bvp) < fps*8 or not np.isfinite(bvp).all():
        raise ValueError("Need >=8 seconds of finite signal")
    bvp = signal.detrend(bvp)
    bvp = signal.sosfiltfilt(signal.butter(3, band, btype="bandpass", fs=fps, output="sos"), bvp)
    if bvp.std() < 1e-10:
        return np.zeros_like(bvp)
    return (bvp-bvp.mean())/bvp.std()


def spectrum(bvp, fps):
    return signal.welch(bvp, fs=fps, nperseg=min(len(bvp), int(10*fps)), nfft=4096)


def hr(bvp, fps, band=(0.7,3.0)):
    if np.std(bvp) < 1e-10:
        return float("nan")
    f, power = spectrum(bvp, fps)
    mask = (f>=band[0]) & (f<=band[1])
    return float(60*f[mask][np.argmax(power[mask])])


def correlation(a, b):
    if min(np.std(a),np.std(b)) < 1e-10:
        return float("nan")
    return float(np.corrcoef(a,b)[0,1])


def reference_band_ratio(bvp, fps, reference_hr, band=(0.7,3.0)):
    """dB power near reference fundamental (+/-0.1 Hz) / remaining cardiac band.
    Not physiological SNR: reference may be the original estimate, not GT.
    """
    if np.std(bvp) < 1e-10 or not np.isfinite(reference_hr):
        return float("nan")
    f,p = spectrum(bvp,fps)
    valid = (f>=band[0]) & (f<=band[1])
    target = valid & (np.abs(f-reference_hr/60)<=0.1)
    return float(10*np.log10((p[target].sum()+1e-20)/(p[valid & ~target].sum()+1e-20)))
