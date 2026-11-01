"""Windowing and feature extraction.

Each 8 s clip is cut into 2 s windows with 50% overlap, and every window is
reduced to a single observation vector combining time-domain and
frequency-domain descriptors. Those vectors are the observations X that the
HMM emits from its hidden activity states.

Shapes, for the record: a 400-sample clip at 100 samples per window and a
50-sample hop gives 7 windows, so 13 clips x 7 = 91 windows per activity.

The feature families follow standard practice in accelerometer-based activity
recognition (Bao & Intille, 2004; Figo et al., 2010), and the 2 s window with
50% overlap follows the same convention as the UCI Human Activity Recognition
benchmark, which uses 2.56 s windows at 50 Hz (Anguita et al., 2013).

Numerical care
--------------
The Still class is the sensor noise floor - gyroscope magnitude standard
deviation rounds to 0.000. Correlation, skewness, kurtosis and spectral
entropy are all undefined on a constant signal, so every one of them is
guarded by EPS and returns a neutral 0.0 rather than a NaN that would
silently poison the covariance estimates downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (CHANNELS, HOP_SAMPLES, STATE_INDEX, TARGET_HZ,
                    WINDOW_SAMPLES)

# References
# -----------
# Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013).
#     A public domain dataset for human activity recognition using
#     smartphones. ESANN 2013, 437-442.
# Bao, L., & Intille, S. S. (2004). Activity recognition from user-annotated
#     acceleration data. Pervasive Computing, LNCS 3001, 1-17.
# Figo, D., Diniz, P. C., Ferreira, D. R., & Cardoso, J. M. P. (2010).
#     Preprocessing techniques for context recognition from accelerometer
#     data. Personal and Ubiquitous Computing, 14(7), 645-662.

EPS = 1e-12

# Band edges in Hz. At 50 Hz with a 100-sample window the FFT resolution is
# exactly 0.5 Hz, so these edges land on bin boundaries. The bands start at
# 0.5 Hz rather than 0: the DC bin is always excluded (the signal is
# mean-removed first), so a 0-0.5 Hz band would contain no bins at all and be
# identically zero for every window.
#
# The edges follow the physics of the four activities: 0.5-1.5 Hz is postural
# sway, 1.5-3 Hz covers walking cadence and its stride harmonic, 3-10 Hz
# captures jumping (measured dominant frequency 3.57 Hz) and landing impact,
# and 10-25 Hz runs up to Nyquist and is mostly sensor noise.
BANDS = [(0.5, 1.5), (1.5, 3.0), (3.0, 10.0), (10.0, 25.0)]

AXIS_PAIRS = [("acc_x", "acc_y"), ("acc_x", "acc_z"), ("acc_y", "acc_z"),
              ("gyr_x", "gyr_y"), ("gyr_x", "gyr_z"), ("gyr_y", "gyr_z")]


def iter_windows(clip: pd.DataFrame):
    """Yield (window_index, start_sample, window_frame) for one recording.

    Only full-length windows are emitted; a trailing partial window would have
    a different sample count and therefore incomparable FFT bins.
    """
    n = len(clip)
    for w, start in enumerate(range(0, n - WINDOW_SAMPLES + 1, HOP_SAMPLES)):
        yield w, start, clip.iloc[start:start + WINDOW_SAMPLES]


def time_features(x: np.ndarray, name: str) -> dict:
    """Time-domain descriptors for one channel."""
    centred = x - x.mean()
    sd = x.std()
    q75, q25 = np.percentile(x, [75, 25])

    if sd > EPS:
        z = centred / sd
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3.0)
        # Zero crossings of the mean-removed signal.
        zcr = float(np.mean(np.diff(np.signbit(centred)) != 0))
    else:
        skew = kurt = zcr = 0.0

    return {
        f"{name}_mean": float(x.mean()),
        f"{name}_std": float(sd),
        f"{name}_var": float(x.var()),
        f"{name}_min": float(x.min()),
        f"{name}_max": float(x.max()),
        f"{name}_range": float(x.max() - x.min()),
        f"{name}_rms": float(np.sqrt(np.mean(x ** 2))),
        f"{name}_median": float(np.median(x)),
        f"{name}_iqr": float(q75 - q25),
        f"{name}_mad": float(np.mean(np.abs(centred))),
        f"{name}_zcr": zcr,
        f"{name}_skew": skew,
        f"{name}_kurt": kurt,
        f"{name}_jerk": float(np.mean(np.abs(np.diff(x))) * TARGET_HZ),
    }


def freq_features(x: np.ndarray, name: str, fs: float = TARGET_HZ) -> dict:
    """Frequency-domain descriptors for one channel.

    Hann-windowed rFFT of the mean-removed signal. The DC bin is dropped
    before the dominant frequency is picked, otherwise any residual offset
    wins trivially.
    """
    n = len(x)
    centred = (x - x.mean()) * np.hanning(n)
    spec = np.abs(np.fft.rfft(centred))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = spec ** 2
    total = power[1:].sum()

    if total > EPS:
        k = int(np.argmax(power[1:]) + 1)
        dom_hz = float(freqs[k])
        dom_mag = float(spec[k])
        p = power[1:] / total
        p = p[p > EPS]
        entropy = float(-np.sum(p * np.log(p)) / np.log(len(power) - 1))
        centroid = float(np.sum(freqs[1:] * power[1:]) / total)
    else:
        dom_hz = dom_mag = entropy = centroid = 0.0

    out = {
        f"{name}_dom_hz": dom_hz,
        f"{name}_dom_mag": dom_mag,
        f"{name}_energy": float(total / n),
        f"{name}_entropy": entropy,
        f"{name}_centroid": centroid,
    }
    for lo, hi in BANDS:
        band = (freqs >= lo) & (freqs < hi)
        band[0] = False                      # never count DC
        share = float(power[band].sum() / total) if total > EPS else 0.0
        out[f"{name}_band_{lo:g}_{hi:g}"] = share
    return out


def sma(window: pd.DataFrame, prefix: str) -> float:
    """Signal magnitude area: (1/T) * sum(|x| + |y| + |z|) for one sensor.

    Named explicitly because the brief calls for it by name.
    """
    axes = [f"{prefix}_{a}" for a in "xyz"]
    total = sum(window[c].abs().sum() for c in axes)
    return float(total / len(window))


def axis_correlations(window: pd.DataFrame) -> dict:
    """Pearson r for the three axis pairs on each of the two sensors.

    Returns 0.0 where a channel is effectively constant, which is the normal
    case for Still rather than an error.
    """
    out = {}
    for a, b in AXIS_PAIRS:
        xa = window[a].to_numpy()
        xb = window[b].to_numpy()
        if xa.std() > EPS and xb.std() > EPS:
            out[f"corr_{a}_{b}"] = float(np.corrcoef(xa, xb)[0, 1])
        else:
            out[f"corr_{a}_{b}"] = 0.0
    return out


def window_features(window: pd.DataFrame) -> dict:
    """Full feature vector for one window."""
    feats: dict = {}
    for ch in CHANNELS:
        x = window[ch].to_numpy()
        feats.update(time_features(x, ch))
        feats.update(freq_features(x, ch))
    feats["acc_sma"] = sma(window, "acc")
    feats["gyr_sma"] = sma(window, "gyr")
    feats.update(axis_correlations(window))
    return feats


def build_feature_table(clean: pd.DataFrame) -> pd.DataFrame:
    """Window every recording and stack the feature vectors.

    One row per window. The identifying columns come first so that splits can
    be made BY RECORDING - splitting by window would put overlapping halves
    of the same clip on both sides and leak badly.
    """
    rows = []
    for rec, clip in clean.groupby("recording", sort=True):
        clip = clip.sort_values("time_s").reset_index(drop=True)
        activity = clip["activity"].iloc[0]
        for w, start, window in iter_windows(clip):
            row = {
                "recording": rec,
                "activity": activity,
                "state": STATE_INDEX[activity],
                "window_index": w,
                "t_start": float(window["time_s"].iloc[0]),
            }
            row.update(window_features(window))
            rows.append(row)

    table = pd.DataFrame(rows)
    bad = table.isna().sum()
    if bad.any():
        raise ValueError(f"NaNs produced in: {bad[bad > 0].to_dict()}")
    return table


def build_continuous_feature_table(samples: pd.DataFrame) -> pd.DataFrame:
    """Window the continuous circuit recordings.

    Identical windowing to the isolated clips, so the features are directly
    comparable, but a circuit window can span a transition. Each window gets
    the majority label of its samples plus a `near_boundary` flag that is True
    if ANY of its samples fall inside the annotation guard band. Metrics are
    computed on the clean windows; the full ordered sequence is still used for
    decoding, because dropping windows would break the temporal continuity the
    HMM depends on.
    """
    rows = []
    for rec, clip in samples.groupby("recording", sort=True):
        clip = clip.sort_values("time_s").reset_index(drop=True)
        for w, start, window in iter_windows(clip):
            labels = window["activity"].value_counts()
            activity = str(labels.index[0])
            row = {
                "recording": rec,
                "activity": activity,
                "state": STATE_INDEX[activity],
                "window_index": w,
                "t_start": float(window["time_s"].iloc[0]),
                "near_boundary": bool(window["near_boundary"].any()),
                "label_purity": float(labels.iloc[0] / labels.sum()),
            }
            row.update(window_features(window))
            rows.append(row)

    table = pd.DataFrame(rows)
    bad = table.isna().sum()
    if bad.any():
        raise ValueError(f"NaNs produced in: {bad[bad > 0].to_dict()}")
    return table


def feature_columns(table: pd.DataFrame) -> list[str]:
    """The numeric feature columns, excluding the identifying metadata."""
    meta = {"recording", "activity", "state", "window_index", "t_start",
            "near_boundary", "label_purity"}
    return [c for c in table.columns if c not in meta]


def prune_features(table: pd.DataFrame, columns: list[str] | None = None,
                   max_corr: float = 0.95, verbose: bool = False) -> list[str]:
    """Drop constant and near-duplicate features, returning the keepers.

    Raw extraction produces 192 descriptors, many of which measure the same
    thing twice - std, var and rms of a zero-mean channel are near-perfectly
    correlated, for instance. Feeding all of them to PCA spreads variance
    across redundant directions and inflates how many components are needed.
    Dropped here rather than relied on PCA to handle.

    MUST be called on the training split only; apply the returned column list
    unchanged to any held-out data.
    """
    cols = columns or feature_columns(table)
    X = table[cols]

    constant = [c for c in cols if X[c].std() < EPS]
    kept = [c for c in cols if c not in constant]

    corr = X[kept].corr().abs().to_numpy()
    drop_redundant = set()
    for i in range(len(kept)):
        if kept[i] in drop_redundant:
            continue
        for j in range(i + 1, len(kept)):
            if corr[i, j] > max_corr:
                drop_redundant.add(kept[j])

    final = [c for c in kept if c not in drop_redundant]
    if verbose:
        print(f"pruned {len(cols)} -> {len(final)} features "
              f"({len(constant)} constant, {len(drop_redundant)} redundant "
              f"at |r| > {max_corr})")
    return final


if __name__ == "__main__":
    from config import PROCESSED
    from data import load_circuits, load_clean, load_unseen

    PROCESSED.mkdir(parents=True, exist_ok=True)

    for name, builder in (("windows_train.csv",
                           lambda: build_feature_table(load_clean())),
                          ("windows_unseen.csv",
                           lambda: build_feature_table(load_unseen()))):
        table = builder()
        table.to_csv(PROCESSED / name, index=False)
        print(f"{name}: {len(table)} windows x "
              f"{len(feature_columns(table))} features")
        print(table.groupby("activity").agg(
            windows=("window_index", "count"),
            recordings=("recording", "nunique")).to_string())
        print()

    samples, _ = load_circuits()
    circuits = build_continuous_feature_table(samples)
    circuits.to_csv(PROCESSED / "windows_circuits.csv", index=False)
    clean = (~circuits["near_boundary"]).sum()
    print(f"windows_circuits.csv: {len(circuits)} windows "
          f"({clean} outside the boundary guard, "
          f"{len(circuits) - clean} inside)")
    print(circuits.groupby("activity").agg(
        windows=("window_index", "count"),
        clean=("near_boundary", lambda s: int((~s).sum())),
        mean_purity=("label_purity", "mean")).to_string())
