"""Ingest Sensor Logger recordings into a clean, uniformly sampled dataset.

Pipeline
--------
raw .zip (Sensor Logger export)
  -> data/raw/<Recording>/{Accelerometer,Gyroscope,Metadata}.csv   (provenance)
  -> data/clean/<Activity>_<nn>.csv                                (8 s @ 50 Hz)
  -> data/processed/windows_*.csv                                  (features)

Why the resampling step exists
------------------------------
Sensor Logger writes each sensor to its own file with its own nanosecond
clock. The requested rate was 50 Hz but the device delivered ~51.34 Hz, and
the accelerometer and gyroscope rows are not aligned with each other. Both
streams are therefore linearly interpolated onto a single uniform 50 Hz grid
before anything else happens, which is what makes every clip exactly 400
samples and every window exactly 100 samples.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import (ANNOTATION_OVERRIDES, BOUNDARY_GUARD_S,
                    CIRCUIT_ZIP_DIR, CLEAN, CLEAN_CIRCUIT, CLEAN_UNSEEN,
                    CLIP_SAMPLES, CLIP_SECONDS, PROCESSED, RAW, RAW_CIRCUIT,
                    RAW_UNSEEN, RAW_ZIP_DIR, TARGET_HZ, UNSEEN_ZIP_DIR)

SENSOR_FILES = ("Accelerometer.csv", "Gyroscope.csv", "Metadata.csv")
# Circuits additionally carry the in-app annotation taps.
CIRCUIT_FILES = SENSOR_FILES + ("Annotation.csv",)


def recording_name(zip_path: Path) -> str:
    """'Walking_01-2026-10-30_22-39-10.zip' -> 'Walking_01'."""
    return zip_path.name.split("-")[0]


def activity_of(recording: str) -> str:
    """'Walking_01' -> 'Walking'."""
    return recording.split("_")[0]


def extract_raw(zip_path: Path, dest_root: Path = RAW,
                members: tuple[str, ...] = SENSOR_FILES) -> Path:
    """Copy the named CSVs out of the zip into their own folder."""
    rec = recording_name(zip_path)
    dest = dest_root / rec
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        available = set(z.namelist())
        for name in members:
            if name in available:
                (dest / name).write_bytes(z.read(name))
    return dest


def _read_sensor(folder: Path, name: str, prefix: str) -> pd.DataFrame:
    """Load one sensor CSV, keeping the elapsed-time column and the three axes.

    Sensor Logger writes its axis columns in the order z, y, x - they are read
    by name here so the ordering can never silently bite us.
    """
    df = pd.read_csv(folder / name)
    out = pd.DataFrame({
        "t": df["seconds_elapsed"].to_numpy(dtype=float),
        f"{prefix}_x": df["x"].to_numpy(dtype=float),
        f"{prefix}_y": df["y"].to_numpy(dtype=float),
        f"{prefix}_z": df["z"].to_numpy(dtype=float),
    })
    return out.sort_values("t").reset_index(drop=True)


def native_rate(folder: Path, name: str) -> float:
    """Achieved sampling rate of one sensor file, in Hz."""
    t = pd.read_csv(folder / name)["seconds_elapsed"].to_numpy(dtype=float)
    return (len(t) - 1) / (t[-1] - t[0])


def clean_recording(folder: Path) -> pd.DataFrame:
    """Trim to the central CLIP_SECONDS and resample onto a uniform grid.

    Returns a frame of exactly CLIP_SAMPLES rows with both sensors aligned on
    a shared time base starting at 0.
    """
    acc = _read_sensor(folder, "Accelerometer.csv", "acc")
    gyr = _read_sensor(folder, "Gyroscope.csv", "gyr")

    # Central window: drops the lead-in/lead-out where the hand is moving to
    # the screen, and brings every clip inside the 5-10 s band the brief asks for.
    duration = min(acc["t"].iloc[-1], gyr["t"].iloc[-1])
    if duration < CLIP_SECONDS:
        raise ValueError(f"{folder.name}: only {duration:.2f} s available, "
                         f"need {CLIP_SECONDS}")
    start = (duration - CLIP_SECONDS) / 2.0
    grid = start + np.arange(CLIP_SAMPLES) / TARGET_HZ

    out = {"time_s": np.round(grid - start, 6)}
    for src, prefix in ((acc, "acc"), (gyr, "gyr")):
        for axis in "xyz":
            col = f"{prefix}_{axis}"
            out[col] = np.interp(grid, src["t"].to_numpy(), src[col].to_numpy())

    df = pd.DataFrame(out)
    df["acc_mag"] = np.sqrt(df.acc_x**2 + df.acc_y**2 + df.acc_z**2)
    df["gyr_mag"] = np.sqrt(df.gyr_x**2 + df.gyr_y**2 + df.gyr_z**2)
    df.insert(0, "recording", folder.name)
    df.insert(1, "activity", activity_of(folder.name))
    return df


def build(zip_dir: Path = RAW_ZIP_DIR, raw_dest: Path = RAW,
          clean_dest: Path = CLEAN, manifest_name: str = "manifest.csv",
          verbose: bool = True) -> pd.DataFrame:
    """Run the raw -> clean stage over one folder of isolated-activity zips.

    Used for both the training session and the held-out unseen session; the
    two differ only in which folder they read and write.
    """
    zips = sorted(Path(zip_dir).glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"no Sensor Logger zips under {zip_dir}")

    clean_dest.mkdir(parents=True, exist_ok=True)
    manifest, frames = [], []

    for zp in zips:
        rec = recording_name(zp)
        folder = extract_raw(zp, raw_dest)
        acc_hz = native_rate(folder, "Accelerometer.csv")
        gyr_hz = native_rate(folder, "Gyroscope.csv")
        raw_dur = pd.read_csv(folder / "Accelerometer.csv")["seconds_elapsed"].iloc[-1]

        df = clean_recording(folder)
        df.to_csv(clean_dest / f"{rec}.csv", index=False)
        frames.append(df)

        manifest.append({
            "recording": rec,
            "activity": activity_of(rec),
            "raw_duration_s": round(float(raw_dur), 3),
            "clean_duration_s": CLIP_SECONDS,
            "native_acc_hz": round(acc_hz, 3),
            "native_gyr_hz": round(gyr_hz, 3),
            "resampled_hz": TARGET_HZ,
            "n_samples": CLIP_SAMPLES,
            "source_zip": zp.name,
        })

    PROCESSED.mkdir(parents=True, exist_ok=True)
    man = pd.DataFrame(manifest)
    man.to_csv(PROCESSED / manifest_name, index=False)

    if verbose:
        print(f"{len(man)} recordings -> {clean_dest}")
        print(man.groupby("activity").agg(
            files=("recording", "count"),
            total_clean_s=("clean_duration_s", "sum"),
            mean_native_hz=("native_acc_hz", "mean"),
        ).to_string())

    return pd.concat(frames, ignore_index=True)


def build_unseen(verbose: bool = True) -> pd.DataFrame:
    """Ingest the held-out session recorded in a different room with the
    phone in a pocket. Identical processing to the training session, so any
    performance difference is attributable to the data and not the pipeline.
    """
    return build(zip_dir=UNSEEN_ZIP_DIR, raw_dest=RAW_UNSEEN,
                 clean_dest=CLEAN_UNSEEN, manifest_name="manifest_unseen.csv",
                 verbose=verbose)


# --- continuous circuit recordings ---------------------------------------
def resample_full(folder: Path) -> pd.DataFrame:
    """Resample a whole recording onto a uniform grid, with no trimming.

    Circuits are kept at full length because their value is the transitions
    between activities, not a clean single-activity excerpt.
    """
    acc = _read_sensor(folder, "Accelerometer.csv", "acc")
    gyr = _read_sensor(folder, "Gyroscope.csv", "gyr")

    start = max(acc["t"].iloc[0], gyr["t"].iloc[0])
    end = min(acc["t"].iloc[-1], gyr["t"].iloc[-1])
    n = int(np.floor((end - start) * TARGET_HZ)) + 1
    grid = start + np.arange(n) / TARGET_HZ

    out = {"time_s": np.round(grid - start, 6)}
    for src, prefix in ((acc, "acc"), (gyr, "gyr")):
        for axis in "xyz":
            col = f"{prefix}_{axis}"
            out[col] = np.interp(grid, src["t"].to_numpy(), src[col].to_numpy())

    df = pd.DataFrame(out)
    df["acc_mag"] = np.sqrt(df.acc_x**2 + df.acc_y**2 + df.acc_z**2)
    df["gyr_mag"] = np.sqrt(df.gyr_x**2 + df.gyr_y**2 + df.gyr_z**2)
    df.insert(0, "recording", folder.name)
    return df


def circuit_segments(folder: Path, duration: float) -> pd.DataFrame:
    """Turn annotation taps into labelled (start_s, end_s, activity) segments.

    Annotation.csv uses a monotonic boot clock while the sensor files use Unix
    epoch, so the two cannot be differenced directly. Only the intervals
    between taps are meaningful, and the first tap is anchored to the start of
    the recording - the protocol was to tap immediately on pressing record.

    Any correction listed in ANNOTATION_OVERRIDES is applied here and flagged
    in the returned frame so it can be reported rather than hidden.
    """
    ann = pd.read_csv(folder / "Annotation.csv")
    if ann.empty:
        raise ValueError(f"{folder.name}: Annotation.csv is empty")

    taps = ann["time"].to_numpy(dtype=float) / 1e9
    taps = taps - taps[0]                       # anchor the first tap at t=0
    labels = [str(s) for s in ann["text"]]

    corrected = set()
    for idx, new_t in ANNOTATION_OVERRIDES.get(folder.name, []):
        if 0 <= idx < len(taps):
            taps[idx] = new_t
            corrected.add(idx)

    order = np.argsort(taps, kind="stable")
    taps = taps[order]
    labels = [labels[i] for i in order]
    corrected = {int(np.where(order == i)[0][0]) for i in corrected}

    bounds = np.append(taps, duration)
    rows = []
    for i, name in enumerate(labels):
        rows.append({
            "recording": folder.name,
            "segment": i,
            "activity": name,
            "start_s": float(bounds[i]),
            "end_s": float(bounds[i + 1]),
            "duration_s": float(bounds[i + 1] - bounds[i]),
            "annotation_corrected": i in corrected,
        })
    return pd.DataFrame(rows)


def label_samples(clip: pd.DataFrame, segments: pd.DataFrame,
                  guard: float = BOUNDARY_GUARD_S) -> pd.DataFrame:
    """Attach a per-sample activity label plus a boundary guard flag.

    `near_boundary` marks samples within `guard` seconds of any annotated
    transition. Those samples carry a label that may be wrong by the tap's
    reaction error, so downstream code drops the windows containing them when
    computing metrics instead of trusting them.
    """
    t = clip["time_s"].to_numpy()
    activity = np.full(len(t), "", dtype=object)
    near = np.zeros(len(t), dtype=bool)

    for _, s in segments.iterrows():
        m = (t >= s.start_s) & (t < s.end_s)
        activity[m] = s.activity

    interior = segments["start_s"].to_numpy()[1:]      # skip the recording start
    for b in interior:
        near |= np.abs(t - b) < guard

    out = clip.copy()
    out["activity"] = activity
    out["near_boundary"] = near
    # Samples before the first tap or after the last have no label.
    return out[out["activity"] != ""].reset_index(drop=True)


def build_circuits(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest the continuous circuit recordings and their annotations.

    Returns (samples, segments).
    """
    zips = sorted(Path(CIRCUIT_ZIP_DIR).glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"no circuit zips under {CIRCUIT_ZIP_DIR}")

    CLEAN_CIRCUIT.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sample_frames, segment_frames = [], []

    for zp in zips:
        rec = recording_name(zp)
        folder = extract_raw(zp, RAW_CIRCUIT, CIRCUIT_FILES)
        clip = resample_full(folder)
        segs = circuit_segments(folder, duration=float(clip["time_s"].iloc[-1]))
        labelled = label_samples(clip, segs)

        labelled.to_csv(CLEAN_CIRCUIT / f"{rec}.csv", index=False)
        segs.to_csv(CLEAN_CIRCUIT / f"{rec}_segments.csv", index=False)
        sample_frames.append(labelled)
        segment_frames.append(segs)

        if verbose:
            kept = (~labelled["near_boundary"]).sum()
            print(f"{rec}: {clip['time_s'].iloc[-1]:.1f} s, {len(segs)} segments, "
                  f"{len(labelled)} labelled samples "
                  f"({kept} outside the {BOUNDARY_GUARD_S:g} s guard)")

    samples = pd.concat(sample_frames, ignore_index=True)
    segments = pd.concat(segment_frames, ignore_index=True)
    segments.to_csv(PROCESSED / "circuit_segments.csv", index=False)

    if verbose:
        print("\nsegments:")
        print(segments[["recording", "segment", "activity", "start_s", "end_s",
                        "duration_s", "annotation_corrected"]]
              .to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
    return samples, segments


# --- loading --------------------------------------------------------------
def _load_dir(folder: Path, what: str) -> pd.DataFrame:
    files = sorted(p for p in folder.glob("*.csv")
                   if not p.stem.endswith("_segments"))
    if not files:
        raise FileNotFoundError(f"no {what} CSVs in {folder} - run the build first")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def load_clean() -> pd.DataFrame:
    """Read every cleaned training recording back as one long frame."""
    return _load_dir(CLEAN, "cleaned")


def load_unseen() -> pd.DataFrame:
    """Read the cleaned held-out session."""
    return _load_dir(CLEAN_UNSEEN, "unseen")


def load_circuits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the labelled circuit samples and their segment tables."""
    samples = _load_dir(CLEAN_CIRCUIT, "circuit")
    segments = pd.read_csv(PROCESSED / "circuit_segments.csv")
    return samples, segments


if __name__ == "__main__":
    print("=== training session ===")
    build()
    print("\n=== unseen session ===")
    build_unseen()
    print("\n=== circuits ===")
    build_circuits()
