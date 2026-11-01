# Dataset

Self-collected smartphone motion data for four human activity states, recorded
for a Hidden Markov Model activity-recognition task.

## Collection setup

| | |
|---|---|
| App | Sensor Logger v1.68 (Android) |
| Device | Xiaomi `2312DRA50G` |
| Platform | Android, platform version 36 |
| Sensors recorded | Accelerometer, Gyroscope (also TotalAcceleration and the uncalibrated variants) |
| Requested sampling rate | **50 Hz** (`sampleRateMs = 20`) |
| Achieved sampling rate | **51.33 Hz** mean across all 52 files (range 51.327–51.341 Hz) |
| Max inter-sample gap | 0.019 s — no dropouts in any file |
| Session date | 2026-10-30, 22:05–23:03 (Africa/Kigali) |
| Timezone | Africa/Kigali |
| Participant | 1 (individual project) |

### Phone placement

| Activity | Placement |
|---|---|
| Still | Phone resting on a flat, stationary surface |
| Standing | Held in hand at waist level, screen facing outward, body upright and stationary |
| Walking | Held in hand at waist level, consistent indoor walking pace |
| Jumping | Held in hand at waist level, continuous two-footed jumps |

## Contents

| Activity | Files | Clean seconds | Raw file length |
|---|---|---|---|
| Still | 13 | 104.0 | 10.33–11.59 s |
| Standing | 13 | 104.0 | 9.97–12.00 s |
| Walking | 13 | 104.0 | 9.27–11.77 s |
| Jumping | 13 | 104.0 | 10.28–11.38 s |
| **Total** | **52** | **416.0** | |

## Directory layout

```
data/
├── raw/<Recording>/          Untouched Sensor Logger export per recording
│   ├── Accelerometer.csv     time (ns), seconds_elapsed, z, y, x
│   ├── Gyroscope.csv         time (ns), seconds_elapsed, z, y, x
│   └── Metadata.csv          device, rate, app version, epoch time
├── clean/<Activity>_<nn>.csv Trimmed + resampled analysis files (below)
└── processed/
    ├── manifest.csv          Per-recording provenance and native rates
    └── windows_*.csv         Windowed feature table (built in the notebook)
```

## Cleaning applied (`src/data.py`)

1. **Central 8.0 s retained.** Raw recordings ran 9.27–12.00 s. The rubric caps
   a file at 10 s, and the first and last ~1.5 s are contaminated by the hand
   reaching for the screen — measured head/tail standard deviation is 1.3–1.8×
   the middle of the clip on Standing and Walking. Keeping the central 8 s
   satisfies the 5–10 s band and gives 13 × 8 = 104 s per activity against a
   90 s floor.
2. **Both sensors resampled onto one uniform 50 Hz grid.** Sensor Logger writes
   each sensor to its own file on its own nanosecond clock; the requested 50 Hz
   was delivered as ~51.33 Hz with jitter, and accelerometer and gyroscope rows
   are not mutually aligned. Both streams are linearly interpolated onto a
   single shared grid at exactly 50 Hz, so every clean file is **400 samples**
   and every 2 s analysis window is **exactly 100 samples** with identical FFT
   bins.
3. **Magnitude channels derived.** `acc_mag` and `gyr_mag` as the Euclidean norm
   of the three axes, which are invariant to how the phone is rotated in the
   hand.

### Clean file schema

| Column | Units | Notes |
|---|---|---|
| `recording` | — | e.g. `Walking_07` |
| `activity` | — | `Still` \| `Standing` \| `Walking` \| `Jumping` |
| `time_s` | s | 0.00 → 7.98 in 0.02 s steps (400 rows) |
| `acc_x`, `acc_y`, `acc_z` | m/s² | Calibrated accelerometer, **gravity removed** |
| `gyr_x`, `gyr_y`, `gyr_z` | rad/s | Calibrated gyroscope |
| `acc_mag`, `gyr_mag` | m/s², rad/s | Euclidean norm of the three axes |

> `Accelerometer.csv` on this device reports linear acceleration with gravity
> already removed (per-axis means ≈ 0 at rest). `TotalAcceleration.csv` in
> `raw/` retains gravity and is kept for any gravity-orientation features.

## Class separability

Mean `acc_mag` standard deviation over the clean 8 s, by activity:

| Activity | acc_mag std | gyr_mag std | Dominant frequency |
|---|---|---|---|
| Still | 0.003 | 0.000 | — (noise floor) |
| Standing | 0.048 | 0.013 | 1.40 Hz |
| Walking | 1.258 | 0.800 | 0.96 Hz |
| Jumping | 9.592 | 1.132 | 3.57 Hz |

The four classes occupy non-overlapping amplitude bands, and no file falls
outside its own class's range — there are no mislabelled recordings.

## Reproducing

```bash
python src/data.py      # raw zips -> data/raw/ + data/clean/ + manifest.csv
```
