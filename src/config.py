"""Project-wide constants.

Every number here is a decision defended in the report; change them in one
place so the notebook, the figures and the write-up can never disagree.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Location of the original Sensor Logger exports. These are the .zip files as
# the app produces them, which are bulky and sit outside the repository; the
# extracted CSVs under data/raw are committed instead, so the repository is
# fully reproducible from data/ onward without them.
#
# Set SENSOR_DATASET_DIR to re-run the ingest stage from the original exports;
# otherwise it falls back to data/sensor_exports inside the repository.
RAW_ZIP_DIR = Path(os.environ.get("SENSOR_DATASET_DIR",
                                  ROOT / "data" / "sensor_exports"))
UNSEEN_ZIP_DIR = RAW_ZIP_DIR / "unseen"
CIRCUIT_ZIP_DIR = RAW_ZIP_DIR / "circuits"

DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
RAW_UNSEEN = DATA / "raw_unseen"
CLEAN_UNSEEN = DATA / "clean_unseen"
RAW_CIRCUIT = DATA / "raw_circuits"
CLEAN_CIRCUIT = DATA / "clean_circuits"
PROCESSED = DATA / "processed"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

# --- Sampling -------------------------------------------------------------
# Sensor Logger was configured at sampleRateMs=20 (50 Hz) for both the
# accelerometer and the gyroscope. The device actually delivered ~51.34 Hz
# with slight jitter, and the two sensors are written to separate files on
# separate clocks. We therefore resample both onto one uniform grid at
# exactly TARGET_HZ so that every window has an identical sample count.
REQUESTED_HZ = 50.0
TARGET_HZ = 50.0

# --- Trimming -------------------------------------------------------------
# Raw recordings run 9.3-12.0 s. The brief asks for clips of 5-10 s, and the
# first/last ~1.5 s are contaminated by the hand reaching for the screen
# (head/tail std measures 1.3-1.8x the middle on Standing and Walking).
# Keeping the CENTRAL 8 s satisfies both: 5 s <= 8 s <= 10 s, and
# 13 files x 8 s = 104 s per activity against the 90 s floor.
CLIP_SECONDS = 8.0
CLIP_SAMPLES = int(CLIP_SECONDS * TARGET_HZ)   # 400

# --- Windowing ------------------------------------------------------------
# 2.0 s at 50 Hz = exactly 100 samples, so FFT bins line up across every
# window. 2 s spans 3-4 walking gait cycles and gives 1/2s = 0.5 Hz spectral
# resolution, enough to separate walking (~1-2 Hz) from jumping (~3.6 Hz).
# A 1.0 s hop (50% overlap) yields 7 windows per 8 s clip -> 91 per activity.
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * TARGET_HZ)   # 100
HOP_SECONDS = 1.0
HOP_SAMPLES = int(HOP_SECONDS * TARGET_HZ)         # 50

# --- States ---------------------------------------------------------------
ACTIVITIES = ["Still", "Standing", "Walking", "Jumping"]
STATE_INDEX = {a: i for i, a in enumerate(ACTIVITIES)}

# --- Dimensionality -------------------------------------------------------
# Raw extraction yields 192 descriptors; pruning constant and |r| > 0.95
# redundant ones leaves ~121. That is far too many emission parameters for
# 364 windows, so the features are standardised and projected with PCA.
#
# The component count is NOT chosen by variance share. After pruning, 90% of
# variance needs 27 components - but variance in a mixed time/frequency
# feature space is not the same as class-discriminative information, and 27
# dimensions with diagonal covariance would be 216 emission parameters fitted
# from 364 observations.
#
# Instead D is the smallest dimension that reaches the accuracy plateau of a
# diagonal-covariance Gaussian classifier (which is exactly the HMM's
# emission model) under 5-fold cross-validation grouped BY RECORDING:
#     D =  2 -> 0.9392      D =  6 -> 1.0000
#     D =  3 -> 0.9971      D = 12 -> 1.0000
#     D =  4 -> 1.0000      D = 16 -> 0.9974  (covariance overfitting)
N_COMPONENTS = 4
MAX_FEATURE_CORR = 0.95
RANDOM_SEED = 42

# --- Circuit annotation handling ------------------------------------------
# The two continuous circuit recordings carry in-app annotation taps naming
# the activity being entered. Two caveats, both measured rather than assumed:
#
# 1. Annotation.csv timestamps come from a different clock to the sensor
#    files - the sensors use Unix epoch nanoseconds while the annotations use
#    a monotonic boot clock (~7.96 days). Only the INTERVALS between taps are
#    meaningful, so the first tap is anchored to the start of the recording.
#
# 2. Taps are placed by hand mid-activity, so boundaries carry human reaction
#    error. Checked against the amplitude envelope, six of the seven
#    Circuit_01 boundaries land within 0.7-4.0 s of the true change (mean
#    ~2 s). Windows within BOUNDARY_GUARD_S of any boundary are therefore
#    excluded from window-level labelling rather than trusted.
#
# The transition matrix depends only on the ORDER of activities and their
# dwell times, both of which survive a few seconds of boundary error, so the
# circuits remain sound for estimating A.
BOUNDARY_GUARD_S = 3.0

# --- Transition smoothing -------------------------------------------------
# Every transition must stay strictly positive or Viterbi can never cross
# that edge, however strongly the emissions argue for it. But the circuits
# supply only 23-84 window-level counts per row, so a full Laplace +1 is a
# heavy prior: it adds 4 to a row of 23 and badly distorts the dwell times
# that the diagonal encodes.
#
# Implied mean dwell, hop / (1 - A[k,k]), against the measured means
# (Still 12.05, Standing 16.91, Walking 11.34, Jumping 11.10 s):
#     smoothing 1.00 ->  7.43, 12.57,  7.40,  5.40   (Jumping off by 2x)
#     smoothing 0.10 -> 11.26, 19.63, 14.52, 10.17   (close)
#     smoothing 0.01 -> 11.92, 20.85, 16.28, 11.35
#
# 0.1 keeps the smallest entry at ~0.0012 - comfortably positive and finite
# in log space - without rewriting the dwell structure. The understated
# dwell at +1 is a smoothing artefact, NOT evidence against the geometric
# dwell assumption, which is a separate and genuine limitation.
TRANSITION_SMOOTHING = 0.1

# One tap in Circuit_02 is a clear mistap: the taps read Jumping at 73.8 s
# then Standing at 76.1 s, 2.3 s apart, while the envelope shows unambiguous
# jumping (log10 std ~0.9-1.1) continuing to ~84 s. Tapping accurately
# mid-jump is not realistic. This override is applied in data.py and is
# reported in the write-up rather than silently corrected.
ANNOTATION_OVERRIDES = {
    "Circuit_02": [
        # (index of the tap to move, corrected time in seconds)
        (6, 84.5),
    ],
}

SENSOR_AXES = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
CHANNELS = SENSOR_AXES + ["acc_mag", "gyr_mag"]
