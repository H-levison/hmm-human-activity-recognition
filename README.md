# Inferring Human Activity States with a Hidden Markov Model

Smartphone accelerometer and gyroscope signals are recorded for four activity
states, reduced to time- and frequency-domain features, and decoded with a
Gaussian-emission Hidden Markov Model implemented from scratch in numpy.

**Author:** HonourGod Kilechukwu Levison
**Course:** Machine Learning Techniques II — Formative 2

## Use case: rehabilitation adherence after knee or hip replacement

After joint replacement surgery a patient is sent home with a prescription that
is entirely about movement: walk a prescribed amount each day, build standing
tolerance gradually, and avoid high-impact loading until the joint has healed.
The surgeon cannot observe any of it. Follow-up relies on the patient's own
recall at a clinic visit weeks later, which is exactly the kind of self-report
that is known to be unreliable. A smartphone already in the patient's pocket
carries the accelerometer and gyroscope needed to measure that prescription
directly, and each of the four activity states maps onto a clinical question:
**Walking** is the prescribed mobilisation dose, **Standing** is weight-bearing
tolerance, **Jumping** stands for contraindicated high-impact movement that
should raise an alert, and **Still** means the phone is lying on a surface and
is not being carried at all. That last state is the one that makes the problem
interesting rather than trivial: a system that cannot separate "the patient is
resting" from "the phone is on the kitchen table" will quietly record a
non-wear period as clinical inactivity and tell the surgeon the patient is not
recovering.

The activity itself is never measured - only a noisy, orientation-dependent
stream of accelerations is. The true state is hidden behind that stream and has
to be inferred, and because activities persist for many seconds and change in
patterned rather than arbitrary ways, the sequence of states carries
information that classifying each window independently would discard. That is
precisely the structure a Hidden Markov Model encodes.

## Data

52 recordings, 13 per activity, one participant, one device, 50 Hz. Full
collection protocol, cleaning steps and schema in [data/README.md](data/README.md).

| Activity | Files | Clean seconds |
|---|---|---|
| Still | 13 | 104.0 |
| Standing | 13 | 104.0 |
| Walking | 13 | 104.0 |
| Jumping | 13 | 104.0 |

## Model

| Element | This project |
|---|---|
| Hidden states `Z` | 4 — Still, Standing, Walking, Jumping |
| Observations `X` | Per-window feature vectors (time + frequency domain, PCA-reduced) |
| Transitions `A` | 4×4 row-stochastic, Laplace-smoothed |
| Emissions `B` | Multivariate Gaussian per state, diagonal covariance |
| Initial `π` | 4-vector over the first window of each sequence |

Implemented from scratch in [src/hmm.py](src/hmm.py), entirely in log space:
Forward, Backward, posteriors, Baum-Welch (EM), Viterbi and MAP decoding.

**Algorithm roles.** The **Forward** algorithm gives `P(X | λ)`, used to compare
the supervised and unsupervised fits. **Baum-Welch** handles learning when
labels are withheld; closed-form MLE replaces it when they are available.
**Viterbi** does the decoding, and its output *is* the predicted label per
window — so it is what feeds the confusion matrix and therefore sensitivity,
specificity and accuracy. Because Viterbi maximises the probability of the
whole path rather than each frame independently, it suppresses isolated
single-window flips.

## Windowing

2.0 s windows with a 1.0 s hop (50% overlap). At 50 Hz that is **exactly 100
samples** per window, so FFT bins are identical across every window; 2 s spans
3–4 walking gait cycles; and the 0.5 Hz spectral resolution separates walking
(~1 Hz) from jumping (~3.6 Hz). Each 8 s clip yields 7 windows → 91 per
activity, 364 in total.

## Repository layout

```
├── data/              Raw export, cleaned CSVs, windowed features  (see data/README.md)
├── notebooks/         Analysis notebook
├── src/
│   ├── config.py      Every tunable constant, defended in one place
│   ├── data.py        Sensor Logger zips -> clean uniform 50 Hz CSVs
│   ├── features.py    Windowing + time/frequency feature extraction
│   ├── pipeline.py    Leakage-safe splits, scaling and PCA
│   ├── hmm.py         Gaussian HMM from scratch (forward/backward/EM/Viterbi)
│   ├── transitions.py Segment-level transition and dwell estimation
│   ├── metrics.py     Per-state sensitivity, specificity, accuracy
│   ├── viz.py         Report figures
│   ├── check_hmm.py   19 correctness properties; exits non-zero on failure
│   ├── experiment.py  Runs every experiment, writes results/
│   ├── report_text.py The report's prose (python src/report_text.py counts it)
│   └── build_report.py Lays that prose out as the .docx
├── results/           Figures and metric tables
└── report/            Written report
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

python src/data.py            # rebuild data/clean from the raw exports
jupyter notebook notebooks/
```

## Video documentation

Four clips, one per activity.

All four clips, plus a recording of the Sensor Logger setup, are in one
link-shareable Google Drive folder:

**https://drive.google.com/drive/folders/1jfF0QRI9SgRetKKAMbNT-0WUdd2tBeEx?usp=sharing**

| Activity | File |
|---|---|
| Still | `Still Log Evidence.mp4` |
| Standing | `Standing Log Evidence.mp4` |
| Walking | `Walking Log Evidence.mp4` |
| Jumping | `Jumping Log Evidence.mp4` |
| Setup | `Sensor Logger Evidence.mp4` |

## Results

| Condition | Windows | Accuracy | Macro F1 |
|---|---|---|---|
| Held-out recordings, same session | 84 | 1.0000 | 1.0000 |
| **Unseen session** (different room, phone in pocket) | 84 | **0.6548** | 0.6111 |
| Circuits, outside the annotation guard band | 90 | 0.6778 | 0.6528 |
| Baum-Welch, unsupervised, training windows | 364 | 1.0000 | 1.0000 |

Unseen-session per-state metrics:

| Activity | Samples | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| Still | 21 | 0.1429 | 1.0000 | 0.7857 |
| Standing | 21 | 0.4762 | 0.7302 | 0.6667 |
| Walking | 21 | 1.0000 | 0.8095 | 0.8571 |
| Jumping | 21 | 1.0000 | 1.0000 | 1.0000 |

Every error is a one-step shift up the amplitude ladder — Still read as Standing,
Standing read as Walking — because carrying the phone in a pocket inflated every
amplitude distribution. Jumping and Walking survive untouched because they are
identified by *frequency* signature rather than scale.

## Reproducing

```bash
python src/data.py          # raw zips -> clean, unseen, circuits
python src/features.py      # -> windowed feature tables
python src/check_hmm.py     # 19 correctness properties (exits non-zero on failure)
python src/experiment.py    # all experiments -> results/tables + results/figures
jupyter notebook notebooks/HonourGod_Levison_Formative2_HMM.ipynb
```

