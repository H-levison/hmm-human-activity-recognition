"""Estimating the transition matrix A from the circuit recordings.

Why this is a separate module
-----------------------------
The 52 isolated clips contain one activity each, so they hold zero observed
transitions. An A estimated from them is pure Laplace prior - every
off-diagonal entry identical, carrying no information about behaviour. The two
continuous circuit recordings are the only source of real transitions.

Why A is estimated from segments rather than from window labels
---------------------------------------------------------------
The circuit boundaries come from hand-placed annotation taps and carry a
measured reaction error of roughly 2 s (up to 4 s). Counting transitions from
per-window labels would inherit that error exactly where it hurts most -
at the boundaries.

A segment-level count avoids the problem entirely. A segment of an activity
lasting D seconds, sampled at one window per HOP_SECONDS, contributes
(D / hop - 1) self-transitions and exactly one transition into the next
activity. That uses only the ORDER of activities and their DWELL TIMES, both
of which survive a few seconds of boundary error - a 2 s error on a 15 s
segment moves a dwell count by two windows and changes nothing structural.

The geometric dwell distribution a standard HMM imposes on each state is a
known limitation of the model class rather than of this implementation;
hidden semi-Markov models exist precisely to relax it (Yu, 2010).

References
----------
Yu, S.-Z. (2010). Hidden semi-Markov models. Artificial Intelligence, 174(2),
    215-243.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (ACTIVITIES, HOP_SECONDS, STATE_INDEX,
                    TRANSITION_SMOOTHING)


def transition_counts(segments: pd.DataFrame, hop_s: float = HOP_SECONDS,
                      n_states: int = len(ACTIVITIES)) -> np.ndarray:
    """Count window-level transitions implied by the annotated segments.

    Segments are grouped by recording so the last segment of one circuit does
    not spuriously transition into the first segment of the next.
    """
    counts = np.zeros((n_states, n_states))

    for _, circuit in segments.groupby("recording", sort=True):
        circuit = circuit.sort_values("segment")
        states = [STATE_INDEX[a] for a in circuit["activity"]]
        dwell_windows = np.maximum(
            np.round(circuit["duration_s"].to_numpy() / hop_s).astype(int), 1)

        for k, n in zip(states, dwell_windows):
            counts[k, k] += max(n - 1, 0)
        for i, j in zip(states[:-1], states[1:]):
            counts[i, j] += 1.0

    return counts


def estimate_A(segments: pd.DataFrame, hop_s: float = HOP_SECONDS,
               smoothing: float = TRANSITION_SMOOTHING,
               n_states: int = len(ACTIVITIES)) -> tuple[np.ndarray, np.ndarray]:
    """Row-stochastic A from the circuits, plus the raw counts.

    Smoothing keeps every entry strictly positive: an unobserved transition
    must stay possible, or Viterbi could never cross that edge no matter how
    strongly the emissions argue for it. With only two circuits several
    physically plausible transitions genuinely are unobserved, so this matters
    here rather than being a formality.

    The default is deliberately 0.1 rather than a full Laplace +1 - see
    TRANSITION_SMOOTHING in config.py. The rows hold only 23-84 counts, and
    +1 per cell distorts the dwell times encoded in the diagonal by up to 2x.
    """
    counts = transition_counts(segments, hop_s, n_states)
    smoothed = counts + smoothing
    A = smoothed / smoothed.sum(axis=1, keepdims=True)
    return A, counts


def estimate_pi(segments: pd.DataFrame, smoothing: float = TRANSITION_SMOOTHING,
                n_states: int = len(ACTIVITIES)) -> np.ndarray:
    """Initial distribution from which activity each circuit starts in."""
    counts = np.full(n_states, smoothing)
    for _, circuit in segments.groupby("recording", sort=True):
        first = circuit.sort_values("segment")["activity"].iloc[0]
        counts[STATE_INDEX[first]] += 1.0
    return counts / counts.sum()


def dwell_summary(segments: pd.DataFrame,
                  hop_s: float = HOP_SECONDS) -> pd.DataFrame:
    """Observed dwell time per activity, and what A's diagonal implies.

    A standard HMM gives each state a geometric dwell distribution with mean
    1 / (1 - A[k, k]) windows. Comparing that against the measured mean dwell
    is the honest way to show where the Markov assumption strains - real
    activities have a far tighter duration spread than a geometric law.
    """
    A, _ = estimate_A(segments, hop_s)
    rows = []
    for k, name in enumerate(ACTIVITIES):
        d = segments.loc[segments["activity"] == name, "duration_s"]
        rows.append({
            "activity": name,
            "segments": int(len(d)),
            "mean_dwell_s": float(d.mean()) if len(d) else float("nan"),
            "min_dwell_s": float(d.min()) if len(d) else float("nan"),
            "max_dwell_s": float(d.max()) if len(d) else float("nan"),
            "self_transition": float(A[k, k]),
            "implied_mean_dwell_s": float(hop_s / (1.0 - A[k, k])),
        })
    return pd.DataFrame(rows)


def as_frame(A: np.ndarray) -> pd.DataFrame:
    """A labelled with activity names, for printing and for the heatmap."""
    return pd.DataFrame(A, index=ACTIVITIES, columns=ACTIVITIES)
