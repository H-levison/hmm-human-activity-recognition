"""Figures for the report. Every plot writes into results/figures/.

Design notes
------------
The figure set is built to read as one system: a single fixed categorical
order for the four activities, one sequential blue ramp for every magnitude
heatmap, recessive axes, thin marks and no dual axes anywhere.

Colour assignment follows the job each colour does, not taste:

* ACTIVITY_COLORS is the CATEGORICAL slot order (identity - which activity).
  Assigned in fixed order and never cycled, so Still is the same blue in every
  figure in the report.
* SEQ_RAMP is a single-hue SEQUENTIAL ramp (magnitude - how large a
  probability or a count is), used for the transition and confusion matrices.
  One hue light-to-dark, never a rainbow, so the reader sees magnitude in the
  lightness rather than having to decode a hue wheel.

The palette was validated rather than eyeballed: on the adjacent pairlist the
four slots pass the lightness band, chroma floor, CVD separation (worst pair
protanopia dE 9.1, target >= 8) and the normal-vision floor (22.9, floor 15).
Two slots - aqua 2.74:1 and yellow 2.11:1 - sit below 3:1 contrast on the light
surface, so the relief rule applies: every panel that uses them carries a
visible text label, and identity is never left to colour alone.

The PCA scatter is faceted rather than overlaid on purpose. Scatter is an
"all pairs" form where any two marks can end up side by side, and the four-slot
palette does not clear the all-pairs floors (yellow against orange). Small
multiples with one highlighted activity per panel keeps every comparison inside
a validated pair.

Figures target print, so only the light surface is defined - a report PDF has
no dark mode to switch into.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

from config import ACTIVITIES, FIGURES, TARGET_HZ

# --- surfaces and ink -----------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8980"
GRID = "#e4e3de"

# --- categorical: identity, fixed order, never cycled ---------------------
ACTIVITY_COLORS = {
    "Still": "#2a78d6",      # slot 1 blue
    "Standing": "#eb6834",   # slot 2 orange
    "Walking": "#1baf7a",    # slot 3 aqua
    "Jumping": "#eda100",    # slot 4 yellow
}

# --- sequential: magnitude, one hue light -> dark -------------------------
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)

# --- diverging: polarity around zero, two hues + neutral grey midpoint --
DIV_RAMP = ["#104281", "#2a78d6", "#86b6ef", "#e8e7e2",
            "#f3a884", "#eb6834", "#a23c16"]
DIV_CMAP = LinearSegmentedColormap.from_list("div_blue_orange", DIV_RAMP)


def use_report_style() -> None:
    """Shared rcParams: recessive chrome, thin marks, no top/right spines."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
    })


def _save(fig, path: Path | str | None, default_name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = Path(path) if path else FIGURES / default_name
    fig.savefig(out)
    plt.close(fig)
    return out


def _cell_ink(value: float, vmax: float) -> str:
    """Ink that stays legible as the sequential fill darkens."""
    return SURFACE if value > 0.55 * vmax else INK


# --- 1. sample signals ----------------------------------------------------
def plot_sample_signals(clean: pd.DataFrame, path=None, recording_index: int = 0,
                        wide: bool = False):
    """Accelerometer magnitude traces for all four activities, stacked.

    One panel per activity rather than one overlaid axes: the amplitudes span
    roughly three orders of magnitude, so a shared y-scale would flatten Still
    and Standing into the zero line. Each panel therefore states its own
    measured standard deviation, which is the number that actually separates
    the classes.
    """
    use_report_style()
    # `wide` lays the four activities in one row rather than a column, which
    # fits a page-width report figure in under half the vertical space.
    if wide:
        fig, axes = plt.subplots(1, len(ACTIVITIES), figsize=(14.0, 2.9),
                                 sharex=True)
    else:
        fig, axes = plt.subplots(len(ACTIVITIES), 1, figsize=(7.2, 7.0),
                                 sharex=True)

    for ax, activity in zip(axes, ACTIVITIES):
        recs = sorted(clean.loc[clean.activity == activity, "recording"].unique())
        clip = clean[clean.recording == recs[recording_index]].sort_values("time_s")
        t = clip["time_s"].to_numpy()
        y = clip["acc_mag"].to_numpy()
        colour = ACTIVITY_COLORS[activity]

        ax.plot(t, y, color=colour, linewidth=1.2)
        ax.set_ylabel("|a|  (m/s$^2$)")
        ax.set_title(f"{activity}   -   {recs[recording_index]}", loc="left")
        ax.text(0.995, 0.92, f"std = {y.std():.3f} m/s$^2$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=INK_SECONDARY)
        ax.margins(x=0)

    axes[-1].set_xlabel("time within clip (s)")
    fig.suptitle("Accelerometer magnitude by activity", x=0.065, y=0.995,
                 ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _save(fig, path, "01_sample_signals.png")


# --- 2. spectra -----------------------------------------------------------
def plot_spectra(clean: pd.DataFrame, path=None):
    """Mean FFT magnitude spectrum per activity, dominant frequency marked.

    Averaged over every clip of the activity so the panel shows the activity's
    signature rather than one recording's noise. Faceted for the same reason as
    the traces - the magnitudes are not comparable on one scale.
    """
    use_report_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))

    for ax, activity in zip(axes.ravel(), ACTIVITIES):
        spectra = []
        for _, clip in clean[clean.activity == activity].groupby("recording"):
            y = clip.sort_values("time_s")["acc_mag"].to_numpy()
            y = y - y.mean()
            spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
            spectra.append(spec)
        n = min(len(s) for s in spectra)
        mean_spec = np.mean([s[:n] for s in spectra], axis=0)
        freqs = np.fft.rfftfreq(2 * (n - 1), d=1.0 / TARGET_HZ)[:n]

        colour = ACTIVITY_COLORS[activity]
        ax.fill_between(freqs, mean_spec, color=colour, alpha=0.18, linewidth=0)
        ax.plot(freqs, mean_spec, color=colour, linewidth=1.5)

        k = int(np.argmax(mean_spec[1:]) + 1)
        ax.axvline(freqs[k], color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
        ax.annotate(f"{freqs[k]:.2f} Hz", xy=(freqs[k], mean_spec[k]),
                    xytext=(6, -2), textcoords="offset points",
                    fontsize=8, color=INK_SECONDARY, va="top")

        ax.set_title(activity, loc="left")
        ax.set_xlim(0, 12)
        ax.margins(y=0.12)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("mean |FFT|")

    fig.suptitle("Frequency signature by activity", x=0.055, y=1.0,
                 ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path, "02_spectra.png")


# --- 3. transition matrix -------------------------------------------------
def plot_transition_heatmap(A: np.ndarray, labels: list[str] | None = None,
                            path=None, title: str | None = None):
    """Annotated heatmap of the transition matrix.

    Sequential single-hue ramp because the quantity is a magnitude (a
    probability), not an identity. Every cell is annotated, so the exact value
    never depends on reading a colour.
    """
    use_report_style()
    labels = labels or ACTIVITIES
    n = len(labels)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(A, cmap=SEQ_CMAP, vmin=0.0, vmax=1.0)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color=_cell_ink(A[i, j], 1.0))

    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("to state at $t+1$")
    ax.set_ylabel("from state at $t$")
    ax.set_title(title or "Transition matrix $A$", loc="left")
    ax.grid(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(next state | current state)", color=INK_SECONDARY, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=INK_SECONDARY)

    fig.tight_layout()
    return _save(fig, path, "03_transition_matrix.png")


# --- 4. confusion matrix --------------------------------------------------
def plot_confusion(cm: np.ndarray, labels: list[str] | None = None, path=None,
                   title: str = "Confusion matrix"):
    """Counts beside row-normalised rates, both on the sequential ramp."""
    use_report_style()
    labels = labels or ACTIVITIES
    n = len(labels)
    rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, data, vmax, fmt, sub in (
            (axes[0], cm, cm.max(), "{:d}", "counts"),
            (axes[1], rates, 1.0, "{:.2f}", "row-normalised (recall)")):
        ax.imshow(data, cmap=SEQ_CMAP, vmin=0, vmax=vmax)
        for i in range(n):
            for j in range(n):
                v = data[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=9, color=_cell_ink(v, vmax))
        ax.set_xticks(range(n), labels, rotation=20, ha="right")
        ax.set_yticks(range(n), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(sub, loc="left")
        ax.grid(False)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)

    fig.suptitle(title, x=0.045, y=1.0, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path, "04_confusion.png")


# --- 5. decoded sequence --------------------------------------------------
def plot_decoded_sequence(y_true, y_pred, t=None, labels=None, path=None,
                          title="Decoded state sequence",
                          guard: np.ndarray | None = None,
                          boundaries: list[float] | None = None):
    """Ground truth against the Viterbi path over time.

    Colour is deliberately NOT doing identity work here - the state is encoded
    by vertical position and named on the y-axis, so the two step lines only
    need to be told apart from each other. Windows where the two disagree are
    shaded, which is the thing a reader is actually looking for.
    """
    use_report_style()
    labels = labels or ACTIVITIES
    y_true = np.asarray(y_true, int).ravel()
    y_pred = np.asarray(y_pred, int).ravel()
    t = np.arange(len(y_true)) if t is None else np.asarray(t, float)

    fig, ax = plt.subplots(figsize=(9.5, 3.4))

    mismatch = y_true != y_pred
    for i in np.flatnonzero(mismatch):
        lo = t[i] - (t[1] - t[0]) / 2 if len(t) > 1 else t[i] - 0.5
        hi = lo + (t[1] - t[0] if len(t) > 1 else 1.0)
        ax.axvspan(lo, hi, color="#e34948", alpha=0.14, linewidth=0, zorder=0)

    if guard is not None:
        for i in np.flatnonzero(np.asarray(guard, bool)):
            lo = t[i] - (t[1] - t[0]) / 2 if len(t) > 1 else t[i] - 0.5
            hi = lo + (t[1] - t[0] if len(t) > 1 else 1.0)
            ax.axvspan(lo, hi, color=INK_MUTED, alpha=0.10, linewidth=0, zorder=0)

    for b in (boundaries or []):
        ax.axvline(b, color=INK_MUTED, linewidth=0.9, linestyle=(0, (3, 3)),
                   zorder=1)

    ax.step(t, y_true, where="mid", color=INK_MUTED, linewidth=2.6,
            label="annotated", zorder=2)
    ax.step(t, y_pred, where="mid", color="#2a78d6", linewidth=1.8,
            label="Viterbi", zorder=3)

    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.set_xlabel("window start time (s)")
    ax.set_xlim(t.min(), t.max())
    agree = (~mismatch).mean()
    ax.set_title(f"{title}   -   {agree:.1%} of windows agree", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncols=2)
    ax.grid(axis="x", alpha=0.5)

    fig.tight_layout()
    return _save(fig, path, "05_decoded_sequence.png")


# --- 6. feature space -----------------------------------------------------
def plot_feature_space(Z: np.ndarray, y: np.ndarray, labels=None, path=None,
                       mu: np.ndarray | None = None,
                       var: np.ndarray | None = None):
    """PCA scatter, faceted one activity per panel, with emission ellipses.

    Faceted rather than overlaid: scatter is an all-pairs form and the
    four-slot categorical palette does not clear the all-pairs separation
    floors, so each panel shows one activity in its own slot colour against
    the rest in muted grey. The 2-sigma ellipses are axis-aligned because the
    emissions use diagonal covariance - the shape of the ellipse is a picture
    of that modelling assumption.
    """
    use_report_style()
    labels = labels or ACTIVITIES
    Z = np.asarray(Z, float)
    y = np.asarray(y, int).ravel()

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4), sharex=True, sharey=True)
    for k, (ax, activity) in enumerate(zip(axes.ravel(), labels)):
        other = y != k
        ax.scatter(Z[other, 0], Z[other, 1], s=14, color="#d8d7d0",
                   linewidths=0, zorder=1)
        ax.scatter(Z[~other, 0], Z[~other, 1], s=22,
                   color=ACTIVITY_COLORS[activity], linewidths=0.6,
                   edgecolors=SURFACE, zorder=2)

        if mu is not None and var is not None:
            sd = np.sqrt(var[k])
            ax.add_patch(Ellipse((mu[k, 0], mu[k, 1]),
                                 width=4 * sd[0], height=4 * sd[1],
                                 facecolor="none",
                                 edgecolor=ACTIVITY_COLORS[activity],
                                 linewidth=1.4, linestyle=(0, (4, 2)), zorder=3))
            ax.plot(mu[k, 0], mu[k, 1], marker="x", markersize=7,
                    markeredgewidth=1.8,
                    color=ACTIVITY_COLORS[activity], zorder=4)

        ax.set_title(f"{activity}  (n = {(~other).sum()})", loc="left")
        if k >= 2:
            ax.set_xlabel("PC 1")
        if k % 2 == 0:
            ax.set_ylabel("PC 2")

    fig.suptitle("Window feature space after PCA, with 2$\\sigma$ emission "
                 "ellipses", x=0.055, y=1.0, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path, "06_feature_space.png")


# --- 7. EM convergence ----------------------------------------------------
def plot_loglik_curve(history, path=None,
                      title="Baum-Welch convergence"):
    """Log-likelihood per EM iteration.

    One series, so no legend box - the title names it. The curve must be
    monotone non-decreasing; that is the visual proof the E and M steps are
    consistent with each other.
    """
    use_report_style()
    history = np.asarray(history, float)
    it = np.arange(1, len(history) + 1)

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(it, history, color="#2a78d6", marker="o", markersize=4.5,
            markerfacecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("total log-likelihood")

    gain = history[-1] - history[0]
    ax.annotate(f"{history[-1]:,.1f}", xy=(it[-1], history[-1]),
                xytext=(-4, -14), textcoords="offset points",
                ha="right", fontsize=8, color=INK_SECONDARY)
    ax.set_title(f"{title}   -   {len(history)} iterations, "
                 f"+{gain:,.1f} nats", loc="left")
    ax.margins(x=0.04)

    fig.tight_layout()
    return _save(fig, path, "07_em_convergence.png")


# --- 8. class separability (supporting figure) ---------------------------
def plot_separability(train: pd.DataFrame, unseen: pd.DataFrame | None = None,
                      column: str = "acc_mag_std", path=None):
    """Per-window distribution of the strongest single feature, by activity.

    This is the figure that explains the unseen-session result: if the
    training and held-out distributions for an activity sit in different
    places, the emission model trained on one cannot score the other.
    Log scale, because the classes span three orders of magnitude.
    """
    use_report_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))

    positions = np.arange(len(ACTIVITIES))
    for offset, (table, name, filled) in enumerate(
            [(train, "training session", True)] +
            ([(unseen, "unseen session", False)] if unseen is not None else [])):
        for k, activity in enumerate(ACTIVITIES):
            vals = table.loc[table.activity == activity, column].to_numpy()
            vals = np.maximum(vals, 1e-4)
            x = positions[k] + (offset - 0.5 * (unseen is not None)) * 0.34
            colour = ACTIVITY_COLORS[activity]
            ax.scatter(np.full(len(vals), x) + np.random.default_rng(k).normal(0, 0.035, len(vals)),
                       vals, s=10, linewidths=0 if filled else 0.7,
                       color=colour if filled else "none",
                       edgecolors=colour, alpha=0.75, zorder=2)
            ax.plot([x - 0.12, x + 0.12], [np.median(vals)] * 2,
                    color=INK, linewidth=1.6, zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(positions, ACTIVITIES)
    ax.set_ylabel(column.replace("_", " "))
    ax.set_title("Filled = training session   |   hollow = unseen session "
                 "(different room, phone in pocket)", loc="left",
                 fontsize=9, color=INK_SECONDARY)
    fig.suptitle("Where the held-out session moved", x=0.055, y=1.0,
                 ha="left", fontsize=11, fontweight="bold", color=INK)
    ax.grid(axis="x", alpha=0.0)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path, "08_separability.png")


# --- 9. emission probabilities -------------------------------------------
def plot_emissions(mu: np.ndarray, var: np.ndarray, Z: np.ndarray | None = None,
                   y: np.ndarray | None = None, labels=None, path=None,
                   ncols: int = 2, figsize=None):
    """Emission probability densities p(x | z) for every state and dimension.

    This is the direct picture of B. The emissions are diagonal-covariance
    Gaussians, so the joint density over the PCA feature vector factorises
    into one univariate Gaussian per component - which is exactly what makes
    it drawable. Each panel is one PCA component; each curve is one activity's
    density along that component.

    Where two curves overlap heavily, the model genuinely cannot separate
    those activities from that component alone; where they sit apart, that
    component is doing discriminative work. Read with the observed windows
    underneath (passed as Z and y) it also shows how well the Gaussian
    assumption fits the data it was fitted to.

    Lines use the adjacent pairlist, so the four validated categorical slots
    are legal here, and every curve is additionally direct-labelled.
    """
    use_report_style()
    labels = labels or ACTIVITIES
    n_states, n_dims = mu.shape

    # A single wide row (ncols = n_dims) carries the same information in about
    # half the vertical space, which is what the report uses.
    nrows = int(np.ceil(n_dims / ncols))
    if figsize is None:
        figsize = (7.4, 2.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for d in range(n_dims):
        ax = axes[d]
        lo = float(np.min(mu[:, d] - 4 * np.sqrt(var[:, d])))
        hi = float(np.max(mu[:, d] + 4 * np.sqrt(var[:, d])))
        if Z is not None:
            lo = min(lo, float(Z[:, d].min()))
            hi = max(hi, float(Z[:, d].max()))
        grid = np.linspace(lo, hi, 600)

        for k, activity in enumerate(labels):
            sd = np.sqrt(var[k, d])
            dens = np.exp(-0.5 * ((grid - mu[k, d]) / sd) ** 2) / (
                sd * np.sqrt(2 * np.pi))
            colour = ACTIVITY_COLORS[activity]
            ax.plot(grid, dens, color=colour, linewidth=1.8, zorder=3)
            ax.fill_between(grid, dens, color=colour, alpha=0.12,
                            linewidth=0, zorder=2)
            # Direct label at each curve's peak, stepped vertically by state
            # so that two states with near-identical means and peak heights
            # (Still and Standing on several components) do not overprint.
            ax.annotate(activity, xy=(mu[k, d], dens.max()),
                        xytext=(0, 4 + 10 * (k % 2)), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK_SECONDARY,
                        zorder=4)

            if Z is not None and y is not None:
                pts = Z[y == k, d]
                ax.plot(pts, np.full(len(pts), -0.04 * dens.max()),
                        marker="|", linestyle="none", markersize=4,
                        markeredgewidth=0.8, color=colour, alpha=0.5, zorder=1)

        ax.set_title(f"PC {d + 1}", loc="left")
        ax.set_xlabel(f"PC {d + 1} value")
        ax.set_ylabel(r"density  $p(x \mid z)$")
        ax.margins(y=0.30)

    for ax in axes[n_dims:]:
        ax.set_visible(False)

    handles = [plt.Line2D([], [], color=ACTIVITY_COLORS[a], linewidth=2,
                          label=a) for a in labels]
    fig.legend(handles=handles, loc="lower center", ncols=len(labels),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(r"Emission probabilities $B$: $p(x \mid z)$ per state, "
                 "per PCA component", x=0.055, y=1.0, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return _save(fig, path, "09_emissions.png")


def plot_emission_parameters(mu: np.ndarray, var: np.ndarray, labels=None,
                             path=None):
    """Emission means and variances as annotated heatmaps.

    The companion to plot_emissions: the densities show the shapes, this shows
    the numbers behind them. Means use a diverging ramp because they are
    signed and centred on zero (PCA components are mean-centred, so the sign
    is meaningful); variances use the sequential ramp because they are a
    magnitude.
    """
    use_report_style()
    labels = labels or ACTIVITIES
    n_states, n_dims = mu.shape
    cols = [f"PC {i + 1}" for i in range(n_dims)]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    lim = float(np.abs(mu).max())

    for ax, data, cmap, vmin, vmax, title, fmt in (
            (axes[0], mu, DIV_CMAP, -lim, lim, r"emission means $\mu_k$", "{:+.2f}"),
            (axes[1], var, SEQ_CMAP, 0.0, float(var.max()),
             r"emission variances $\sigma^2_k$", "{:.2f}")):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(n_states):
            for j in range(n_dims):
                v = data[i, j]
                if cmap is SEQ_CMAP:
                    ink = _cell_ink(v, vmax)
                else:
                    ink = SURFACE if abs(v) > 0.55 * lim else INK
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=8.5, color=ink)
        ax.set_xticks(range(n_dims), cols)
        ax.set_yticks(range(n_states), labels)
        ax.set_title(title, loc="left")
        ax.grid(False)
        ax.set_xticks(np.arange(-0.5, n_dims, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_states, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)

    fig.suptitle("Emission parameters per hidden state", x=0.045, y=1.0,
                 ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path, "09b_emission_parameters.png")
