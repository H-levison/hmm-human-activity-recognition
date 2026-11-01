"""Run every experiment the report needs and write the tables and figures.

    python src/experiment.py

Design of the evaluation
------------------------
Three test conditions, reported separately because they answer different
questions:

1. HELD-OUT RECORDINGS from the training session. Split by recording, never by
   window. Answers "does the model generalise beyond the specific clips it was
   fitted on?" - the easy question.
2. The UNSEEN SESSION: a different room, phone in a trouser pocket rather than
   in hand, recorded on a later day. Answers "does it generalise beyond the
   conditions it was trained in?" - the real question, and the one the brief
   asks about.
3. The CIRCUITS: two continuous recordings that actually contain transitions.
   Used for the transition matrix and the decoded-sequence figure. Their
   window-level accuracy is reported but flagged, because the boundaries come
   from hand-placed annotation taps carrying a measured ~2 s reaction error.

Emissions always come from the isolated clips, which have clean labels.
A and pi always come from the circuits, which are the only recordings
containing real transitions.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import viz
from config import (ACTIVITIES, FIGURES, HOP_SECONDS, N_COMPONENTS, PROCESSED,
                    RANDOM_SEED, TABLES)
from hmm import GaussianHMM, align_states, remap
from metrics import confusion, format_report, overall, per_state_report
from pipeline import Preprocessor, split_recordings, subset, to_sequences
from transitions import as_frame, dwell_summary, estimate_A, estimate_pi

pd.set_option("display.width", 200, "display.max_columns", 40)


def rule(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / name, index=index)


def evaluate(model: GaussianHMM, sequences, label_seqs, name: str,
             decoder: str = "viterbi") -> tuple[pd.DataFrame, dict]:
    decode = model.viterbi if decoder == "viterbi" else model.map_decode
    pred = np.concatenate([decode(X) for X in sequences])
    gold = np.concatenate(label_seqs)
    report = per_state_report(gold, pred, ACTIVITIES)
    stats = overall(gold, pred)
    print(f"\n{name}  ({decoder})")
    print(format_report(report).to_string(index=False))
    print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in stats.items()))
    return report, stats


def main() -> int:
    train = pd.read_csv(PROCESSED / "windows_train.csv")
    unseen = pd.read_csv(PROCESSED / "windows_unseen.csv")
    circuits = pd.read_csv(PROCESSED / "windows_circuits.csv")
    segments = pd.read_csv(PROCESSED / "circuit_segments.csv")

    summary_rows = []

    # ---------------------------------------------------------------- A, pi
    rule("TRANSITION STRUCTURE (from the two circuit recordings)")
    A_circuit, counts = estimate_A(segments, hop_s=HOP_SECONDS)
    pi_circuit = estimate_pi(segments)

    print("\nwindow-level transition counts:")
    print(as_frame(counts).astype(int).to_string())
    print("\nA:")
    print(as_frame(A_circuit).round(4).to_string())
    print("\npi: " + ", ".join(f"{a}={p:.3f}"
                               for a, p in zip(ACTIVITIES, pi_circuit)))
    dwell = dwell_summary(segments, hop_s=HOP_SECONDS)
    print("\ndwell times, measured against what A's diagonal implies:")
    print(dwell.round(2).to_string(index=False))

    save_table(as_frame(A_circuit).round(6), "transition_matrix.csv", index=True)
    save_table(as_frame(counts).astype(int), "transition_counts.csv", index=True)
    save_table(dwell.round(3), "dwell_times.csv")
    save_table(segments, "circuit_segments.csv")

    # -------------------------------------------- 1. held-out recordings
    rule("1. HELD-OUT RECORDINGS from the training session")
    tr_recs, te_recs = split_recordings(train, n_test_per_activity=3,
                                        seed=RANDOM_SEED)
    tr, te = subset(train, tr_recs), subset(train, te_recs)
    print(f"train {len(tr_recs)} recordings | held out {len(te_recs)}: "
          f"{', '.join(te_recs)}")

    pre_split = Preprocessor().fit(tr, verbose=True)
    Xtr, ytr, _ = to_sequences(tr, pre_split.transform(tr))
    Xte, yte, _ = to_sequences(te, pre_split.transform(te))

    split_model = GaussianHMM(len(ACTIVITIES), N_COMPONENTS,
                              seed=RANDOM_SEED).fit_supervised(Xtr, ytr)
    split_model.set_params(pi_circuit, A_circuit, split_model.mu, split_model.var)
    rep, stats = evaluate(split_model, Xte, yte, "held-out recordings")
    save_table(format_report(rep), "eval_heldout_recordings.csv")
    summary_rows.append({"condition": "held-out recordings (same session)",
                         **stats})

    # ------------------------------------------- full model for conditions 2, 3
    rule("FULL MODEL: emissions from all 52 clips, A and pi from the circuits")
    pre = Preprocessor().fit(train, verbose=True)
    Xall, yall, _ = to_sequences(train, pre.transform(train))

    model = GaussianHMM(len(ACTIVITIES), N_COMPONENTS,
                        seed=RANDOM_SEED).fit_supervised(Xall, yall)
    A_isolated = model.A.copy()          # before overriding: the degenerate one
    model.set_params(pi_circuit, A_circuit, model.mu, model.var)
    print(f"train log-likelihood {model.log_likelihood(Xall):,.2f}")

    save_table(pd.DataFrame(model.mu, index=ACTIVITIES,
                            columns=[f"PC{i+1}" for i in range(N_COMPONENTS)]
                            ).round(4), "emission_means.csv", index=True)
    save_table(pd.DataFrame(model.var, index=ACTIVITIES,
                            columns=[f"PC{i+1}" for i in range(N_COMPONENTS)]
                            ).round(4), "emission_variances.csv", index=True)

    # --------------------------------------------------- 2. unseen session
    rule("2. UNSEEN SESSION - different room, phone in pocket, later day")
    Xun, yun, un_names = to_sequences(unseen, pre.transform(unseen))
    rep_un, stats_un = evaluate(model, Xun, yun, "unseen session")
    save_table(format_report(rep_un), "eval_unseen_session.csv")
    summary_rows.append({"condition": "unseen session", **stats_un})

    cm_un = confusion(np.concatenate(yun),
                      np.concatenate([model.viterbi(X) for X in Xun]),
                      len(ACTIVITIES))
    print("\nconfusion (rows = true, cols = predicted):")
    print(as_frame(cm_un).astype(int).to_string())
    save_table(as_frame(cm_un).astype(int), "confusion_unseen.csv", index=True)

    print("\nper recording:")
    per_rec = []
    for n, X, y in zip(un_names, Xun, yun):
        p = model.viterbi(X)
        acc = float((p == y).mean())
        per_rec.append({"recording": n, "true": ACTIVITIES[y[0]],
                        "windows": len(y), "correct": int((p == y).sum()),
                        "accuracy": round(acc, 4),
                        "decoded": " ".join(ACTIVITIES[q][:4] for q in p)})
        print(f"  {n:13s} true={ACTIVITIES[y[0]]:9s} acc={acc:5.2f}  "
              f"{' '.join(ACTIVITIES[q][:4] for q in p)}")
    save_table(pd.DataFrame(per_rec), "eval_unseen_per_recording.csv")

    # MAP decoding for comparison
    evaluate(model, Xun, yun, "unseen session", decoder="map")

    # -------------------------------------------------------- 3. circuits
    rule("3. CIRCUITS - continuous recordings containing real transitions")
    Xci, yci, ci_names = to_sequences(circuits, pre.transform(circuits))
    rep_ci, stats_ci = evaluate(model, Xci, yci, "circuits, all windows")
    summary_rows.append({"condition": "circuits (all windows)", **stats_ci})

    # restricted to windows outside the annotation guard band
    pred_ci = np.concatenate([model.viterbi(X) for X in Xci])
    gold_ci = np.concatenate(yci)
    order = (circuits.sort_values(["recording", "window_index"])
             .index.to_numpy())
    guard_sorted = ~circuits.loc[order, "near_boundary"].to_numpy(bool)

    rep_clean = per_state_report(gold_ci[guard_sorted], pred_ci[guard_sorted],
                                 ACTIVITIES)
    stats_clean = overall(gold_ci[guard_sorted], pred_ci[guard_sorted])
    print("\ncircuits, windows outside the annotation guard band")
    print(format_report(rep_clean).to_string(index=False))
    print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in stats_clean.items()))
    save_table(format_report(rep_clean), "eval_circuits_guarded.csv")
    summary_rows.append({"condition": "circuits (outside guard band)",
                         **stats_clean})

    # ------------------------------------------------- 4. A-source ablation
    rule("4. ABLATION - does a real A matter?")
    print("\nA estimated from the 52 isolated clips (no transitions exist in them):")
    print(as_frame(A_isolated).round(4).to_string())
    print("\n-> every off-diagonal entry is identical: this is the Laplace prior,")
    print("   not measured behaviour. The circuits are the only real source.")
    save_table(as_frame(A_isolated).round(6), "transition_matrix_isolated.csv",
               index=True)

    degenerate = GaussianHMM(len(ACTIVITIES), N_COMPONENTS, seed=RANDOM_SEED)
    degenerate.set_params(np.full(len(ACTIVITIES), 1 / len(ACTIVITIES)),
                          A_isolated, model.mu, model.var)
    _, stats_deg = evaluate(degenerate, Xci, yci,
                            "circuits with the degenerate A")
    summary_rows.append({"condition": "circuits, degenerate A", **stats_deg})

    # ------------------------------------------------- 5. Baum-Welch
    rule("5. BAUM-WELCH - the same model fitted with the labels withheld")
    bw = GaussianHMM(len(ACTIVITIES), N_COMPONENTS, seed=RANDOM_SEED)
    bw.fit_baum_welch(Xall, n_iter=200, tol=1e-6, init="kmeans")
    mapping = align_states(np.concatenate([bw.viterbi(X) for X in Xall]),
                           np.concatenate(yall), len(ACTIVITIES))
    print(f"converged in {len(bw.loglik_history_)} iterations, "
          f"log-likelihood {bw.loglik_history_[0]:,.2f} -> "
          f"{bw.loglik_history_[-1]:,.2f}")
    print(f"Hungarian state mapping (learned -> true): {mapping}")
    print(f"supervised train log-likelihood {model.log_likelihood(Xall):,.2f}  "
          f"| Baum-Welch {bw.log_likelihood(Xall):,.2f}")

    bw_pred = remap(np.concatenate([bw.viterbi(X) for X in Xall]), mapping)
    bw_rep = per_state_report(np.concatenate(yall), bw_pred, ACTIVITIES)
    bw_stats = overall(np.concatenate(yall), bw_pred)
    print("\nBaum-Welch on the training windows, after alignment")
    print(format_report(bw_rep).to_string(index=False))
    print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in bw_stats.items()))
    save_table(format_report(bw_rep), "eval_baum_welch_train.csv")
    summary_rows.append({"condition": "Baum-Welch (unsupervised, train)",
                         **bw_stats})
    save_table(pd.DataFrame({"iteration": range(1, len(bw.loglik_history_) + 1),
                             "log_likelihood": bw.loglik_history_}),
               "baum_welch_convergence.csv")

    # ----------------------------------------------------------- summary
    rule("SUMMARY")
    summary = pd.DataFrame(summary_rows)
    print(summary.round(4).to_string(index=False))
    save_table(summary.round(4), "summary.csv")

    # ----------------------------------------------------------- figures
    rule("FIGURES")
    from data import load_clean

    clean = load_clean()
    written = [
        viz.plot_sample_signals(clean),
        viz.plot_spectra(clean),
        viz.plot_transition_heatmap(
            A_circuit, title="Transition matrix $A$ (from the circuits)"),
        viz.plot_transition_heatmap(
            A_isolated, path=FIGURES / "03b_transition_matrix_isolated.png",
            title="$A$ from isolated clips - Laplace prior only"),
        viz.plot_confusion(cm_un, title="Unseen session - Viterbi decoding"),
        viz.plot_feature_space(pre.transform(train),
                               train["state"].to_numpy(),
                               mu=model.mu, var=model.var),
        viz.plot_emissions(model.mu, model.var, Z=pre.transform(train),
                           y=train["state"].to_numpy()),
        viz.plot_emission_parameters(model.mu, model.var),
        viz.plot_loglik_curve(bw.loglik_history_),
        viz.plot_separability(train, unseen),
    ]

    # decoded sequence per circuit, with the annotation guard shaded
    for name, X, y in zip(ci_names, Xci, yci):
        grp = (circuits[circuits.recording == name]
               .sort_values("window_index"))
        bounds = segments.loc[segments.recording == name, "start_s"].to_numpy()[1:]
        written.append(viz.plot_decoded_sequence(
            y, model.viterbi(X), t=grp["t_start"].to_numpy(),
            guard=grp["near_boundary"].to_numpy(bool),
            boundaries=list(bounds),
            path=FIGURES / f"05_decoded_{name.lower()}.png",
            title=f"{name} - annotated vs decoded"))

    for p in written:
        print(f"  {p}")
    print(f"\ntables -> {TABLES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
