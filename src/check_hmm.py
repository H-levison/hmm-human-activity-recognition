"""Correctness checks for the from-scratch HMM.

These are properties that must hold if forward, backward, Baum-Welch and
Viterbi are implemented correctly, checked on synthetic data where the answer
is known independently of the real dataset.

Run:  python src/check_hmm.py
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from hmm import GaussianHMM, align_states, remap

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))


def toy_model(seed: int = 0) -> GaussianHMM:
    """A 3-state, 2-D model with well-separated emissions."""
    m = GaussianHMM(n_states=3, n_features=2, seed=seed)
    pi = np.array([0.6, 0.3, 0.1])
    A = np.array([[0.80, 0.15, 0.05],
                  [0.10, 0.80, 0.10],
                  [0.05, 0.25, 0.70]])
    mu = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    var = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    return m.set_params(pi, A, mu, var)


def sample(m: GaussianHMM, T: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Draw a state path and observations from a fitted model."""
    A, pi = m.A, m.pi
    states = np.empty(T, dtype=int)
    states[0] = rng.choice(m.n_states, p=pi)
    for t in range(1, T):
        states[t] = rng.choice(m.n_states, p=A[states[t - 1]])
    X = np.array([rng.normal(m.mu[s], np.sqrt(m.var[s])) for s in states])
    return X, states


# --- 1. emissions integrate to one ----------------------------------------
def test_emission_normalised():
    m = toy_model()
    grid = np.linspace(-12, 16, 1400)
    dx = grid[1] - grid[0]
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    mass = np.exp(m.log_emission(pts)).reshape(len(grid), len(grid), 3)
    totals = mass.sum(axis=(0, 1)) * dx * dx
    check("emission density integrates to 1",
          np.allclose(totals, 1.0, atol=2e-3),
          f"per-state mass {np.round(totals, 5)}")


# --- 2. forward and backward agree on the likelihood ----------------------
def test_forward_backward_consistent():
    m = toy_model()
    rng = np.random.default_rng(1)
    X, _ = sample(m, 60, rng)
    log_B = m.log_emission(X)

    _, ll_fwd = m.forward(log_B)
    log_beta = m.backward(log_B)
    ll_bwd = float(logsumexp(m.log_pi + log_B[0] + log_beta[0]))
    check("forward and backward give the same log-likelihood",
          abs(ll_fwd - ll_bwd) < 1e-8, f"{ll_fwd:.10f} vs {ll_bwd:.10f}")

    # gamma must be a distribution at every step
    log_gamma, log_xi, ll = m.posteriors(log_B)
    check("gamma sums to 1 at every step",
          np.allclose(np.exp(log_gamma).sum(axis=1), 1.0, atol=1e-9))
    check("xi sums to 1 at every step",
          np.allclose(np.exp(log_xi).sum(axis=(1, 2)), 1.0, atol=1e-9))
    # marginalising xi over the destination recovers gamma
    check("xi marginalises to gamma",
          np.allclose(np.exp(log_xi).sum(axis=2), np.exp(log_gamma[:-1]),
                      atol=1e-9))


# --- 3. forward matches brute-force enumeration ---------------------------
def test_forward_matches_bruteforce():
    """Sum over every one of N^T paths explicitly, for a tiny T."""
    m = toy_model()
    rng = np.random.default_rng(2)
    X, _ = sample(m, 6, rng)
    log_B = m.log_emission(X)
    T, N = log_B.shape

    from itertools import product
    terms = []
    for path in product(range(N), repeat=T):
        lp = m.log_pi[path[0]] + log_B[0, path[0]]
        for t in range(1, T):
            lp += m.log_A[path[t - 1], path[t]] + log_B[t, path[t]]
        terms.append(lp)
    brute = float(logsumexp(terms))
    _, fwd = m.forward(log_B)
    check("forward equals brute-force path enumeration",
          abs(brute - fwd) < 1e-9, f"{fwd:.10f} vs {brute:.10f} over {N**T} paths")


# --- 4. Viterbi is the true argmax path ----------------------------------
def test_viterbi_matches_bruteforce():
    m = toy_model()
    rng = np.random.default_rng(3)
    X, _ = sample(m, 7, rng)
    log_B = m.log_emission(X)
    T, N = log_B.shape

    from itertools import product
    best, best_lp = None, -np.inf
    for path in product(range(N), repeat=T):
        lp = m.log_pi[path[0]] + log_B[0, path[0]]
        for t in range(1, T):
            lp += m.log_A[path[t - 1], path[t]] + log_B[t, path[t]]
        if lp > best_lp:
            best, best_lp = path, lp

    got = m.viterbi(X)
    check("Viterbi equals brute-force best path",
          np.array_equal(got, np.array(best)),
          f"{got.tolist()} vs {list(best)}")

    # Viterbi can never beat the total likelihood
    check("Viterbi path probability <= total likelihood",
          best_lp <= m.forward(log_B)[1] + 1e-9)


# --- 5. Baum-Welch increases the likelihood monotonically -----------------
def test_baum_welch_monotone():
    truth = toy_model()
    rng = np.random.default_rng(4)
    seqs = [sample(truth, 120, rng)[0] for _ in range(12)]

    m = GaussianHMM(n_states=3, n_features=2, seed=7)
    m.fit_baum_welch(seqs, n_iter=60, tol=1e-7, init="kmeans")
    hist = np.array(m.loglik_history_)
    diffs = np.diff(hist)

    check("Baum-Welch log-likelihood is non-decreasing",
          bool((diffs >= -1e-6).all()),
          f"{len(hist)} iters, min step {diffs.min():+.2e}, "
          f"{hist[0]:,.1f} -> {hist[-1]:,.1f}")

    check("Baum-Welch log-likelihood beats its own starting point",
          hist[-1] > hist[0], f"{hist[0]:,.1f} -> {hist[-1]:,.1f}")

    # EM should use the whole state space, not collapse states together
    paths = np.concatenate([m.viterbi(X) for X in seqs])
    check("Baum-Welch recovers 3 distinct states",
          len(np.unique(paths)) == 3, f"states used {np.unique(paths).tolist()}")

    # and land near the generating model, once the arbitrary EM labelling is
    # undone against the true paths
    rng2 = np.random.default_rng(4)
    truth_seqs = [sample(truth, 120, rng2) for _ in range(12)]
    pred = np.concatenate([m.viterbi(X) for X, _ in truth_seqs])
    gold = np.concatenate([s for _, s in truth_seqs])
    mapping = align_states(pred, gold, 3)
    acc = (remap(pred, mapping) == gold).mean()
    check("Baum-Welch recovers the generating states after alignment",
          acc > 0.90, f"accuracy {acc:.4f} with mapping {mapping}")


# --- 6. supervised fit recovers known parameters --------------------------
def test_supervised_recovers_params():
    truth = toy_model()
    rng = np.random.default_rng(5)
    data = [sample(truth, 2000, rng) for _ in range(6)]
    seqs = [d[0] for d in data]
    labels = [d[1] for d in data]

    m = GaussianHMM(n_states=3, n_features=2, seed=0)
    m.fit_supervised(seqs, labels, smoothing=1.0)

    check("supervised A recovers the true transition matrix",
          np.allclose(m.A, truth.A, atol=0.02),
          f"max abs error {np.abs(m.A - truth.A).max():.4f}")
    check("supervised mu recovers the true means",
          np.allclose(m.mu, truth.mu, atol=0.08),
          f"max abs error {np.abs(m.mu - truth.mu).max():.4f}")
    check("A rows are row-stochastic",
          np.allclose(m.A.sum(axis=1), 1.0))
    check("no zero transition after Laplace smoothing",
          bool((m.A > 0).all()), f"min entry {m.A.min():.2e}")


# --- 7. decoding accuracy on data from a known model ---------------------
def test_decoding_accuracy():
    truth = toy_model()
    rng = np.random.default_rng(6)
    data = [sample(truth, 400, rng) for _ in range(8)]

    acc_v, acc_m = [], []
    for X, states in data:
        acc_v.append((truth.viterbi(X) == states).mean())
        acc_m.append((truth.map_decode(X) == states).mean())
    check("Viterbi decodes well-separated states accurately",
          np.mean(acc_v) > 0.95, f"Viterbi {np.mean(acc_v):.4f}, "
                                 f"MAP {np.mean(acc_m):.4f}")


# --- 8. Hungarian state alignment ----------------------------------------
def test_align_states():
    true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2] * 5)
    perm = {0: 2, 1: 0, 2: 1}
    pred = np.array([perm[t] for t in true])

    mapping = align_states(pred, true, 3)
    check("align_states inverts a known permutation",
          np.array_equal(remap(pred, mapping), true), f"mapping {mapping}")


# --- 9. numerical stability on long sequences ----------------------------
def test_long_sequence_stability():
    m = toy_model()
    rng = np.random.default_rng(8)
    X, _ = sample(m, 5000, rng)
    ll = m.score(X)
    log_gamma, log_xi, _ = m.posteriors(m.log_emission(X))
    check("5000-step sequence stays finite in log space",
          np.isfinite(ll) and np.isfinite(log_gamma).all()
          and np.isfinite(log_xi).all(),
          f"loglik {ll:,.1f} (a linear-space implementation underflows here)")


if __name__ == "__main__":
    for fn in [test_emission_normalised,
               test_forward_backward_consistent,
               test_forward_matches_bruteforce,
               test_viterbi_matches_bruteforce,
               test_baum_welch_monotone,
               test_supervised_recovers_params,
               test_decoding_accuracy,
               test_align_states,
               test_long_sequence_stability]:
        fn()

    width = max(len(n) for _, n, _ in results)
    for status, name, detail in results:
        line = f"[{status}] {name.ljust(width)}"
        print(f"{line}  {detail}" if detail else line)

    failed = sum(1 for s, _, _ in results if s == FAIL)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    raise SystemExit(1 if failed else 0)
