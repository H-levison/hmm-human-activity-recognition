"""A Gaussian-emission Hidden Markov Model written from scratch in numpy.

Everything runs in log space via scipy.special.logsumexp: a 100-step sequence
of small Gaussian likelihoods underflows float64 almost immediately otherwise.

Components
----------
pi   (N,)      initial state distribution
A    (N, N)    row-stochastic transition matrix
mu   (N, D)    per-state emission means
var  (N, D)    per-state emission variances (diagonal covariance - full
               covariance needs far more data than ~360 windows affords)

Two fitting routes, both reported:
  fit_supervised  - MLE by counting, since the window labels are known
  fit_baum_welch  - unsupervised EM, with the learned states afterwards
                    Hungarian-matched back onto activity names

Conventions
-----------
A "sequence" is one (T, D) float array of consecutive observation vectors -
in this project, the 7 windows of one 8 s recording, or the full window
sequence of a continuous circuit recording. Methods that learn take a list of
such arrays; methods that decode take a single array.

References
----------
Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970). A maximization
    technique occurring in the statistical analysis of probabilistic functions
    of Markov chains. The Annals of Mathematical Statistics, 41(1), 164-171.
Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from
    incomplete data via the EM algorithm. Journal of the Royal Statistical
    Society: Series B, 39(1), 1-38.
Kuhn, H. W. (1955). The Hungarian method for the assignment problem. Naval
    Research Logistics Quarterly, 2(1-2), 83-97.
Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected
    applications in speech recognition. Proceedings of the IEEE, 77(2),
    257-286.
Viterbi, A. J. (1967). Error bounds for convolutional codes and an
    asymptotically optimum decoding algorithm. IEEE Transactions on
    Information Theory, 13(2), 260-269.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

# Floors that keep the model from walking into -inf or a singular state.
# A variance collapsing to 0 gives a state infinite likelihood on its own
# points; a zero row in A makes Viterbi take log(0) and the path collapses.
VAR_FLOOR = 1e-6
PROB_FLOOR = 1e-300

LOG_2PI = float(np.log(2.0 * np.pi))


def _safe_log(p: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.asarray(p, dtype=float), PROB_FLOOR))


class GaussianHMM:
    """Discrete-state, continuous-observation HMM with diagonal covariance."""

    def __init__(self, n_states: int, n_features: int, seed: int = 0):
        self.n_states = n_states
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)
        self.log_pi: np.ndarray | None = None
        self.log_A: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.loglik_history_: list[float] = []
        self.converged_: bool | None = None
        self.n_iter_: int = 0

    # --- parameters -------------------------------------------------------
    @property
    def pi(self) -> np.ndarray:
        return np.exp(self.log_pi)

    @property
    def A(self) -> np.ndarray:
        return np.exp(self.log_A)

    def _check_fitted(self) -> None:
        if self.log_A is None or self.mu is None:
            raise RuntimeError("model is not fitted yet")

    def set_params(self, pi, A, mu, var) -> "GaussianHMM":
        """Install parameters directly, in probability (not log) space."""
        self.log_pi = _safe_log(np.asarray(pi, float).ravel())
        self.log_A = _safe_log(np.asarray(A, float))
        self.mu = np.asarray(mu, float)
        self.var = np.maximum(np.asarray(var, float), VAR_FLOOR)
        return self

    # --- emissions --------------------------------------------------------
    def log_emission(self, X: np.ndarray) -> np.ndarray:
        """Log p(x_t | z_t = k) for every t, k. Returns shape (T, N).

        Diagonal-covariance Gaussian, written out rather than called from
        scipy so the whole emission model stays visible:

            log N(x; mu, var) = -0.5 * sum_d [ log(2*pi*var_d)
                                               + (x_d - mu_d)^2 / var_d ]
        """
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, float))
        # (T, 1, D) - (1, N, D) -> (T, N, D)
        dev = X[:, None, :] - self.mu[None, :, :]
        quad = (dev ** 2) / self.var[None, :, :]
        norm = LOG_2PI + np.log(self.var)             # (N, D)
        return -0.5 * (quad.sum(axis=2) + norm.sum(axis=1)[None, :])

    # --- inference --------------------------------------------------------
    def forward(self, log_B: np.ndarray) -> tuple[np.ndarray, float]:
        """Log alpha, shape (T, N), and the sequence log-likelihood.

        alpha[t, j] = p(x_1..x_t, z_t = j), recursed in log space.
        """
        T, N = log_B.shape
        log_alpha = np.empty((T, N))
        log_alpha[0] = self.log_pi + log_B[0]
        for t in range(1, T):
            # logsumexp over the previous state i, for each destination j
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + self.log_A,
                                     axis=0) + log_B[t]
        return log_alpha, float(logsumexp(log_alpha[-1]))

    def backward(self, log_B: np.ndarray) -> np.ndarray:
        """Log beta, shape (T, N).

        beta[t, i] = p(x_{t+1}..x_T | z_t = i), with beta[T-1] = 1.
        """
        T, N = log_B.shape
        log_beta = np.zeros((T, N))
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                self.log_A + (log_B[t + 1] + log_beta[t + 1])[None, :], axis=1)
        return log_beta

    def posteriors(self, log_B: np.ndarray):
        """(log_gamma (T,N), log_xi (T-1,N,N), loglik).

        gamma[t, i] = p(z_t = i | X); xi[t, i, j] = p(z_t = i, z_{t+1} = j | X).
        """
        log_alpha, loglik = self.forward(log_B)
        log_beta = self.backward(log_B)
        log_gamma = log_alpha + log_beta - loglik

        T = log_B.shape[0]
        if T > 1:
            log_xi = (log_alpha[:-1, :, None]
                      + self.log_A[None, :, :]
                      + log_B[1:, None, :]
                      + log_beta[1:, None, :]
                      - loglik)
        else:
            log_xi = np.zeros((0, self.n_states, self.n_states))
        return log_gamma, log_xi, loglik

    def score(self, X: np.ndarray) -> float:
        """log P(X | lambda) for one sequence - the Forward algorithm."""
        return self.forward(self.log_emission(X))[1]

    def log_likelihood(self, sequences: list[np.ndarray]) -> float:
        """Total log P(X | lambda) across sequences.

        Used to compare the supervised fit against the Baum-Welch fit.
        """
        return float(sum(self.score(X) for X in sequences))

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Most likely state path for one sequence. Shape (T,).

        This is what produces the predicted labels feeding the confusion
        matrix, and therefore sensitivity, specificity and accuracy. Because
        it maximises the probability of the whole path rather than each frame
        independently, it suppresses isolated single-window flips.
        """
        log_B = self.log_emission(X)
        T, N = log_B.shape

        delta = np.empty((T, N))
        psi = np.zeros((T, N), dtype=int)
        delta[0] = self.log_pi + log_B[0]

        for t in range(1, T):
            trans = delta[t - 1][:, None] + self.log_A     # (N_from, N_to)
            psi[t] = np.argmax(trans, axis=0)
            delta[t] = trans[psi[t], np.arange(N)] + log_B[t]

        path = np.empty(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def map_decode(self, X: np.ndarray) -> np.ndarray:
        """Per-step posterior (MAP) decode, for comparison against Viterbi.

        Maximises expected per-window accuracy but can return a path that the
        transition matrix assigns zero probability to.
        """
        log_gamma, _, _ = self.posteriors(self.log_emission(X))
        return np.argmax(log_gamma, axis=1)

    # --- learning ---------------------------------------------------------
    def fit_supervised(self, sequences: list[np.ndarray],
                       label_seqs: list[np.ndarray], smoothing: float = 1.0):
        """Closed-form MLE from labelled windows.

        pi and A from Laplace-smoothed state and transition counts; mu and var
        from the windows assigned to each state. Smoothing is not cosmetic: an
        unobserved transition would otherwise be exactly 0, and Viterbi would
        then be unable to ever cross that edge no matter what the emissions say.
        """
        N, D = self.n_states, self.n_features

        start_counts = np.full(N, smoothing)
        trans_counts = np.full((N, N), smoothing)
        for labels in label_seqs:
            labels = np.asarray(labels, dtype=int)
            start_counts[labels[0]] += 1.0
            for i, j in zip(labels[:-1], labels[1:]):
                trans_counts[i, j] += 1.0

        pi = start_counts / start_counts.sum()
        A = trans_counts / trans_counts.sum(axis=1, keepdims=True)

        X_all = np.vstack([np.atleast_2d(X) for X in sequences])
        y_all = np.concatenate([np.asarray(l, int).ravel() for l in label_seqs])
        if len(X_all) != len(y_all):
            raise ValueError("observation and label counts disagree")

        mu = np.zeros((N, D))
        var = np.zeros((N, D))
        for k in range(N):
            Xk = X_all[y_all == k]
            if len(Xk) == 0:
                raise ValueError(f"state {k} has no labelled observations")
            mu[k] = Xk.mean(axis=0)
            var[k] = Xk.var(axis=0)

        self.set_params(pi, A, mu, var)
        self.loglik_history_ = [self.log_likelihood(sequences)]
        return self

    def _init_params(self, sequences: list[np.ndarray], init: str,
                     self_transition: float = 0.9):
        """Starting point for EM."""
        N = self.n_states
        X_all = np.vstack([np.atleast_2d(X) for X in sequences])

        if init == "kmeans":
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=N, n_init=10,
                        random_state=int(self.rng.integers(1 << 31)))
            assign = km.fit_predict(X_all)
            mu = km.cluster_centers_
            var = np.vstack([
                X_all[assign == k].var(axis=0) if (assign == k).sum() > 1
                else X_all.var(axis=0)
                for k in range(N)
            ])
        elif init == "random":
            idx = self.rng.choice(len(X_all), size=N, replace=False)
            mu = X_all[idx]
            var = np.tile(X_all.var(axis=0), (N, 1))
        else:
            raise ValueError(f"unknown init {init!r}")

        # Diagonal-dominant A: activities persist for many windows, so a
        # uniform start would fight the data for the first several iterations.
        off = (1.0 - self_transition) / (N - 1)
        A = np.full((N, N), off)
        np.fill_diagonal(A, self_transition)
        pi = np.full(N, 1.0 / N)

        self.set_params(pi, A, mu, var)

    def fit_baum_welch(self, sequences: list[np.ndarray], n_iter: int = 100,
                       tol: float = 1e-6, init: str = "kmeans",
                       verbose: bool = False):
        """Expectation-Maximisation, labels withheld.

        Stopping criterion
        ------------------
        EM halts when the RELATIVE improvement in total log-likelihood falls
        below `tol`:

            (L_t - L_{t-1}) / |L_{t-1}|  <  tol

        Relative rather than absolute, because the total log-likelihood scales
        with the number of sequences and their length - an absolute threshold
        of 1e-4 means something quite different for 40 sequences than for 400,
        so the same `tol` would stop too early or too late depending on the
        dataset. The relative form is scale-free.

        EM is guaranteed to be monotone non-decreasing, so a negative step
        indicates a bug rather than a search that needs restarting; the loop
        therefore raises instead of silently continuing if one appears.

        Returns self; the log-likelihood per iteration is kept in
        `loglik_history_` so the monotone convergence curve can be plotted as
        evidence that the implementation is correct.
        """
        N, D = self.n_states, self.n_features
        sequences = [np.atleast_2d(np.asarray(X, float)) for X in sequences]
        self._init_params(sequences, init)
        self.loglik_history_ = []

        prev = -np.inf
        for it in range(n_iter):
            start_acc = np.zeros(N)
            trans_acc = np.zeros((N, N))
            gamma_acc = np.zeros(N)          # for emissions (all t)
            gamma_trans = np.zeros(N)        # for A denominator (t < T-1)
            mean_acc = np.zeros((N, D))
            sq_acc = np.zeros((N, D))
            total = 0.0

            # --- E step
            for X in sequences:
                log_B = self.log_emission(X)
                log_gamma, log_xi, loglik = self.posteriors(log_B)
                total += loglik

                gamma = np.exp(log_gamma)                 # (T, N)
                start_acc += gamma[0]
                gamma_acc += gamma.sum(axis=0)
                if len(log_xi):
                    trans_acc += np.exp(log_xi).sum(axis=0)
                    gamma_trans += gamma[:-1].sum(axis=0)
                mean_acc += gamma.T @ X                   # (N, D)
                sq_acc += gamma.T @ (X ** 2)

            self.loglik_history_.append(total)

            # --- M step
            pi = start_acc / start_acc.sum()
            denom = np.maximum(gamma_trans, PROB_FLOOR)[:, None]
            A = trans_acc / denom
            A = A / A.sum(axis=1, keepdims=True)

            w = np.maximum(gamma_acc, PROB_FLOOR)[:, None]
            mu = mean_acc / w
            var = sq_acc / w - mu ** 2

            self.set_params(pi, A, mu, var)

            if it > 0:
                step = total - prev
                if step < -1e-6 * abs(prev):
                    raise RuntimeError(
                        f"EM log-likelihood decreased at iteration {it + 1} "
                        f"({prev:,.6f} -> {total:,.6f}); this is impossible for "
                        f"a correct E/M pair and indicates a bug")
                relative = step / max(abs(prev), PROB_FLOOR)
                if verbose:
                    print(f"  iter {it + 1:3d}  loglik {total:,.3f}  "
                          f"rel. improvement {relative:.2e}")
                if relative < tol:
                    self.converged_ = True
                    break
            elif verbose:
                print(f"  iter {it + 1:3d}  loglik {total:,.3f}")
            prev = total
        else:
            self.converged_ = False

        self.n_iter_ = len(self.loglik_history_)
        return self


def align_states(pred: np.ndarray, true: np.ndarray, n_states: int) -> dict:
    """Map unsupervised state ids onto activity ids.

    Baum-Welch state numbering is arbitrary - EM has no idea which cluster we
    chose to call "Walking". A contingency table of learned-vs-true is
    resolved with the Hungarian algorithm, which picks the one-to-one mapping
    maximising total agreement. Reporting any metric before doing this would
    understate accuracy by an arbitrary permutation.

    Returns {learned_state: true_state}.
    """
    pred = np.asarray(pred, int).ravel()
    true = np.asarray(true, int).ravel()

    table = np.zeros((n_states, n_states), dtype=int)
    for p, t in zip(pred, true):
        table[p, t] += 1

    rows, cols = linear_sum_assignment(-table)       # maximise agreement
    return {int(r): int(c) for r, c in zip(rows, cols)}


def remap(pred: np.ndarray, mapping: dict) -> np.ndarray:
    """Apply an align_states mapping to a predicted path."""
    return np.array([mapping[int(p)] for p in np.asarray(pred).ravel()])
