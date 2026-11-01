# -*- coding: utf-8 -*-
"""The prose of the report, separated from its layout.

Keeping the text here means the word budget can be checked without rendering:
    python src/report_text.py
The brief asks for 4 to 5 pages, and with five figures, two tables and the
reference list taking about 19 of the 49 available inches, the prose budget is
roughly 2,300 words at 11 pt.
"""

BG = (
    "After a knee or hip replacement the prescription is almost entirely about "
    "movement, yet the surgeon observes none of it, and adults misestimate "
    "their own activity badly when recall is checked against direct "
    "measurement (Prince et al., 2008). I built this project around passive "
    "adherence monitoring on the phone a patient already carries, where "
    "walking is the mobilisation dose, standing is weight-bearing tolerance, "
    "jumping is impact loading that should raise an alert, and still means "
    "the phone lying on a surface, so not carried at all. That last state is "
    "the hard one, since a system that cannot separate a resting patient from "
    "a phone on a table logs non-wear as clinical inactivity (Choi et al., "
    "2011). The activity itself is never measured, only a noisy "
    "orientation-dependent stream, so the state is hidden; and since "
    "activities persist and change in patterned ways, the sequence carries "
    "information a per-window classifier discards (Rabiner, 1989)."
)

COLLECT = (
    "All recordings are mine, made with Sensor Logger v1.68 on one Xiaomi "
    "2312DRA50G running Android 36, logging accelerometer and gyroscope. The "
    "training session gave 52 files, 13 per activity, phone held in hand at "
    "waist level. A later unseen session gave 12 more, three per activity, "
    "deliberately changed in two ways: a different room, and the phone in a "
    "trouser pocket. I also recorded two continuous 94 s circuits through all "
    "four activities, tapping the annotation button at each change, since "
    "these are the only recordings with real transitions. Still always means "
    "the phone on a flat surface, representing non-wear rather than a "
    "posture. Video documentation is at:"
)

RESAMPLE = (
    "Three timing problems had to be resolved first. The delivered rate is not "
    "the 50 Hz requested, averaging 51.33 Hz with jitter, though a maximum gap "
    "of 0.019 s means no dropouts. The two sensors are written to separate "
    "files on separate nanosecond clocks whose rows are not mutually aligned. "
    "And the annotations use a third clock, 7.96 days behind the sensors\u2019 "
    "Unix epoch, so only intervals between taps are meaningful. I resolve all "
    "three by interpolating both streams onto one uniform 50 Hz grid, so that "
    "values sharing an index are simultaneous. I then keep the central 8.0 s, "
    "which puts every file inside the 5 to 10 s band at 104 s per activity and "
    "removes contamination rather than signal, the first and last 1.5 s "
    "carrying 1.3 to 1.8 times the standard deviation of the middle."
)

WINDOW = (
    "Each clip is cut into 2.0 s windows with a 1.0 s hop, and every part of "
    "that follows from the 50 Hz rate. At 50 Hz a 2.0 s window is exactly 100 "
    "samples, and an integer count matters because it fixes the FFT bins "
    "identically in every window, so a spectral feature means the same thing "
    "throughout. That length also gives 0.5 Hz resolution, fine enough to "
    "separate the walking cadence I measured near 1 Hz from jumping near 3.6 "
    "Hz. The 50 per cent overlap "
    "doubles the window count to 364, at the cost of not localising a "
    "transition better than about a second, visible as lag in Figure 5. It is "
    "close to the 2.56 s window of the UCI benchmark (Anguita et al., 2013)."
)

FEAT_TIME = (
    "A window of 100 samples across eight channels is 800 mostly redundant "
    "numbers, so each is reduced to descriptors of how the signal moves. The "
    "channels are the six raw "
    "axes plus the two Euclidean magnitudes, which are invariant to phone "
    "rotation while the axes are not, and from each I take 14 time-domain and 9 "
    "frequency-domain descriptors, giving 192 per window (Bao & Intille, 2004; "
    "Figo et al., 2010). Variance, standard deviation, RMS and signal "
    "magnitude area measure movement energy, which discriminates most "
    "directly because the activities span three orders of magnitude: 0.003 "
    "m/s\u00b2 for still, 0.048 for standing, 1.26 for walking and 9.59 for "
    "jumping. Mean absolute jerk and the zero-crossing rate describe how "
    "abruptly and how often the signal reverses, skewness and kurtosis "
    "separate the peaked profile of jumping from the near-symmetric one of "
    "walking, and axis-pair correlation catches the vertical-forward "
    "coupling of gait."
)

FEAT_FREQ = (
    "The frequency features come from the Hann-windowed FFT of the mean-removed "
    "signal. Dominant frequency is the clearest physical separator in the set, "
    "since walking sits near 1 Hz, jumping near 3.6 Hz and the static "
    "postures have no periodic component. Spectral energy separates motion "
    "from rest independently of waveform shape, entropy is low for regular "
    "periodicity and high for the noise dominating still, and the centroid "
    "rises with impact. Four band-energy ratios over 0.5\u20131.5, 1.5\u20133, "
    "3\u201310 and 10\u201325 Hz span postural sway up to sensor noise at "
    "Nyquist, starting at 0.5 Hz because the mean-removed signal leaves a "
    "0\u20130.5 Hz band with no bins at all. Rather than justify 192 features "
    "one by one I ranked them by Fisher ratio: acc_mag_mean scores 50.2 and "
    "acc_z_jerk 27.8, so amplitude and jerk dominate, while the frequency "
    "features score lower but generalise far better."
)

NORM = (
    "The 192 features sit on very different scales, since a variance in "
    "(m/s²)² is thousands of times larger than a correlation bounded "
    "in [-1, 1], so I standardise all of them with Z-scores, for three reasons. "
    "PCA maximises variance, so without rescaling the components would be "
    "chosen by whichever features carry the largest raw numbers rather than by "
    "how informative they are (Jolliffe, 2002). My emissions are "
    "diagonal-covariance Gaussians, so a high-variance feature would dominate "
    "the log-likelihood and silence the others. And landing impacts give genuine "
    "extremes, so min-max scaling would be driven by single outliers. The "
    "scaler, pruning and PCA basis are fitted on training windows alone, since "
    "fitting them on everything would leak held-out information."
)

PCA = (
    "Pruning constant and correlated features above |r| = 0.95 leaves 121, "
    "still too many emission parameters for 364 windows, so PCA follows. I did "
    "not choose the component count by variance share: 90 per cent of variance "
    "needs 27 components, which would mean 216 emission parameters from 364 "
    "observations. Instead I took the smallest dimension reaching the accuracy "
    "plateau of a diagonal-covariance Gaussian classifier \u2014 exactly the "
    "HMM\u2019s emission model \u2014 under five-fold cross-validation grouped "
    "by recording: 0.9392 at two dimensions, 0.9971 at three, 1.0000 from four "
    "to twelve. I use four, which retain only 67.9 per cent of variance, so "
    "variance share and discriminative power are clearly different quantities."
)

IMPL = (
    "The model is implemented from scratch in NumPy with no HMM library, and "
    "every routine runs in log space using logsumexp, since a 100-step product "
    "of small Gaussian densities underflows float64 almost immediately and a "
    "linear-space implementation returns zero silently rather than failing "
    "loudly. Across nine modules I implement the forward "
    "and backward recursions, the \u03b3 and \u03be posteriors, Viterbi "
    "decoding (Viterbi, 1967), posterior decoding, closed-form supervised "
    "estimation, Baum-Welch expectation-maximisation (Baum et al., 1970), and "
    "Hungarian matching of learned states onto activity names (Kuhn, 1955). I "
    "also verify nineteen properties against synthetic data, two decisive: the "
    "forward algorithm matches brute-force enumeration over all 3\u2077 = 729 "
    "paths of a three-state model to within 1e-9, and Viterbi returns exactly "
    "the brute-force argmax path. All nineteen pass."
)

ALGOS = (
    "The three algorithms do different jobs and only one produces my evaluation "
    "numbers. The forward algorithm gives P(X | \u03bb), a model-level score I "
    "use to compare the supervised and Baum-Welch fits, but it yields no "
    "per-window label and so cannot produce sensitivity or specificity at all. "
    "Baum-Welch is the learning algorithm when labels are unavailable; mine "
    "are labelled, so closed-form estimation replaces it and Baum-Welch is "
    "run separately to test whether the activities are recoverable without "
    "supervision. Viterbi is what matters for the metrics, since its output "
    "is the predicted state per window, which is what the confusion matrix, "
    "and so every figure in Table 2, is built from. It is also the right "
    "choice rather than a convenient one, because maximising the whole path "
    "suppresses isolated flips that would reach a clinician as spurious "
    "one-second alerts."
)

TRANS = (
    "The 52 isolated clips each hold one activity and therefore zero observed "
    "transitions, so a matrix estimated from them is the smoothing prior and "
    "nothing else, every off-diagonal entry identical at 0.0122. The two "
    "circuits are the only real source, and I count at the segment level: a "
    "segment of D seconds contributes D/hop \u2212 1 self-transitions and one "
    "transition onward, which uses only the order of activities and their "
    "dwell times. Both survive the roughly 2 s reaction error in my "
    "hand-placed taps, whereas per-window counting would inherit it precisely "
    "at the boundaries. I use a smoothing constant of 0.1 rather than Laplace "
    "+1, because the rows hold only 23 to 84 counts and adding 1 to every cell "
    "of a row of 23 halves the dwell times encoded in the diagonal, while 0.1 "
    "keeps the smallest entry at 0.0012, finite in log space. One Circuit_02 "
    "boundary also needed a documented correction, the taps reading standing at "
    "76.1 s where the envelope shows jumping to about 84 s."
)

A_INTERP = (
    "The transition matrix is interpretable in a way the degenerate one is not. "
    "The largest off-diagonal entry is jumping to standing at 0.0897, which is "
    "what happens physically since a person lands and then stands; the smallest "
    "is standing to jumping at 0.0012, which is also right because people take "
    "a step before leaping; and walking to still is near zero at 0.0030, since "
    "a phone is not set down on a table mid-stride. Implied mean dwells, "
    "hop / (1 \u2212 A[k,k]), are 11.3, 19.6, 14.5 and 10.2 s against measured "
    "means of 12.1, 16.9, 11.3 and 11.1 s. In honesty that agreement only "
    "appeared after I reduced the smoothing: at +1 the implied dwells were "
    "about half the measured values, which I would have written up as "
    "evidence against the geometric dwell assumption."
)

UNSEEN_INTRO = (
    "The unseen data is a separate session on a later day, in a different room, "
    "with the phone in a trouser pocket. It went through the same pipeline with "
    "the scaler and PCA basis fitted on the training session alone. For "
    "reference, the same model on twelve held-out training recordings, split by "
    "recording so overlapping windows cannot leak, classifies all 84 windows "
    "correctly."
)

FAILURE = (
    "The model does not generalise well here, and how it fails is more "
    "informative than the headline number. Accuracy falls from 1.0000 on "
    "held-out recordings to 0.6548, but the errors are not scattered: still is "
    "read as standing in 16 of 21 windows and standing as walking in 11 of 21, "
    "while nothing is misread in the opposite direction. A pocket inflated and "
    "widened every amplitude distribution, so each quiet class came to "
    "resemble the next louder one, unseen standing alone spanning the amplitude "
    "range where training still, standing and walking all lie. Walking and "
    "jumping were untouched at sensitivity "
    "1.0000, because a cadence is a shape rather than a scale and survives a "
    "change in carriage as raw amplitude does not."
)

CONTINUOUS = (
    "The circuits are the only recordings with real transitions and so the only "
    "place a decoded sequence can be shown against ground truth. Because "
    "their boundaries carry tap reaction error I exclude windows within three "
    "seconds of a boundary, which raises accuracy from 0.6237 across all 186 "
    "windows to 0.6778 across the 90 clean ones, reported as secondary "
    "results. Substituting the degenerate matrix tests what A contributes: "
    "circuit accuracy is 0.6237 either way, the decoded paths differ on 1 of "
    "186 windows, and on the unseen session Viterbi and posterior decoding "
    "agree on all 84, so the emissions dominate and the transition structure "
    "does interpretive rather than predictive work. Finally, Baum-Welch with "
    "the labels withheld converges monotonically in three iterations and, "
    "once the arbitrary state numbering is resolved, agrees with the true "
    "labels on all 364 training windows — good "
    "evidence of genuinely separable clusters rather than categories the "
    "supervised model was told to find."
)

EASY_HARD = (
    "Jumping was easiest by a wide margin and held sensitivity and specificity "
    "of 1.0000 even on unseen data in a different placement, because it is "
    "the only activity with a distinctive frequency signature near 3.6 Hz as "
    "well as a large amplitude, and walking was robust for the same reason. "
    "Still and standing were the hard pair, and hard in a specific way: on "
    "same-session data they separate perfectly, since a phone on a table "
    "reads the noise floor at 0.003 m/s\u00b2 while a phone in a hand picks "
    "up postural sway at 0.048, but in a pocket that sixteen-fold gap "
    "closes, because body contact transmits micro-movement to a phone "
    "previously decoupled from it. The 50 Hz rate was comfortably sufficient, "
    "Nyquist at 25 Hz sitting far above the roughly 5 Hz ceiling of activity "
    "energy; the real constraint is the quiet end, where still sits at the "
    "noise floor, so every statistic undefined on a constant signal had to be "
    "guarded explicitly in my feature code."
)

IMPROVE = (
    "The highest-value improvement would be orientation- and scale-invariant "
    "features, because the unseen failure is entirely a change in signal scale, "
    "so per-recording normalisation, or features built from spectral shape "
    "rather than absolute amplitude, should recover much of the lost accuracy. "
    "Beyond that, multiple placements would make the emissions "
    "placement-agnostic, more than one participant would stop the model "
    "learning one person\u2019s gait, and a barometer would respond to the "
    "vertical displacement of jumping independently of the accelerometer. A "
    "hidden semi-Markov model would relax the geometric dwell distribution a "
    "standard HMM imposes on every state, which poorly describes activities "
    "whose durations cluster around a mean (Yu, 2010). My A also rests on only "
    "two circuits and twelve observed transitions, though a uniform \u03c0 "
    "changes none of the reported results."
)

CONCLUDE = (
    "In conclusion, a four-state Gaussian hidden Markov model implemented from "
    "scratch separates the four activities perfectly on recordings held out "
    "from the training session, and Baum-Welch recovers the same four states "
    "with no labels at all. On genuinely unseen data recorded on a different "
    "day, in a different room and with the phone in a pocket, accuracy falls "
    "to 0.6548, and the structure of that failure is the most useful result "
    "here: jumping and walking, identified by frequency, survive at "
    "sensitivity 1.0000, while still and standing, distinguished only by "
    "amplitude, collapse into their louder neighbours. A system built on "
    "amplitude features would therefore produce adherence data that looked "
    "plausible while being wrong in one direction, overstating how much a "
    "patient had moved. The remedy is specific: describe the shape of the "
    "motion rather than its size."
)

if __name__ == "__main__":
    names = [n for n in dir() if n.isupper()]
    total = 0
    for n in sorted(names):
        w = len(globals()[n].split())
        total += w
        print(f"{n:14s} {w:4d}")
    print(f"{'TOTAL':14s} {total:4d}")
