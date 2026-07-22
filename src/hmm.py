from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 3          # mean-reverting, trending, volatile-breakdown
    max_iter: int = 100        # cap on Baum-Welch training iterations
    tolerance: float = 1e-5    # stop early once log-likelihood improvement is smaller than this
    variance_floor: float = 1e-6   # prevents any state's variance from collapsing to exactly zero
    transition_stickiness: float = 0.90  # initial guess: states tend to persist rather than flip every day
    random_state: Optional[int] = 17
    feature_column: str = "residual_z_score"


@dataclass(frozen=True)
class GaussianHMMResult:
    start_probabilities: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    log_likelihoods: tuple[float, ...]
    state_labels: tuple[str, ...]
    feature_column: str
    config: HMMConfig


def fit_gaussian_hmm(values: Sequence[float] | np.ndarray | pd.Series, config: Optional[HMMConfig] = None) -> GaussianHMMResult:
    """Fits a Gaussian Hidden Markov Model to a 1D observation series
    (the residual z-score) using the Baum-Welch algorithm -- an
    Expectation-Maximization procedure that alternates between two steps
    until the fit stops improving:

      E-step ("how likely is each hidden state, given the data and the
      current parameter guess?"): run the forward and backward
      algorithms to get `gamma` (the probability of being in each state
      on each day) and `xi` (the probability of transitioning between
      each pair of states from one day to the next).

      M-step ("given those probabilities, what parameters best explain
      them?"): re-estimate the start probabilities, transition matrix,
      and each state's Gaussian mean/variance as weighted averages, using
      `gamma`/`xi` as the weights.

    Each iteration is guaranteed not to decrease the data's overall
    log-likelihood under the model (a core EM property), which is why
    tracking `log_likelihoods` and stopping once it plateaus is a valid
    convergence check.
    """
    cfg = config or HMMConfig()
    _validate_config(cfg)
    obs = _coerce_observations(values)
    if obs.shape[0] < cfg.n_states * 3:
        raise ValueError("at least three observations per state are required")

    start_prob, transition, means, variances = _initial_parameters(obs, cfg)
    log_likelihoods: list[float] = []

    for _ in range(cfg.max_iter):
        # E-step: score how well the current parameters explain the data
        emissions = gaussian_emission_matrix(obs, means, variances, variance_floor=cfg.variance_floor)
        alpha, scales, log_likelihood = forward_algorithm(emissions, start_prob, transition)
        beta = backward_algorithm(emissions, transition, scales)
        gamma = _posterior_probabilities(alpha, beta)
        xi = _expected_transitions(alpha, beta, emissions, transition, scales)

        # M-step: re-estimate every parameter as a gamma/xi-weighted
        # average rather than a hard assignment -- each observation
        # contributes partially to every state, in proportion to how
        # likely it is to have come from that state
        start_prob = gamma[0] / gamma[0].sum()
        transition = _normalize_rows(xi.sum(axis=0), floor=1e-12)
        weights = gamma.sum(axis=0)
        means = (gamma.T @ obs) / np.maximum(weights, 1e-12)
        diff2 = (obs[:, None] - means[None, :]) ** 2
        variances = np.sum(gamma * diff2, axis=0) / np.maximum(weights, 1e-12)
        variances = np.maximum(variances, cfg.variance_floor)
        log_likelihoods.append(float(log_likelihood))

        if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < cfg.tolerance:
            break

    labels = infer_regime_labels(means, variances)
    return GaussianHMMResult(
        start_probabilities=start_prob,
        transition_matrix=transition,
        means=means,
        variances=variances,
        log_likelihoods=tuple(log_likelihoods),
        state_labels=tuple(labels),
        feature_column=cfg.feature_column,
        config=cfg,
    )


def gaussian_emission_matrix(
    values: Sequence[float] | np.ndarray | pd.Series,
    means: Sequence[float] | np.ndarray,
    variances: Sequence[float] | np.ndarray,
    variance_floor: float = 1e-6,
) -> np.ndarray:
    """For every observation and every state, computes how likely that
    observation would be if it came from that state's Gaussian
    distribution (the normal-distribution probability density formula).
    Returns an (n_observations x n_states) matrix -- the "emission
    probabilities" the forward/backward algorithms are built on.

    Values are floored at a tiny positive number rather than allowed to
    reach exactly zero, since the forward/backward recursions multiply
    many of these together and an exact zero would permanently kill that
    entire computation path.
    """
    obs = _coerce_observations(values)
    mu = np.asarray(means, dtype=float)
    var = np.maximum(np.asarray(variances, dtype=float), variance_floor)
    if mu.ndim != 1 or var.ndim != 1 or mu.shape[0] != var.shape[0]:
        raise ValueError("means and variances must be one-dimensional arrays with the same length")
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
    exponent = -0.5 * ((obs[:, None] - mu[None, :]) ** 2) / var[None, :]
    return np.maximum(coeff[None, :] * np.exp(exponent), 1e-300)


def forward_algorithm(
    emission_probabilities: np.ndarray,
    start_probabilities: Sequence[float] | np.ndarray,
    transition_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """The forward pass: for each day, how likely is it that the process
    is in each state, given everything observed so far (today included)?

    `alpha[t]` combines yesterday's alpha (weighted by how likely each
    state transitions into each other state) with today's emission
    probability -- "given where we probably were yesterday and how the
    states usually transition, and given what we actually saw today,
    where are we probably today?"

    Without rescaling, alpha would shrink toward zero over many days
    (it's a product of many probabilities each less than 1) until it
    underflows to exactly zero in floating point. `scales[t]` is exactly
    the normalizing constant that keeps alpha summing to 1 at every step,
    and the sum of the logs of those scales is mathematically equal to
    the overall log-likelihood of the whole observation sequence -- which
    is how `log_likelihood` is obtained essentially for free as a
    byproduct of the rescaling that was needed anyway.
    """
    emissions = _coerce_emissions(emission_probabilities)
    start = _normalize_vector(np.asarray(start_probabilities, dtype=float))
    transition = _coerce_transition_matrix(transition_matrix, emissions.shape[1])

    alpha = np.zeros_like(emissions, dtype=float)
    scales = np.zeros(emissions.shape[0], dtype=float)
    alpha[0] = start * emissions[0]
    scales[0] = max(float(alpha[0].sum()), 1e-300)
    alpha[0] /= scales[0]

    for t in range(1, emissions.shape[0]):
        alpha[t] = (alpha[t - 1] @ transition) * emissions[t]
        scales[t] = max(float(alpha[t].sum()), 1e-300)
        alpha[t] /= scales[t]

    log_likelihood = float(np.sum(np.log(scales)))
    return alpha, scales, log_likelihood


def backward_algorithm(
    emission_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    scales: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """The mirror image of the forward pass: for each day, how likely is
    everything observed AFTER that day, given the process was in each
    state on that day? Runs backward from the last observation to the
    first. Combined with `alpha` (see `_posterior_probabilities`), this
    gives the probability of each state on each day using ALL the data
    (past and future), not just the data up to that point -- a genuinely
    better estimate than the forward pass alone provides, since it's
    informed by what happened afterward too.

    Reuses the exact same `scales` computed during the forward pass to
    stay numerically consistent with it -- forward and backward must be
    rescaled by the same factors for `alpha * beta` to combine correctly
    into a valid probability later.
    """
    emissions = _coerce_emissions(emission_probabilities)
    transition = _coerce_transition_matrix(transition_matrix, emissions.shape[1])
    scale_arr = np.maximum(np.asarray(scales, dtype=float), 1e-300)
    beta = np.zeros_like(emissions, dtype=float)
    beta[-1] = 1.0

    for t in range(emissions.shape[0] - 2, -1, -1):
        beta[t] = transition @ (emissions[t + 1] * beta[t + 1])
        beta[t] /= scale_arr[t + 1]
    return beta


def viterbi_decode(
    values: Sequence[float] | np.ndarray | pd.Series,
    result: GaussianHMMResult,
) -> np.ndarray:
    """Finds the single most likely SEQUENCE of hidden states for the
    whole observation history (as opposed to `posterior_state_probabilities`,
    which gives the most likely state independently on each individual
    day). Dynamic programming: at each day, for each state, remembers the
    best-scoring path that could have led there (`backpointers`), then
    walks that chain of pointers backward from the single best final
    state to reconstruct the full path.

    Works entirely in log-probabilities (`log_start`, `log_transition`,
    `log_emissions`) rather than raw probabilities: this turns the
    products used elsewhere in this module into sums, which is both
    faster and avoids the same underflow problem the forward algorithm's
    rescaling exists to prevent.
    """
    obs = _coerce_observations(values)
    emissions = gaussian_emission_matrix(obs, result.means, result.variances, variance_floor=result.config.variance_floor)
    log_start = np.log(np.maximum(result.start_probabilities, 1e-300))
    log_transition = np.log(np.maximum(result.transition_matrix, 1e-300))
    log_emissions = np.log(np.maximum(emissions, 1e-300))

    n_obs, n_states = log_emissions.shape
    scores = np.zeros((n_obs, n_states), dtype=float)
    backpointers = np.zeros((n_obs, n_states), dtype=int)
    scores[0] = log_start + log_emissions[0]

    for t in range(1, n_obs):
        # for every state at time t, find which state at t-1 would have
        # produced the highest-scoring path into it, and remember that
        # choice (backpointers) so the full path can be reconstructed later
        candidate = scores[t - 1][:, None] + log_transition
        backpointers[t] = np.argmax(candidate, axis=0)
        scores[t] = np.max(candidate, axis=0) + log_emissions[t]

    # start from whichever final state has the best overall score, then
    # follow the backpointers in reverse to recover the entire optimal path
    path = np.zeros(n_obs, dtype=int)
    path[-1] = int(np.argmax(scores[-1]))
    for t in range(n_obs - 2, -1, -1):
        path[t] = backpointers[t + 1, path[t + 1]]
    return path


def posterior_state_probabilities(
    values: Sequence[float] | np.ndarray | pd.Series,
    result: GaussianHMMResult,
) -> tuple[pd.DataFrame, float]:
    # Runs forward + backward and combines them into per-day, per-state
    # probabilities -- this is the "soft" regime read used by
    # make_regime_probability_table (a probability for every state every
    # day), as distinct from viterbi_decode's "hard" single best path.
    obs = _coerce_observations(values)
    emissions = gaussian_emission_matrix(obs, result.means, result.variances, variance_floor=result.config.variance_floor)
    alpha, scales, log_likelihood = forward_algorithm(emissions, result.start_probabilities, result.transition_matrix)
    beta = backward_algorithm(emissions, result.transition_matrix, scales)
    gamma = _posterior_probabilities(alpha, beta)
    columns = [f"state_{state}_probability" for state in range(result.config.n_states)]
    return pd.DataFrame(gamma, columns=columns), float(log_likelihood)


def infer_regime_labels(means: Sequence[float] | np.ndarray, variances: Sequence[float] | np.ndarray) -> list[str]:
    """Maps the HMM's anonymous numbered states (0, 1, 2 -- the fitting
    process has no idea what they "mean") onto the three human-readable
    regime names used throughout this project, using two simple
    heuristics specific to the 3-state case:
      - whichever state has the largest variance is labeled
        "volatile_breakdown" -- a regime characterized by large, erratic
        swings in the residual.
      - of the remaining two, whichever has a mean closest to zero is
        "mean_reverting" -- the residual sitting calmly near its typical
        (near-zero) level.
      - the last one is "trending" -- a sustained, non-erratic drift away
        from zero.
    Falls back to generic "state_N" labels for any n_states other than 3,
    since this specific heuristic is written for exactly three regimes.
    """
    mu = np.asarray(means, dtype=float)
    var = np.asarray(variances, dtype=float)
    n_states = mu.shape[0]
    labels = [f"state_{idx}" for idx in range(n_states)]
    if n_states != 3:
        return labels

    breakdown = int(np.argmax(var))
    labels[breakdown] = "volatile_breakdown"
    remaining = [idx for idx in range(n_states) if idx != breakdown]
    mean_reverting = remaining[int(np.argmin(np.abs(mu[remaining])))]
    labels[mean_reverting] = "mean_reverting"
    for idx in remaining:
        if idx != mean_reverting:
            labels[idx] = "trending"
    return labels


def make_regime_probability_table(
    residuals: pd.DataFrame,
    result: GaussianHMMResult,
    date_column: str = "date",
    value_column: Optional[str] = None,
    triplet_id: Optional[str] = None,
    method: str = "gaussian_hmm",
) -> pd.DataFrame:
    # Assembles a single, complete output table combining the soft
    # (posterior probability) and hard (Viterbi) regime reads with the
    # human-readable labels attached -- this is the table
    # scripts/run_universe_pipeline.py and the dashboard actually consume.
    feature_col = value_column or result.feature_column
    required = {date_column, feature_col}
    missing = required.difference(residuals.columns)
    if missing:
        raise KeyError(f"missing residual columns: {sorted(missing)}")

    frame = residuals.sort_values(date_column).reset_index(drop=True).copy()
    probabilities, log_likelihood = posterior_state_probabilities(frame[feature_col], result)
    viterbi_path = viterbi_decode(frame[feature_col], result)
    output = frame.loc[:, [date_column]].copy().rename(columns={date_column: "date"})
    output["triplet_id"] = triplet_id if triplet_id is not None else _single_value(frame, "triplet_id", default="UNKNOWN")
    output["method"] = method
    output["model_type"] = "gaussian_hmm"
    output["feature_column"] = feature_col
    output["feature_value"] = frame[feature_col].astype(float).to_numpy()
    output = pd.concat([output, probabilities], axis=1)

    for state, label in enumerate(result.state_labels):
        output[f"state_{state}_label"] = label
        output[f"{label}_probability"] = probabilities[f"state_{state}_probability"]

    state_prob_cols = [f"state_{state}_probability" for state in range(result.config.n_states)]
    most_likely_states = probabilities[state_prob_cols].to_numpy().argmax(axis=1)
    output["most_likely_state"] = most_likely_states
    output["most_likely_regime"] = [result.state_labels[state] for state in most_likely_states]
    output["viterbi_state"] = viterbi_path
    output["viterbi_regime"] = [result.state_labels[state] for state in viterbi_path]
    output["log_likelihood"] = log_likelihood
    output["n_states"] = result.config.n_states
    return output


def fit_hmm_by_triplet(
    residuals: pd.DataFrame,
    value_column: str = "residual_z_score",
    date_column: str = "date",
    triplet_column: str = "triplet_id",
    config: Optional[HMMConfig] = None,
) -> dict[str, pd.DataFrame | dict[str, GaussianHMMResult]]:
    # Batch entry point: fits an entirely independent HMM per triplet
    # (no information shared across triplets) and skips any triplet with
    # too little history to reliably fit the requested number of states
    # -- this is what makes regime detection scale to however many
    # triplets are actually present, rather than a fixed hand-picked
    # subset.
    required = {date_column, triplet_column, value_column}
    missing = required.difference(residuals.columns)
    if missing:
        raise KeyError(f"missing residual columns: {sorted(missing)}")

    cfg = config or HMMConfig(feature_column=value_column)
    probability_frames = []
    parameter_frames = []
    models: dict[str, GaussianHMMResult] = {}

    for triplet_id, group in residuals.dropna(subset=[value_column]).groupby(triplet_column, sort=True):
        ordered = group.sort_values(date_column).reset_index(drop=True)
        if ordered.shape[0] < cfg.n_states * 3:
            continue
        result = fit_gaussian_hmm(ordered[value_column], cfg)
        models[str(triplet_id)] = result
        probability_frames.append(
            make_regime_probability_table(
                ordered,
                result,
                date_column=date_column,
                value_column=value_column,
                triplet_id=str(triplet_id),
            )
        )
        parameter_frames.append(hmm_parameter_frame(result, triplet_id=str(triplet_id)))

    probabilities = pd.concat(probability_frames, ignore_index=True) if probability_frames else pd.DataFrame()
    parameters = pd.concat(parameter_frames, ignore_index=True) if parameter_frames else pd.DataFrame()
    return {"models": models, "regime_probabilities": probabilities, "regime_parameters": parameters}


def hmm_parameter_frame(result: GaussianHMMResult, triplet_id: str = "UNKNOWN") -> pd.DataFrame:
    # A readable table of the fitted model's actual parameters (one row
    # per state) -- lets you inspect what each regime "looks like"
    # (its mean/variance) and how sticky it is (its transition
    # probabilities) without digging into the raw numpy arrays.
    rows = []
    for state in range(result.config.n_states):
        row = {
            "triplet_id": triplet_id,
            "model_type": "gaussian_hmm",
            "state": state,
            "regime_label": result.state_labels[state],
            "mean": float(result.means[state]),
            "variance": float(result.variances[state]),
            "start_probability": float(result.start_probabilities[state]),
            "log_likelihood": float(result.log_likelihoods[-1]) if result.log_likelihoods else np.nan,
        }
        for next_state in range(result.config.n_states):
            row[f"transition_to_state_{next_state}"] = float(result.transition_matrix[state, next_state])
        rows.append(row)
    return pd.DataFrame(rows)


def apply_regime_trade_filter(
    events: pd.DataFrame,
    regime_probabilities: pd.DataFrame,
    threshold: float = 0.60,
    event_date_column: str = "event_date",
) -> pd.DataFrame:
    # Joins each trade event to that triplet's regime read on the same
    # date, and flags whether the mean-reverting probability cleared the
    # threshold -- an optional additional filter layer on top of the
    # logistic model's own probability threshold.
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    event_required = {"event_id", "triplet_id", event_date_column}
    regime_required = {"triplet_id", "date", "mean_reverting_probability"}
    missing_events = event_required.difference(events.columns)
    missing_regime = regime_required.difference(regime_probabilities.columns)
    if missing_events:
        raise KeyError(f"missing event columns: {sorted(missing_events)}")
    if missing_regime:
        raise KeyError(f"missing regime columns: {sorted(missing_regime)}")

    left = events.copy()
    right = regime_probabilities.copy()
    left[event_date_column] = pd.to_datetime(left[event_date_column])
    right["date"] = pd.to_datetime(right["date"])
    merged = left.merge(
        right.loc[:, ["triplet_id", "date", "mean_reverting_probability", "most_likely_regime", "viterbi_regime"]],
        left_on=["triplet_id", event_date_column],
        right_on=["triplet_id", "date"],
        how="left",
    )
    merged["regime_probability_threshold"] = float(threshold)
    merged["allowed_by_regime_filter"] = merged["mean_reverting_probability"].gt(threshold)
    return merged.drop(columns=["date"], errors="ignore")


def summarize_strategy_performance_by_regime(
    trades: pd.DataFrame,
    regime_probabilities: pd.DataFrame,
    threshold: float = 0.60,
) -> pd.DataFrame:
    # Splits realized trade performance into two buckets -- trades the
    # regime filter would have allowed vs. would have blocked -- to check
    # whether the HMM's regime read is actually informative about which
    # trades do better, independent of whatever filter is actually live
    # in the backtest.
    required = {"event_id", "triplet_id", "event_date", "strategy", "net_pnl", "label"}
    missing = required.difference(trades.columns)
    if missing:
        raise KeyError(f"missing trade columns: {sorted(missing)}")
    merged = apply_regime_trade_filter(trades, regime_probabilities, threshold=threshold)
    if merged.empty:
        return pd.DataFrame()
    merged["regime_bucket"] = np.where(
        merged["allowed_by_regime_filter"],
        "mean_reverting_probability_above_threshold",
        "mean_reverting_probability_below_threshold",
    )
    summary = (
        merged.groupby(["strategy", "regime_bucket"], dropna=False)
        .agg(
            trade_count=("event_id", "count"),
            average_mean_reverting_probability=("mean_reverting_probability", "mean"),
            success_rate=("label", "mean"),
            net_pnl=("net_pnl", "sum"),
            average_net_pnl=("net_pnl", "mean"),
            turnover=("turnover", "sum") if "turnover" in merged.columns else ("net_pnl", "count"),
        )
        .reset_index()
    )
    summary["regime_probability_threshold"] = float(threshold)
    return summary


def plot_regime_timeline(
    regime_probabilities: pd.DataFrame,
    output_path: str | Path,
    triplet_id: Optional[str] = None,
) -> Path:
    # A simple line chart of the three regime probabilities over time for
    # one triplet -- the fastest way to visually sanity-check that the
    # fitted HMM's regime calls line up with what the residual was
    # actually doing.
    required = {"date", "triplet_id", "mean_reverting_probability", "trending_probability", "volatile_breakdown_probability"}
    missing = required.difference(regime_probabilities.columns)
    if missing:
        raise KeyError(f"missing regime columns: {sorted(missing)}")
    frame = regime_probabilities.copy()
    if triplet_id is None:
        triplet_id = str(frame["triplet_id"].iloc[0])
    frame = frame.loc[frame["triplet_id"].astype(str).eq(str(triplet_id))].copy()
    if frame.empty:
        raise ValueError("no regime rows found for selected triplet")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame["date"], frame["mean_reverting_probability"], label="mean-reverting")
    ax.plot(frame["date"], frame["trending_probability"], label="trending")
    ax.plot(frame["date"], frame["volatile_breakdown_probability"], label="volatile breakdown")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Regime probability")
    ax.set_title(f"Regime probabilities: {triplet_id}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_performance_by_regime(performance: pd.DataFrame, output_path: str | Path) -> Path:
    required = {"strategy", "regime_bucket", "net_pnl"}
    missing = required.difference(performance.columns)
    if missing:
        raise KeyError(f"missing performance columns: {sorted(missing)}")
    frame = performance.copy()
    path = _prepare_output_path(output_path)

    labels = [f"{row.strategy}\n{row.regime_bucket}" for row in frame.itertuples(index=False)]
    x = np.arange(frame.shape[0])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, frame["net_pnl"].astype(float))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Net PnL units")
    ax.set_title("Strategy performance by inferred regime")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _posterior_probabilities(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    # gamma[t] = P(state at time t | ALL observations) is proportional to
    # alpha[t] * beta[t] -- forward gives "probability given the past,"
    # backward gives "probability given the future," and their product
    # (renormalized to sum to 1) combines both into the full-information
    # estimate.
    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
    return gamma


def _expected_transitions(
    alpha: np.ndarray,
    beta: np.ndarray,
    emissions: np.ndarray,
    transition: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    # xi[t, i, j] = P(state i at time t AND state j at time t+1 | all
    # observations) -- the transition-level equivalent of gamma. Used in
    # the M-step to re-estimate the transition matrix: summing xi over
    # time and normalizing gives the expected fraction of transitions
    # that went from each state to each other state.
    n_obs, n_states = alpha.shape
    xi = np.zeros((n_obs - 1, n_states, n_states), dtype=float)
    for t in range(n_obs - 1):
        numerator = alpha[t][:, None] * transition * emissions[t + 1][None, :] * beta[t + 1][None, :]
        denominator = max(float(numerator.sum()), 1e-300)
        xi[t] = numerator / denominator
    return xi


def _initial_parameters(obs: np.ndarray, cfg: HMMConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Starting guess for Baum-Welch to refine, rather than random
    # initialization: state means are spread across the data's own
    # quantiles (so, e.g. with 3 states, one starts near the low end of
    # the observed range, one in the middle, one near the high end --
    # already roughly separated instead of all starting in the same
    # place), and the transition matrix starts biased toward staying in
    # the same state (`transition_stickiness`), matching the real-world
    # expectation that regimes persist for a while rather than flipping
    # every single day.
    n_states = cfg.n_states
    quantiles = np.linspace(0.15, 0.85, n_states)
    means = np.quantile(obs, quantiles)
    global_var = max(float(np.var(obs)), cfg.variance_floor)
    variances = np.full(n_states, global_var, dtype=float)
    start_prob = np.full(n_states, 1.0 / n_states, dtype=float)
    off_diag = (1.0 - cfg.transition_stickiness) / max(n_states - 1, 1)
    transition = np.full((n_states, n_states), off_diag, dtype=float)
    np.fill_diagonal(transition, cfg.transition_stickiness)
    return start_prob, _normalize_rows(transition), means.astype(float), variances


def _normalize_vector(values: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    clean = np.maximum(np.asarray(values, dtype=float), floor)
    return clean / clean.sum()


def _normalize_rows(values: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    clean = np.maximum(np.asarray(values, dtype=float), floor)
    return clean / clean.sum(axis=1, keepdims=True)


def _coerce_observations(values: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    obs = np.asarray(values, dtype=float).reshape(-1)
    obs = obs[np.isfinite(obs)]
    if obs.shape[0] == 0:
        raise ValueError("at least one finite observation is required")
    return obs


def _coerce_emissions(emission_probabilities: np.ndarray) -> np.ndarray:
    emissions = np.asarray(emission_probabilities, dtype=float)
    if emissions.ndim != 2 or emissions.shape[0] == 0 or emissions.shape[1] == 0:
        raise ValueError("emission probabilities must be a non-empty two-dimensional matrix")
    if not np.isfinite(emissions).all() or (emissions < 0.0).any():
        raise ValueError("emission probabilities must be finite and non-negative")
    return np.maximum(emissions, 1e-300)


def _coerce_transition_matrix(matrix: np.ndarray, n_states: int) -> np.ndarray:
    transition = np.asarray(matrix, dtype=float)
    if transition.shape != (n_states, n_states):
        raise ValueError("transition matrix shape does not match number of states")
    if not np.isfinite(transition).all() or (transition < 0.0).any():
        raise ValueError("transition matrix must contain finite non-negative values")
    return _normalize_rows(transition)


def _validate_config(config: HMMConfig) -> None:
    if config.n_states < 2:
        raise ValueError("n_states must be at least 2")
    if config.max_iter < 1:
        raise ValueError("max_iter must be positive")
    if config.tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if config.variance_floor <= 0.0:
        raise ValueError("variance_floor must be positive")
    if not 0.0 < config.transition_stickiness < 1.0:
        raise ValueError("transition_stickiness must be between 0 and 1")


def _single_value(frame: pd.DataFrame, column: str, default: str) -> str:
    if column not in frame.columns or frame[column].dropna().empty:
        return default
    return str(frame[column].dropna().iloc[0])


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
