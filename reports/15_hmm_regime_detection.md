# Optional HMM Regime Detection

This optional module adds a hidden-regime layer for residual behavior. The model is a simple Gaussian hidden Markov model estimated from residual z-scores by triplet. It is intended as a research diagnostic and strategy filter, not as proof that the market has exactly three true states.

## Research purpose

The HMM estimates time-varying probabilities for latent residual regimes:

- mean-reverting regime
- trending regime
- volatile breakdown regime

The practical test is whether residual trades behave differently when the estimated mean-reverting regime probability is high.

## Model specification

The hidden state follows a Markov transition matrix:

\[
P(S_t=j \mid S_{t-1}=i)
\]

The observed residual feature is modeled with Gaussian emissions:

\[
z_t \mid S_t=k \sim \mathcal{N}(\mu_k, \sigma_k^2)
\]

The implementation estimates start probabilities, transition probabilities, state means, and state variances using an expectation-maximization loop with scaled forward-backward probabilities.

## Regime labeling

For the three-state configuration:

- the highest-variance state is labeled volatile breakdown
- among the remaining states, the state with mean closest to zero is labeled mean-reverting
- the remaining state is labeled trending

These are statistical labels based on the residual process. They should be validated against strategy behavior and market context.

## Outputs

- `data/processed/hmm_regime_probability_table.csv`
- `data/processed/hmm_regime_parameters.csv`
- `data/processed/hmm_regime_filtered_trades.csv`
- `data/processed/hmm_strategy_performance_by_regime.csv`
- `figures/regime_timeline_chart.png`
- `figures/performance_by_regime.png`

## Limitations

The placeholder outputs are generated from synthetic residual z-score series. They demonstrate expected file formats and workflow behavior only. They should not be interpreted as real market-data conclusions.

The HMM can overfit if the residual series is short, if regime behavior is unstable, or if the number of states is too high. It should remain optional until validated through walk-forward testing.
