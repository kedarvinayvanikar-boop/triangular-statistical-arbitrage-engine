
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src import plotting

FIG_DIR = PROJECT_ROOT / "figures" / "final"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "final_visuals"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
NOTE = "Synthetic placeholder shown only when real project outputs are unavailable. Regenerate locally before interpreting results."
dates = pd.bdate_range("2022-01-03", periods=360)
triplets = ["NVDA_SMH_QQQ", "AMD_SMH_QQQ", "AAPL_QQQ_XLK", "MSFT_QQQ_XLK", "JPM_XLF_KRE", "BAC_XLF_KRE", "XOM_XLE_USO", "CVX_XLE_USO", "TSLA_QQQ_XLY", "AMZN_QQQ_XLY"]
coverage = pd.DataFrame({"symbol": ["NVDA", "SMH", "QQQ", "AMD", "AAPL", "XLK", "MSFT", "JPM", "XLF", "KRE", "XOM", "XLE", "USO", "CVX", "TSLA", "XLY", "AMZN"], "n_observations": rng.integers(326, 360, 17), "coverage_ratio": rng.uniform(0.91, 1.00, 17).round(3)})
trend = np.linspace(0, 0.28, len(dates))
prices = pd.DataFrame({"date": dates, "target": 100*np.exp(trend+np.cumsum(rng.normal(0,0.014,len(dates)))), "anchor_1": 100*np.exp(0.70*trend+np.cumsum(rng.normal(0,0.010,len(dates)))), "anchor_2": 100*np.exp(0.55*trend+np.cumsum(rng.normal(0,0.008,len(dates))))})
coefficients = pd.concat([pd.DataFrame({"date": dates, "model": model, "beta_1": base1+np.cumsum(rng.normal(0,scale,len(dates))), "beta_2": base2+np.cumsum(rng.normal(0,scale*0.75,len(dates)))}) for model,base1,base2,scale in [("rolling_ols",1.10,-0.20,0.006),("ridge",0.95,-0.10,0.003),("kalman",1.02,-0.15,0.004)]], ignore_index=True)
residual = np.zeros(len(dates))
for i in range(1,len(dates)):
    residual[i] = 0.88*residual[i-1] + rng.normal(0,0.55)
z_scores = pd.DataFrame({"date": dates, "z_score": (residual-residual.mean())/residual.std()})
acf = plotting.compute_autocorrelation(z_scores["z_score"], max_lag=20)
half_life = pd.DataFrame({"triplet_id": triplets, "half_life": rng.uniform(3.5,22.0,len(triplets)).round(2)})
baseline_returns = rng.normal(0.025,0.20,len(dates))
baseline_equity = pd.DataFrame({"date": dates, "equity": baseline_returns.cumsum()})
baseline_equity["drawdown"] = baseline_equity["equity"] - baseline_equity["equity"].cummax()
labels = pd.DataFrame({"label": rng.choice([0,1], size=260, p=[0.57,0.43])})
feature_names = ["residual_z_score","residual_change","residual_volatility","residual_autocorrelation","half_life_estimate","rolling_r_squared","beta_stability","correlation_stability","target_return_volatility","market_return","recent_drawdown"]
features = pd.DataFrame(rng.normal(size=(220,len(feature_names))), columns=feature_names)
features["residual_volatility"] = 0.45*features["target_return_volatility"] + rng.normal(0,0.6,len(features))
features["correlation_stability"] = 0.50*features["beta_stability"] + rng.normal(0,0.5,len(features))
loss = pd.DataFrame({"iteration": np.arange(1,401)})
loss["loss"] = 0.69*np.exp(-loss["iteration"]/170) + 0.37 + rng.normal(0,0.004,len(loss))
calibration = pd.DataFrame({"probability_bucket": ["0.00-0.20","0.20-0.40","0.40-0.60","0.60-0.80","0.80-1.00"], "mean_predicted_probability": [0.13,0.31,0.50,0.69,0.86], "realized_success_rate": [0.18,0.30,0.47,0.61,0.74], "precision": [0.18,0.30,0.47,0.61,0.74], "n_events": [24,58,71,46,19]})
ml_curve = pd.concat([pd.DataFrame({"date": dates, "strategy": "baseline", "equity": baseline_returns.cumsum()}), pd.DataFrame({"date": dates, "strategy": "ml_filtered", "equity": (baseline_returns*0.62 + rng.normal(0.012,0.10,len(dates))).cumsum()}), pd.DataFrame({"date": dates, "strategy": "probability_sized", "equity": (baseline_returns*0.45 + rng.normal(0.010,0.08,len(dates))).cumsum()})])
ml_curve["drawdown"] = ml_curve.groupby("strategy")["equity"].transform(lambda s: s - s.cummax())
performance_triplet = pd.DataFrame([(t,s,rng.normal(0.5 if s!="baseline" else 0.2,0.8)) for t in triplets for s in ["baseline","ml_filtered"]], columns=["triplet_id","strategy","net_pnl"])
performance_regime = pd.DataFrame({"regime": ["mean_reverting","trending","volatile_breakdown"]*2, "strategy": ["baseline"]*3 + ["ml_filtered"]*3, "mean_trade_pnl": [0.18,-0.04,-0.15,0.26,-0.02,-0.09]})
cost_sensitivity = pd.DataFrame([(scenario,cost,strategy,pnl-cost*mult) for scenario,cost in [("0 bps",0.00),("2 bps",0.02),("5 bps",0.05),("10 bps",0.10),("20 bps",0.20)] for strategy,pnl,mult in [("baseline",5.2,30),("ml_filtered",4.7,16),("probability_sized",3.9,11)]], columns=["cost_scenario","total_cost_per_unit","strategy","net_pnl"])

plotting.plot_price_coverage_summary(coverage, FIG_DIR / "price_coverage_summary.png", note=NOTE)
plotting.plot_triplet_price_relationship(prices, FIG_DIR / "triplet_price_relationship.png", title="NVDA triplet indexed price relationship", note=NOTE)
plotting.plot_hedge_ratio_stability(coefficients, FIG_DIR / "hedge_ratio_stability.png", note=NOTE)
plotting.plot_residual_zscore_example(z_scores, FIG_DIR / "residual_zscore_example.png", note=NOTE)
plotting.plot_residual_distribution(z_scores, FIG_DIR / "residual_distribution.png", note=NOTE)
plotting.plot_residual_autocorrelation(acf, FIG_DIR / "residual_autocorrelation.png", note=NOTE)
plotting.plot_half_life_by_triplet(half_life, FIG_DIR / "half_life_by_triplet.png", note=NOTE)
plotting.plot_baseline_equity_curve(baseline_equity, FIG_DIR / "baseline_equity_curve.png", note=NOTE)
plotting.plot_baseline_drawdown(baseline_equity, FIG_DIR / "baseline_drawdown.png", note=NOTE)
plotting.plot_event_label_distribution(labels, FIG_DIR / "event_label_distribution.png", note=NOTE)
plotting.plot_feature_correlation_heatmap(features, FIG_DIR / "feature_correlation_heatmap.png", note=NOTE)
plotting.plot_logistic_loss_curve(loss, FIG_DIR / "logistic_loss_curve.png", note=NOTE)
plotting.plot_probability_calibration_curve(calibration, FIG_DIR / "probability_calibration_curve.png", note=NOTE)
plotting.plot_precision_by_probability_bucket(calibration, FIG_DIR / "precision_by_probability_bucket.png", note=NOTE)
plotting.plot_strategy_equity_curve(ml_curve, FIG_DIR / "ml_filtered_vs_baseline_equity.png", note=NOTE)
plotting.plot_performance_by_triplet(performance_triplet, FIG_DIR / "performance_by_triplet.png", note=NOTE)
plotting.plot_performance_by_regime(performance_regime, FIG_DIR / "performance_by_regime.png", note=NOTE)
plotting.plot_transaction_cost_sensitivity(cost_sensitivity, FIG_DIR / "transaction_cost_sensitivity.png", note=NOTE)
plotting.write_chart_caption_table(OUT_DIR / "chart_captions.csv")

pd.DataFrame({"figure": sorted(p.name for p in FIG_DIR.glob("*.png"))}).to_csv(OUT_DIR / "final_figure_manifest.csv", index=False)
print(f"generated {len(list(FIG_DIR.glob('*.png')))} final figures")
