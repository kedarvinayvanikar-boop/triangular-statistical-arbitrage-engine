"""
Rebuilds dashboard/index.html from the current contents of data/processed/.

The dashboard is a static snapshot, not a live view -- it does not read the
CSVs at open time. Running this script is the only way to make it reflect
new pipeline output; re-running the notebooks or scripts/generate_final_visuals.py
alone has no effect on dashboard/index.html until this is run afterward.

The output is a single self-contained HTML file: dashboard/template.html
holds the markup/CSS/JS with three placeholders, and this script substitutes
the aggregated data and the two vendored libraries (Chart.js, Three.js +
OrbitControls) so the result has no external dependency and no server
requirement -- it opens directly in a browser, online or offline.

Run from the repository root: python scripts/build_dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
TEMPLATE_PATH = DASHBOARD_DIR / "template.html"
OUTPUT_PATH = DASHBOARD_DIR / "index.html"
VENDOR_DIR = DASHBOARD_DIR / "vendor"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def build_payload() -> dict:
    payload: dict = {}

    eq = _read("ml_backtest_equity_curve.csv")
    payload["equity_curves"] = {
        strat: {
            "dates": g["date"].tolist(),
            "equity": [round(v, 3) for v in g["equity"]],
            "drawdown": [round(v, 3) for v in g["drawdown"]],
        }
        for strat, g in eq.groupby("strategy")
    }

    cmp_ = _read("baseline_vs_ml_pnl.csv")
    payload["strategy_comparison"] = cmp_.round(3).to_dict(orient="records")

    regime = _read("hmm_strategy_performance_by_regime.csv")
    payload["performance_by_regime"] = regime.round(3).to_dict(orient="records")

    cost = _read("cost_adjusted_performance_summary.csv")
    payload["cost_sensitivity"] = cost.round(4).to_dict(orient="records")

    robust = _read("robustness_summary.csv")
    payload["robustness"] = robust.round(3).to_dict(orient="records")

    succ = _read("event_success_rate_by_triplet.csv")
    payload["triplet_success"] = succ.round(3).to_dict(orient="records")

    bct = _read("backtest_comparison_table.csv")
    payload["triplet_diagnostics"] = (
        bct[["triplet_id", "method", "residual_std", "residual_abs_mean", "autocorr_1",
             "std_ratio_vs_static", "data_source", "interpretation_status"]]
        .round(4)
        .to_dict(orient="records")
    )

    calib = _read("probability_calibration_curve.csv")
    payload["calibration"] = calib.round(4).to_dict(orient="records")

    roc = _read("roc_curve.csv")
    roc = roc[np.isfinite(roc["threshold"])]
    payload["roc"] = roc.round(4).to_dict(orient="records")

    fc = _read("feature_correlation_matrix.csv").rename(columns={"Unnamed: 0": "feature"})
    payload["feature_correlation"] = {
        "features": fc["feature"].tolist(),
        "matrix": fc.drop(columns=["feature"]).round(3).values.tolist(),
    }

    thresh = _read("threshold_sensitivity_table.csv")
    payload["threshold_sensitivity"] = thresh.round(3).to_dict(orient="records")

    hmm = _read("hmm_regime_probability_table.csv").sort_values(["triplet_id", "date"])
    surface = {}
    for trip, g in hmm.groupby("triplet_id"):
        g = g.reset_index(drop=True)
        surface[trip] = {
            "t": list(range(len(g))),
            "z": [round(v, 4) for v in g["feature_value"]],
            "p_mean_revert": [round(v, 4) for v in g["mean_reverting_probability"]],
            "p_breakdown": [round(v, 4) for v in g["volatile_breakdown_probability"]],
            "p_trending": [round(v, 4) for v in g["trending_probability"]],
            "regime": g["most_likely_regime"].tolist(),
        }
    payload["regime_surface"] = surface

    return payload


def main() -> None:
    for required in (TEMPLATE_PATH, VENDOR_DIR / "chart.umd.min.js", VENDOR_DIR / "three-bundle.min.js"):
        if not required.exists():
            raise FileNotFoundError(
                f"missing {required} -- dashboard/vendor/ and dashboard/template.html "
                "must be present to rebuild the dashboard"
            )

    payload = build_payload()
    data_json = json.dumps(payload, separators=(",", ":"))

    template = TEMPLATE_PATH.read_text()
    chart_js = (VENDOR_DIR / "chart.umd.min.js").read_text()
    three_js = (VENDOR_DIR / "three-bundle.min.js").read_text()

    out = (
        template
        .replace("/*__CHART_JS__*/", chart_js)
        .replace("/*__THREE_JS__*/", three_js)
        .replace("/*__DATA__*/", data_json)
    )
    OUTPUT_PATH.write_text(out)
    print(f"wrote {OUTPUT_PATH} ({len(out):,} bytes)")


if __name__ == "__main__":
    main()
