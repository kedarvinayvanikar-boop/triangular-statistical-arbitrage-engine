from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATABASE_DIR = DATA_DIR / "database"
FIGURES_DIR = PROJECT_ROOT / "figures"
SQL_DIR = PROJECT_ROOT / "sql"

DEFAULT_DATABASE_PATH = DATABASE_DIR / "triangular_stat_arb.db"
DEFAULT_ROLLING_WINDOW = 60
DEFAULT_RIDGE_ALPHA = 1.0

# Each triplet is (target stock, hedge ETF 1, hedge ETF 2). The two hedge
# legs are meant to be economically related to the target -- typically a
# broad or Nasdaq proxy plus a sector-specific ETF, or two sector-adjacent
# ETFs -- so the residual has a defensible interpretation as "target rich or
# cheap relative to its own sector/market backdrop," not an arbitrary pair.
# `theme` groups triplets that share both hedge legs; see
# src/portfolio.py:shared_leg_groups for why that grouping matters when
# aggregating results across the universe.
#
# These are real, currently-listed tickers as of this repository's
# knowledge cutoff. A few of the smaller thematic ETFs (e.g. CARZ, HACK,
# IPAY) are less liquid and have historically been more prone to closures,
# mergers, or ticker changes than the large sector SPDRs -- verify they
# still trade before running ingestion against them.
TRIPLET_DEFINITIONS = [
    # -- semiconductors (SMH, QQQ) --
    {"triplet_id": "NVDA_SMH_QQQ", "target": "NVDA", "hedge_1": "SMH", "hedge_2": "QQQ", "theme": "semiconductors"},
    {"triplet_id": "AMD_SMH_QQQ", "target": "AMD", "hedge_1": "SMH", "hedge_2": "QQQ", "theme": "semiconductors"},
    {"triplet_id": "INTC_SMH_QQQ", "target": "INTC", "hedge_1": "SMH", "hedge_2": "QQQ", "theme": "semiconductors"},
    {"triplet_id": "QCOM_SMH_QQQ", "target": "QCOM", "hedge_1": "SMH", "hedge_2": "QQQ", "theme": "semiconductors"},
    {"triplet_id": "TXN_SMH_QQQ", "target": "TXN", "hedge_1": "SMH", "hedge_2": "QQQ", "theme": "semiconductors"},
    {"triplet_id": "MU_SMH_QQQ", "target": "MU", "hedge_1": "SMH", "hedge_2": "QQQ", "theme": "semiconductors"},
    {"triplet_id": "AVGO_SMH_QQQ", "target": "AVGO", "hedge_1": "SMH", "hedge_2": "QQQ", "theme": "semiconductors"},
    {"triplet_id": "LRCX_SMH_QQQ", "target": "LRCX", "hedge_1": "SMH", "hedge_2": "QQQ", "theme": "semiconductors"},

    # -- big tech / software (QQQ, XLK) --
    {"triplet_id": "AAPL_QQQ_XLK", "target": "AAPL", "hedge_1": "QQQ", "hedge_2": "XLK", "theme": "big_tech"},
    {"triplet_id": "MSFT_QQQ_XLK", "target": "MSFT", "hedge_1": "QQQ", "hedge_2": "XLK", "theme": "big_tech"},
    {"triplet_id": "GOOGL_QQQ_XLK", "target": "GOOGL", "hedge_1": "QQQ", "hedge_2": "XLK", "theme": "big_tech"},
    {"triplet_id": "META_QQQ_XLK", "target": "META", "hedge_1": "QQQ", "hedge_2": "XLK", "theme": "big_tech"},
    {"triplet_id": "ORCL_QQQ_XLK", "target": "ORCL", "hedge_1": "QQQ", "hedge_2": "XLK", "theme": "big_tech"},
    {"triplet_id": "ADBE_QQQ_XLK", "target": "ADBE", "hedge_1": "QQQ", "hedge_2": "XLK", "theme": "big_tech"},

    # -- consumer discretionary (QQQ, XLY) --
    {"triplet_id": "TSLA_QQQ_XLY", "target": "TSLA", "hedge_1": "QQQ", "hedge_2": "XLY", "theme": "consumer_discretionary"},
    {"triplet_id": "AMZN_QQQ_XLY", "target": "AMZN", "hedge_1": "QQQ", "hedge_2": "XLY", "theme": "consumer_discretionary"},
    {"triplet_id": "HD_QQQ_XLY", "target": "HD", "hedge_1": "QQQ", "hedge_2": "XLY", "theme": "consumer_discretionary"},
    {"triplet_id": "NKE_QQQ_XLY", "target": "NKE", "hedge_1": "QQQ", "hedge_2": "XLY", "theme": "consumer_discretionary"},
    {"triplet_id": "SBUX_QQQ_XLY", "target": "SBUX", "hedge_1": "QQQ", "hedge_2": "XLY", "theme": "consumer_discretionary"},
    {"triplet_id": "LOW_QQQ_XLY", "target": "LOW", "hedge_1": "QQQ", "hedge_2": "XLY", "theme": "consumer_discretionary"},
    {"triplet_id": "BKNG_QQQ_XLY", "target": "BKNG", "hedge_1": "QQQ", "hedge_2": "XLY", "theme": "consumer_discretionary"},

    # -- big banks (XLF, KRE) --
    {"triplet_id": "JPM_XLF_KRE", "target": "JPM", "hedge_1": "XLF", "hedge_2": "KRE", "theme": "banks"},
    {"triplet_id": "BAC_XLF_KRE", "target": "BAC", "hedge_1": "XLF", "hedge_2": "KRE", "theme": "banks"},
    {"triplet_id": "WFC_XLF_KRE", "target": "WFC", "hedge_1": "XLF", "hedge_2": "KRE", "theme": "banks"},
    {"triplet_id": "C_XLF_KRE", "target": "C", "hedge_1": "XLF", "hedge_2": "KRE", "theme": "banks"},
    {"triplet_id": "GS_XLF_KRE", "target": "GS", "hedge_1": "XLF", "hedge_2": "KRE", "theme": "banks"},
    {"triplet_id": "MS_XLF_KRE", "target": "MS", "hedge_1": "XLF", "hedge_2": "KRE", "theme": "banks"},
    {"triplet_id": "USB_XLF_KRE", "target": "USB", "hedge_1": "XLF", "hedge_2": "KRE", "theme": "banks"},

    # -- energy majors (XLE, USO) --
    {"triplet_id": "XOM_XLE_USO", "target": "XOM", "hedge_1": "XLE", "hedge_2": "USO", "theme": "energy"},
    {"triplet_id": "CVX_XLE_USO", "target": "CVX", "hedge_1": "XLE", "hedge_2": "USO", "theme": "energy"},
    {"triplet_id": "COP_XLE_USO", "target": "COP", "hedge_1": "XLE", "hedge_2": "USO", "theme": "energy"},
    {"triplet_id": "SLB_XLE_USO", "target": "SLB", "hedge_1": "XLE", "hedge_2": "USO", "theme": "energy"},
    {"triplet_id": "EOG_XLE_USO", "target": "EOG", "hedge_1": "XLE", "hedge_2": "USO", "theme": "energy"},
    {"triplet_id": "OXY_XLE_USO", "target": "OXY", "hedge_1": "XLE", "hedge_2": "USO", "theme": "energy"},
    {"triplet_id": "MPC_XLE_USO", "target": "MPC", "hedge_1": "XLE", "hedge_2": "USO", "theme": "energy"},

    # -- pharma / general healthcare (XLV, IYH) --
    {"triplet_id": "JNJ_XLV_IYH", "target": "JNJ", "hedge_1": "XLV", "hedge_2": "IYH", "theme": "healthcare"},
    {"triplet_id": "PFE_XLV_IYH", "target": "PFE", "hedge_1": "XLV", "hedge_2": "IYH", "theme": "healthcare"},
    {"triplet_id": "UNH_XLV_IYH", "target": "UNH", "hedge_1": "XLV", "hedge_2": "IYH", "theme": "healthcare"},
    {"triplet_id": "MRK_XLV_IYH", "target": "MRK", "hedge_1": "XLV", "hedge_2": "IYH", "theme": "healthcare"},
    {"triplet_id": "ABBV_XLV_IYH", "target": "ABBV", "hedge_1": "XLV", "hedge_2": "IYH", "theme": "healthcare"},
    {"triplet_id": "LLY_XLV_IYH", "target": "LLY", "hedge_1": "XLV", "hedge_2": "IYH", "theme": "healthcare"},

    # -- biotech (XLV, IBB) --
    {"triplet_id": "GILD_XLV_IBB", "target": "GILD", "hedge_1": "XLV", "hedge_2": "IBB", "theme": "biotech"},
    {"triplet_id": "REGN_XLV_IBB", "target": "REGN", "hedge_1": "XLV", "hedge_2": "IBB", "theme": "biotech"},
    {"triplet_id": "VRTX_XLV_IBB", "target": "VRTX", "hedge_1": "XLV", "hedge_2": "IBB", "theme": "biotech"},
    {"triplet_id": "AMGN_XLV_IBB", "target": "AMGN", "hedge_1": "XLV", "hedge_2": "IBB", "theme": "biotech"},

    # -- industrials (XLI, DIA) --
    {"triplet_id": "CAT_XLI_DIA", "target": "CAT", "hedge_1": "XLI", "hedge_2": "DIA", "theme": "industrials"},
    {"triplet_id": "BA_XLI_DIA", "target": "BA", "hedge_1": "XLI", "hedge_2": "DIA", "theme": "industrials"},
    {"triplet_id": "HON_XLI_DIA", "target": "HON", "hedge_1": "XLI", "hedge_2": "DIA", "theme": "industrials"},
    {"triplet_id": "GE_XLI_DIA", "target": "GE", "hedge_1": "XLI", "hedge_2": "DIA", "theme": "industrials"},
    {"triplet_id": "UNP_XLI_DIA", "target": "UNP", "hedge_1": "XLI", "hedge_2": "DIA", "theme": "industrials"},

    # -- aerospace / defense (XLI, ITA) --
    {"triplet_id": "LMT_XLI_ITA", "target": "LMT", "hedge_1": "XLI", "hedge_2": "ITA", "theme": "aerospace_defense"},
    {"triplet_id": "RTX_XLI_ITA", "target": "RTX", "hedge_1": "XLI", "hedge_2": "ITA", "theme": "aerospace_defense"},
    {"triplet_id": "NOC_XLI_ITA", "target": "NOC", "hedge_1": "XLI", "hedge_2": "ITA", "theme": "aerospace_defense"},

    # -- utilities (XLU, SPY) --
    {"triplet_id": "NEE_XLU_SPY", "target": "NEE", "hedge_1": "XLU", "hedge_2": "SPY", "theme": "utilities"},
    {"triplet_id": "DUK_XLU_SPY", "target": "DUK", "hedge_1": "XLU", "hedge_2": "SPY", "theme": "utilities"},
    {"triplet_id": "SO_XLU_SPY", "target": "SO", "hedge_1": "XLU", "hedge_2": "SPY", "theme": "utilities"},

    # -- REITs (XLRE, SPY) --
    {"triplet_id": "PLD_XLRE_SPY", "target": "PLD", "hedge_1": "XLRE", "hedge_2": "SPY", "theme": "reits"},
    {"triplet_id": "AMT_XLRE_SPY", "target": "AMT", "hedge_1": "XLRE", "hedge_2": "SPY", "theme": "reits"},
    {"triplet_id": "SPG_XLRE_SPY", "target": "SPG", "hedge_1": "XLRE", "hedge_2": "SPY", "theme": "reits"},

    # -- communication services (XLC, QQQ) --
    {"triplet_id": "DIS_XLC_QQQ", "target": "DIS", "hedge_1": "XLC", "hedge_2": "QQQ", "theme": "communications"},
    {"triplet_id": "NFLX_XLC_QQQ", "target": "NFLX", "hedge_1": "XLC", "hedge_2": "QQQ", "theme": "communications"},
    {"triplet_id": "CMCSA_XLC_QQQ", "target": "CMCSA", "hedge_1": "XLC", "hedge_2": "QQQ", "theme": "communications"},
    {"triplet_id": "VZ_XLC_QQQ", "target": "VZ", "hedge_1": "XLC", "hedge_2": "QQQ", "theme": "communications"},

    # -- materials / metals miners (XLB, GDX) --
    {"triplet_id": "NEM_XLB_GDX", "target": "NEM", "hedge_1": "XLB", "hedge_2": "GDX", "theme": "materials_miners"},
    {"triplet_id": "GOLD_XLB_GDX", "target": "GOLD", "hedge_1": "XLB", "hedge_2": "GDX", "theme": "materials_miners"},
    {"triplet_id": "FCX_XLB_GDX", "target": "FCX", "hedge_1": "XLB", "hedge_2": "GDX", "theme": "materials_miners"},

    # -- autos (XLY, CARZ) --
    {"triplet_id": "F_XLY_CARZ", "target": "F", "hedge_1": "XLY", "hedge_2": "CARZ", "theme": "autos"},
    {"triplet_id": "GM_XLY_CARZ", "target": "GM", "hedge_1": "XLY", "hedge_2": "CARZ", "theme": "autos"},

    # -- airlines (XLI, JETS) --
    {"triplet_id": "DAL_XLI_JETS", "target": "DAL", "hedge_1": "XLI", "hedge_2": "JETS", "theme": "airlines"},
    {"triplet_id": "UAL_XLI_JETS", "target": "UAL", "hedge_1": "XLI", "hedge_2": "JETS", "theme": "airlines"},
    {"triplet_id": "LUV_XLI_JETS", "target": "LUV", "hedge_1": "XLI", "hedge_2": "JETS", "theme": "airlines"},

    # -- cybersecurity / cloud software (QQQ, HACK) --
    {"triplet_id": "CRWD_QQQ_HACK", "target": "CRWD", "hedge_1": "QQQ", "hedge_2": "HACK", "theme": "cyber_cloud"},
    {"triplet_id": "PANW_QQQ_HACK", "target": "PANW", "hedge_1": "QQQ", "hedge_2": "HACK", "theme": "cyber_cloud"},
    {"triplet_id": "NOW_QQQ_HACK", "target": "NOW", "hedge_1": "QQQ", "hedge_2": "HACK", "theme": "cyber_cloud"},
    {"triplet_id": "INTU_QQQ_HACK", "target": "INTU", "hedge_1": "QQQ", "hedge_2": "HACK", "theme": "cyber_cloud"},

    # -- payments / fintech (XLF, IPAY) --
    {"triplet_id": "V_XLF_IPAY", "target": "V", "hedge_1": "XLF", "hedge_2": "IPAY", "theme": "payments"},
    {"triplet_id": "MA_XLF_IPAY", "target": "MA", "hedge_1": "XLF", "hedge_2": "IPAY", "theme": "payments"},
    {"triplet_id": "PYPL_XLF_IPAY", "target": "PYPL", "hedge_1": "XLF", "hedge_2": "IPAY", "theme": "payments"},

    # -- consumer staples (XLP, SPY) --
    {"triplet_id": "PG_XLP_SPY", "target": "PG", "hedge_1": "XLP", "hedge_2": "SPY", "theme": "staples"},
    {"triplet_id": "KO_XLP_SPY", "target": "KO", "hedge_1": "XLP", "hedge_2": "SPY", "theme": "staples"},
    {"triplet_id": "PEP_XLP_SPY", "target": "PEP", "hedge_1": "XLP", "hedge_2": "SPY", "theme": "staples"},
    {"triplet_id": "WMT_XLP_SPY", "target": "WMT", "hedge_1": "XLP", "hedge_2": "SPY", "theme": "staples"},
]
