"""dashboard.py
--------------
Streamlit dashboard for the BIA405 mini-project: monthly urban residential
electricity consumption forecasting for Bhutan's dzongkhags.

Unlike a pipeline-based app, this dashboard reads the artefacts that
``run_analysis.py`` already wrote to ``outputs/`` and ``data/cleaned/``.
Nothing is refit on page load except one small backtest for the selected
dzongkhag, so it opens instantly.

Run with:
    streamlit run dashboard.py

Requires (pip install): streamlit pandas numpy matplotlib statsmodels
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# Chart text follows the UI font: Poppins when installed, falling back to Segoe UI.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Poppins", "Segoe UI", "DejaVu Sans"]

# Must be the very first Streamlit command in the script (before the cached
# loader or anything else that touches the page).
st.set_page_config(page_title="Bhutan Electricity Forecast — BIA405", layout="wide")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
RESULTS = OUT / "model_results"
FORECASTS = OUT / "forecasts"
CLEANED_CSV = ROOT / "data" / "cleaned" / "electricity_monthly_cleaned.csv"
QUALITY_JSON = ROOT / "data" / "cleaned" / "data_quality_summary.json"
TARGET = "consumption_urban_residents_GWh"

MODELS = ["Seasonal Naive", "Holt-Winters", "SARIMA"]

REQUIRED_FILES = [
    RESULTS / "test_2025_model_metrics.csv",
    RESULTS / "best_model_by_dzongkhag.csv",
    RESULTS / "residual_diagnostics.csv",
    RESULTS / "expanding_window_metrics.csv",
    RESULTS / "national_direct_2025_model_metrics.csv",
    RESULTS / "bottom_up_vs_direct_national_2025.csv",
    FORECASTS / "dzongkhag_2026_forecasts.csv",
    FORECASTS / "national_2026_bottom_up_forecast.csv",
    FORECASTS / "national_2026_direct_forecast.csv",
    OUT / "eda" / "data_driven_seasonal_groups.csv",
    OUT / "eda" / "monthly_seasonal_profiles.csv",
    CLEANED_CSV,
    QUALITY_JSON,
]


# ==============================================================================
# Palette — fixed categorical slots from the validated reference palette
# (see the dataviz method: hues are assigned by entity, never cycled).
#   slot 1 blue = Seasonal Naive / bottom-up · slot 2 orange = Holt-Winters /
#   direct · slot 3 aqua = SARIMA.  Observed data is always ink, never a hue.
# The first three slots validate all-pairs in both modes, so three model series
# are safe on any chart form this dashboard uses.
# ==============================================================================
def is_dark_mode() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # st.context.theme unavailable in some runtimes
        return False


def palette(dark: bool) -> dict:
    if dark:
        return {
            "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",
            "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
            "grid": "#2c2c2a", "baseline": "#383835",
            "good": "#0ca30c", "warn": "#fab219", "serious": "#ec835a",
            "seq": ["#0d366b", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4", "#cde2fb"],
        }
    return {
        "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "baseline": "#c3c2b7",
        "good": "#0ca30c", "warn": "#fab219", "serious": "#ec835a",
        "seq": ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"],
    }


MODEL_COLOR = {"Seasonal Naive": "s1", "Holt-Winters": "s2", "SARIMA": "s3"}


def accents(dark: bool) -> dict:
    """Faded card tints — one plain background + soft border + accent dot each.
    Card text always stays in ink; the accent is decoration only."""
    if dark:
        return {
            "blue":   ("#2a78d61f", "#2a78d645", "#5598e7"),
            "aqua":   ("#1baf7a1f", "#1baf7a45", "#2fc492"),
            "orange": ("#eb68341f", "#eb683445", "#f0855a"),
            "violet": ("#7a5fd01f", "#7a5fd045", "#9d87e0"),
            "amber":  ("#d9a0211f", "#d9a02145", "#e6b64f"),
        }
    return {
        "blue":   ("#eef4fc", "#d8e6f7", "#2a78d6"),
        "aqua":   ("#e9f7f1", "#d2ede1", "#1baf7a"),
        "orange": ("#fdf0ea", "#f7dccb", "#eb6834"),
        "violet": ("#f1effb", "#e0dbf3", "#7a5fd0"),
        "amber":  ("#fbf5e6", "#f0e3c4", "#c9930f"),
    }


# ==============================================================================
# Data loading
# ==============================================================================
@st.cache_data(show_spinner="Loading pipeline outputs...")
def load_all() -> dict:
    cleaned = pd.read_csv(CLEANED_CSV, parse_dates=["date"])
    cleaned["dzongkhag"] = cleaned["dzongkhag"].str.strip().str.upper()
    d = {
        "cleaned": cleaned,
        "metrics": pd.read_csv(RESULTS / "test_2025_model_metrics.csv"),
        "best": pd.read_csv(RESULTS / "best_model_by_dzongkhag.csv"),
        "diag": pd.read_csv(RESULTS / "residual_diagnostics.csv"),
        "exp": pd.read_csv(RESULTS / "expanding_window_metrics.csv"),
        "direct_metrics": pd.read_csv(RESULTS / "national_direct_2025_model_metrics.csv"),
        "comparison": pd.read_csv(RESULTS / "bottom_up_vs_direct_national_2025.csv"),
        "fc": pd.read_csv(FORECASTS / "dzongkhag_2026_forecasts.csv", parse_dates=["Month"]),
        "bu": pd.read_csv(FORECASTS / "national_2026_bottom_up_forecast.csv", parse_dates=["Month"]),
        "direct": pd.read_csv(FORECASTS / "national_2026_direct_forecast.csv", parse_dates=["Month"]),
        "groups": pd.read_csv(OUT / "eda" / "data_driven_seasonal_groups.csv"),
        "profiles": pd.read_csv(OUT / "eda" / "monthly_seasonal_profiles.csv"),
        "quality": json.loads(QUALITY_JSON.read_text(encoding="utf-8")),
    }
    return d


missing = [str(p.relative_to(ROOT)) for p in REQUIRED_FILES if not p.exists()]
if missing:
    st.error("Dashboard inputs are missing. Run the analysis first (`python run_analysis.py`), "
             "then refresh this page.\n\nMissing: " + ", ".join(f"`{m}`" for m in missing))
    st.stop()

D = load_all()
CLEANED: pd.DataFrame = D["cleaned"]
DZONGKHAGS = sorted(CLEANED["dzongkhag"].unique())


# ==============================================================================
# Chart helpers — recessive chrome, ink follows the theme, thin marks
# ==============================================================================
def style_ax(ax, C, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["baseline"])
    ax.tick_params(colors=C["muted"], labelsize=9)
    ax.grid(axis=grid_axis, color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def new_ax(w, h, C, grid_axis="y"):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    style_ax(ax, C, grid_axis)
    return fig, ax


def finish(fig, ax, C, title=None, xlabel=None, ylabel=None, legend=False, date_axis=False):
    if title:
        ax.set_title(title, color=C["ink"], fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=C["ink2"], fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["ink2"], fontsize=9.5)
    if date_axis:
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.tick_params(axis="x", colors=C["muted"])
    if legend:
        leg = ax.legend(frameon=False, fontsize=9, loc="best")
        for t in leg.get_texts():
            t.set_color(C["ink2"])
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def mape_tone(v, C) -> str:
    """Status colour for a test-MAPE cell (value is always printed alongside,
    so colour never carries the number alone)."""
    if pd.isna(v):
        return ""
    if v <= 10:
        c = C["good"]
    elif v <= 20:
        c = C["warn"]
    else:
        c = C["serious"]
    return f"background-color:{c}26"


def flag_tone(v, C) -> str:
    return f"color:{C['serious']}; font-weight:600" if v else f"color:{C['ink2']}"


def title_case(name: str) -> str:
    return name.title()


# ==============================================================================
# Live backtest for the selected dzongkhag (same specifications as run_analysis.py)
# ==============================================================================
@st.cache_data(show_spinner=False)
def backtest_2025(dz: str, data_sig: str) -> dict:
    s = (CLEANED[CLEANED["dzongkhag"] == dz]
         .set_index("date")[TARGET].asfreq("MS"))
    train, test = s.loc[:"2024-12-01"], s.loc["2025-01-01":]

    def snaive(t, h):
        return pd.Series(np.resize(np.asarray(t)[-12:], h), index=test.index)

    def hw(t, h):
        fits = []
        for trend, seasonal in [("add", "add"), (None, "add"), ("add", "mul"), (None, "mul")]:
            try:
                f = ExponentialSmoothing(t, trend=trend, seasonal=seasonal, seasonal_periods=12,
                                         initialization_method="estimated").fit(optimized=True)
                fits.append((f.aic, f))
            except Exception:
                pass
        f = min(fits, key=lambda x: x[0])[1]
        return pd.Series(np.asarray(f.forecast(h)), index=test.index)

    def sarima(t, h):
        fits = []
        for p, q, P, Q in [(0, 1, 0, 1), (1, 1, 0, 1)]:
            try:
                f = SARIMAX(t, order=(p, 1, q), seasonal_order=(P, 1, Q, 12), trend="c",
                            enforce_stationarity=False, enforce_invertibility=False
                            ).fit(disp=False, maxiter=40)
                fits.append((f.aic, f))
            except Exception:
                pass
        f = min(fits, key=lambda x: x[0])[1]
        return pd.Series(np.asarray(f.forecast(h)), index=test.index)

    return {"actual": test,
            "preds": {"Seasonal Naive": snaive(train, 12),
                      "Holt-Winters": hw(train, 12),
                      "SARIMA": sarima(train, 12)}}


# ==============================================================================
# Page setup
# ==============================================================================
DARK = is_dark_mode()
C = palette(DARK)
A = accents(DARK)

INK = "#111827" if not DARK else "#f4f5f7"          # primary text on the page surface
INK2 = "#4b5563" if not DARK else "#b9bcc5"         # secondary text
MUTED = "#9aa0ab" if not DARK else "#7e828d"        # captions
SURFACE = "#f6f8fb" if not DARK else "#1b1d23"      # page background
CARD_BG = "#ffffff" if not DARK else "#23262e"      # plain card surface
LINE = "#e5e8ee" if not DARK else "#343843"         # hairlines

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Poppins', 'Segoe UI', sans-serif;
}}
.stApp {{ background: {SURFACE}; }}
h1, h2, h3, .stMarkdown h1 {{ font-family: 'Poppins', 'Segoe UI', sans-serif; color: {INK}; }}
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}

/* Section titles — small, tracked, quiet */
.app-h3 {{ font-size: 0.95rem; font-weight: 600; color: {INK};
           margin: 0.1rem 0 0.35rem 0; letter-spacing: 0.01em; }}

/* ---------- Uniform KPI cards ---------- */
.kpi-row {{ display: flex; gap: 14px; margin-bottom: 4px; }}
.kpi-card {{
    flex: 1; min-width: 0;
    background: {CARD_BG};
    border: 1px solid {LINE};
    border-top: 3px solid var(--kpi-accent, {C['s1']});
    border-radius: 14px;
    padding: 16px 18px 14px 18px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}}
.kpi-card .kpi-label {{
    display: flex; align-items: center; gap: 8px;
    font-size: 0.78rem; font-weight: 500; color: {INK2};
    line-height: 1.3; margin-bottom: 6px;
    white-space: normal; word-break: normal; overflow-wrap: anywhere;
}}
.kpi-card .kpi-dot {{
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--kpi-accent, {C['s1']}); flex: 0 0 9px;
}}
.kpi-card .kpi-value {{
    font-size: clamp(1.05rem, 1.7vw, 1.45rem);
    font-weight: 700; color: {INK}; line-height: 1.15;
    letter-spacing: -0.01em; white-space: nowrap;
}}
.kpi-card .kpi-value-sm {{  /* long text values shrink to fit */
    font-size: clamp(0.92rem, 1.15vw, 1.12rem); white-space: normal;
    overflow-wrap: anywhere; line-height: 1.25;
}}
.kpi-card .kpi-sub {{
    font-size: 0.72rem; font-weight: 400; color: {MUTED};
    margin-top: 5px; line-height: 1.35; overflow-wrap: anywhere;
}}

/* ---------- Tinted info cards (faded backgrounds) ---------- */
.info-card {{
    border-radius: 14px; padding: 14px 16px;
    background: var(--card-tint); border: 1px solid var(--card-line);
    color: {INK2}; font-size: 0.86rem; line-height: 1.55;
}}
.info-card b, .info-card strong {{ color: {INK}; }}
.info-card .info-title {{
    font-weight: 600; color: {INK}; margin-bottom: 6px; font-size: 0.92rem;
}}
.info-card ul {{ margin: 0 0 0 1.1rem; padding: 0; }}
.info-card li {{ margin-bottom: 4px; }}

div[data-testid="stMetric"] {{
    background: {CARD_BG}; border: 1px solid {LINE}; border-radius: 14px;
    padding: 16px 18px 12px 18px; box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}}
[data-testid="stMetricLabel"] p {{ font-size: 0.78rem; color: {INK2}; font-weight: 500; }}
[data-testid="stMetricValue"] {{ color: {INK}; font-weight: 700; }}
div[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Poppins', 'Segoe UI', sans-serif;
    font-size: 0.9rem; font-weight: 500; color: {INK2};
    border-radius: 10px 10px 0 0; padding: 8px 14px;
}}
.stTabs [aria-selected="true"] {{ color: {INK}; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {C['s1']}; }}
.sidebar [data-testid="stSidebar"] {{ background: {CARD_BG}; }}
section[data-testid="stSidebar"] {{
    background: {CARD_BG}; border-right: 1px solid {LINE};
}}
</style>""", unsafe_allow_html=True)


def kpi_card(cols, i, label, value, sub="", accent=None, small_value=False):
    """Render one uniform KPI card into column ``cols[i]``."""
    col = cols[i]
    accent = accent or C["s1"]
    val_cls = "kpi-value kpi-value-sm" if small_value else "kpi-value"
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    col.markdown(
        f"""<div class="kpi-card" style="--kpi-accent:{accent}">
        <div class="kpi-label"><span class="kpi-dot"></span>{label}</div>
        <div class="{val_cls}">{value}</div>{sub_html}</div>""",
        unsafe_allow_html=True,
    )


def info_card(title, body_html, tint="blue"):
    bg, line, _ = A[tint]
    st.markdown(
        f"""<div class="info-card" style="--card-tint:{bg};--card-line:{line}">
        <div class="info-title">{title}</div>{body_html}</div>""",
        unsafe_allow_html=True,
    )

# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
    st.markdown("### Forecast dashboard")
    st.caption("BIA405 mini-project · urban residential electricity "
               f"{CLEANED['date'].min():%b %Y} – {CLEANED['date'].max():%b %Y}")
    st.markdown(
        f"**{len(DZONGKHAGS)} dzongkhags** · {len(CLEANED):,} monthly rows · "
        "target: `consumption_urban_residents_GWh`"
    )
    st.markdown("---")
    st.markdown("**Legend**")
    for m in MODELS:
        st.markdown(
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:3px;"
            f"background:{C[MODEL_COLOR[m]]};margin-right:7px'></span>{m}",
            unsafe_allow_html=True,
        )
    st.caption("Observed values are always drawn in ink (black/white), "
               "forecast lines take the legend colours.")
    st.markdown("---")
    with st.expander("About the data"):
        q = D["quality"]
        st.markdown(
            f"- Sarpang has a confirmed {len(q['sarpang_missing_months'])}-month gap "
            "(2022–2024); it is **never imputed** and gets a 2026 seasonal-naive "
            "forecast only.\n"
            "- Lhuentse Aug-2015 spike retained and flagged (4.5× its median month).\n"
            "- All source dates normalised to the first day of their month."
        )
    st.caption("Refresh numbers: `python run_analysis.py`, then reload. "
               "MAPE shading in tables: green ≤10% · amber 10–20% · red >20%")

# ==============================================================================
# Header
# ==============================================================================
st.title("Bhutan Urban Residential Electricity (26 Forecast)")
st.caption("Monthly consumption (GWh) · trained Jan 2015 – Dec 2024 · validated against the real "
           "Jan–Dec 2025 holdout · Seasonal Naive vs Holt-Winters vs SARIMA, best per dzongkhag.")

tab_overview, tab_detail, tab_diag, tab_data = st.tabs(
    ["National overview", "Dzongkhag detail", "Model diagnostics", "Data & methodology"])

# Shared national quantities (fixed 18-dzongkhag panel: Sarpang excluded — no
# 2022–24 records, so including it would draw an artificial dip-and-rebound).
nat = (CLEANED.query("dzongkhag != 'SARPANG'")
       .groupby("date")[TARGET].sum().asfreq("MS"))
fc19 = D["fc"]
bu18 = (fc19[fc19["Dzongkhag"] != "SARPANG"]
        .set_index("Month")[["Forecast", "lower_80", "upper_80", "lower_95", "upper_95"]]
        .groupby(level=0).sum().sort_index())
fc_tot_dz = fc19.groupby("Dzongkhag").agg(total=("Forecast", "sum"), model=("Selected_Model", "first"))
act25_dz = CLEANED[CLEANED["date"].dt.year == 2025].groupby("dzongkhag")[TARGET].sum()

mape_wide = D["metrics"].pivot(index="dzongkhag", columns="model", values="MAPE")
best_mape = pd.Series({dz: mape_wide.loc[dz, m] for dz, m in
                       zip(D["best"]["dzongkhag"], D["best"]["Best Model"].str.replace(
                           "Seasonal Naive \\(.*", "Seasonal Naive", regex=True))
                       if dz in mape_wide.index}, name="Best MAPE")

overview_tbl = (D["best"][["dzongkhag", "Best Model"]]
                .assign(Best_Model=lambda x: x["Best Model"].str.replace(
                    r"Seasonal Naive \(.*", "Seasonal Naive †", regex=True))
                .merge(best_mape, left_on="dzongkhag", right_index=True, how="left")
                .merge(fc_tot_dz, left_on="dzongkhag", right_index=True, how="left")
                .merge(act25_dz.rename("actual_2025"), left_on="dzongkhag", right_index=True, how="left")
                .merge(D["groups"][["dzongkhag", "data_driven_group"]], on="dzongkhag", how="left")
                .rename(columns={"dzongkhag": "Dzongkhag", "Best_Model": "Best model",
                                 "total": "2026 forecast (GWh)", "actual_2025": "2025 actual (GWh)",
                                 "Best MAPE": "Test MAPE (%)",
                                 "data_driven_group": "Seasonal group"}))
overview_tbl["YoY (%)"] = (overview_tbl["2026 forecast (GWh)"] - overview_tbl["2025 actual (GWh)"]) \
    / overview_tbl["2025 actual (GWh)"] * 100

# ==============================================================================
# TAB 1 — NATIONAL OVERVIEW
# ==============================================================================
with tab_overview:
    total26 = float(bu18["Forecast"].sum())
    total25 = float(nat.loc["2025-01-01":].sum())
    yoy = (total26 - total25) / total25 * 100  # shown on the national chart context
    winter_idx = bu18.index.month.isin([11, 12, 1, 2])
    winter_share = bu18.loc[winter_idx, "Forecast"].sum() / total26 * 100
    direct_best = D["direct_metrics"].loc[D["direct_metrics"]["RMSE"].idxmin()]

    kk = st.columns(5)
    kpi_card(kk, 0, "2026 national forecast", f"{total26:,.1f} GWh",
             sub="Bottom-up sum of the 18 complete dzongkhags (Sarpang excluded).",
             accent=A["blue"][2])
    kpi_card(kk, 1, "Average test MAPE", f"{best_mape.mean():.1f}%",
             sub="Mean holdout MAPE of each dzongkhag's selected model (18).",
             accent=A["aqua"][2])
    kpi_card(kk, 2, "Most common winner",
             D["best"]["Best Model"].str.replace(r" \(.*", "", regex=True).mode()[0],
             sub="Model chosen most often across dzongkhags on the 2025 holdout.",
             accent=A["orange"][2], small_value=True)
    kpi_card(kk, 3, "Winter share of 2026", f"{winter_share:.0f}%",
             sub="Nov–Feb share of the 2026 national forecast — heating-driven.",
             accent=A["violet"][2])
    kpi_card(kk, 4, "Direct national model", direct_best["model"],
             sub=f"Best single model on the aggregate series · RMSE {direct_best['RMSE']:.3f}",
             accent=A["amber"][2], small_value=True)

    st.markdown("#### Observed national consumption and the 2026 bottom-up forecast")
    fig, ax = new_ax(11.5, 4.6, C)
    ax.plot(nat.index, nat.values, color=C["ink"], linewidth=1.8, label="Observed (18 dzongkhags)")
    ax.plot(bu18.index, bu18["Forecast"], color=C["s1"], linewidth=2, label="2026 forecast (bottom-up)")
    ax.fill_between(bu18.index, bu18["lower_80"], bu18["upper_80"], color=C["s1"], alpha=0.15,
                    linewidth=0, label="80% interval")
    ax.fill_between(bu18.index, bu18["lower_95"], bu18["upper_95"], color=C["s1"], alpha=0.07,
                    linewidth=0, label="95% interval")
    ax.axvline(pd.Timestamp("2026-01-01"), color=C["baseline"], linewidth=1, linestyle=":")
    finish(fig, ax, C, ylabel="GWh", legend=True, date_axis=True)

    left, right = st.columns([2, 3])
    with left:
        st.markdown("#### 2026 forecast by dzongkhag")
        bars = fc_tot_dz["total"].sort_values()
        fig, ax = new_ax(5.2, 7.4, C, grid_axis="x")
        ax.barh(bars.index, bars.values, color=C["s1"], height=0.62)
        ax.set_yticks(range(len(bars.index)))
        ax.set_yticklabels([title_case(d) for d in bars.index], fontsize=8.5)
        ax.margins(y=0.01)
        finish(fig, ax, C, xlabel="GWh")
    with right:
        st.markdown("#### Which model wins where")
        counts = (D["best"]["Best Model"].str.replace(r" \(.*", "", regex=True).value_counts()
                  .reindex(MODELS).dropna())
        fig, ax = new_ax(7.0, 1.9, C, grid_axis="x")
        ypos = np.arange(len(counts))[::-1]
        ax.barh(ypos, counts.values, color=[C[MODEL_COLOR[m]] for m in counts.index], height=0.6)
        for y, v in zip(ypos, counts.values):
            ax.text(v + 0.15, y, f"{int(v)} dzongkhags", va="center", fontsize=9, color=C["ink2"])
        ax.set_yticks(ypos)
        ax.set_yticklabels(counts.index, fontsize=9.5)
        ax.set_xlim(0, counts.max() * 1.3)
        finish(fig, ax, C, xlabel="Dzongkhags")

        st.markdown("#### How hard is each dzongkhag to forecast?")
        hard = best_mape.sort_values()
        fig, ax = new_ax(7.0, 5.0, C, grid_axis="x")
        ax.barh(hard.index, hard.values, color=C["s1"], height=0.62)
        ax.set_yticks(range(len(hard.index)))
        ax.set_yticklabels([title_case(d) for d in hard.index], fontsize=8.5)
        ax.axvline(10, color=C["warn"], linewidth=1, linestyle="--")
        ax.axvline(20, color=C["serious"], linewidth=1, linestyle="--")
        ax.margins(y=0.01)
        finish(fig, ax, C, xlabel="Test MAPE of the selected model (%) — dashed lines: 10% / 20%")

    left2, right2 = st.columns(2)
    with left2:
        st.markdown("#### Annual totals (18 dzongkhags)")
        annual = nat.resample("YS").sum()
        fig, ax = new_ax(6.4, 3.6, C)
        ax.bar(annual.index.year, annual.values, color=C["s1"], width=0.62)
        ax.set_xticks(annual.index.year)
        ax.set_xticklabels(annual.index.year, fontsize=8.5)
        finish(fig, ax, C, ylabel="GWh")
    with right2:
        st.markdown("#### Average seasonal profile")
        prof = nat.groupby(nat.index.month).mean()
        fig, ax = new_ax(6.4, 3.6, C)
        ax.plot(prof.index, prof.values, color=C["s1"], linewidth=2, marker="o", markersize=5)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        finish(fig, ax, C, ylabel="Mean GWh / month")

    st.markdown("#### Seasonality by dzongkhag — monthly index (each dzongkhag's mean = 100)")
    pmat = D["profiles"].set_index("month").T
    pmat = pmat.div(pmat.mean(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    im = ax.imshow(pmat.values, aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", C["seq"]))
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                       fontsize=9)
    ax.set_yticks(range(len(pmat.index)))
    ax.set_yticklabels([title_case(d) for d in pmat.index], fontsize=8.5)
    ax.tick_params(colors=C["muted"])
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.tick_params(colors=C["muted"], labelsize=8.5)
    cbar.outline.set_visible(False)
    cbar.set_label("Index (mean = 100)", color=C["ink2"], fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Blues run light→dark with magnitude; values far from 100 mark each dzongkhag's "
               "high/low-consumption months. Thimphu's winter peak (≈+30% above its mean) dominates the pattern.")

    st.markdown("#### Summary table")
    styled = (overview_tbl.style
              .format({"Test MAPE (%)": "{:.2f}", "2026 forecast (GWh)": "{:.3f}",
                       "2025 actual (GWh)": "{:.3f}", "YoY (%)": "{:+.1f}"})
              .map(lambda v: mape_tone(v, C), subset=["Test MAPE (%)"])
              .hide(axis="index"))
    st.dataframe(styled, use_container_width=True)
    st.caption("† Sarpang has no 2025 holdout — its seasonal-naive forecast uses its complete 2025 cycle only. "
               "MAPE shading: green ≤10% · amber 10–20% · red >20%.")

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download summary table (CSV)", overview_tbl.to_csv(index=False).encode("utf-8"),
                       "dzongkhag_summary_2026.csv", "text/csv", use_container_width=True)
    d2.download_button("Download all 2026 forecasts (CSV)", fc19.to_csv(index=False).encode("utf-8"),
                       "dzongkhag_2026_forecasts.csv", "text/csv", use_container_width=True)
    d3.download_button("Download national 2026 forecast (CSV)",
                       D["bu"].to_csv(index=False).encode("utf-8"),
                       "national_2026_bottom_up_forecast.csv", "text/csv", use_container_width=True)

# ==============================================================================
# TAB 2 — DZONGKHAG DETAIL
# ==============================================================================
with tab_detail:
    sel = st.selectbox("Dzongkhag", DZONGKHAGS,
                       index=DZONGKHAGS.index("THIMPHU") if "THIMPHU" in DZONGKHAGS else 0,
                       format_func=title_case)
    row = overview_tbl[overview_tbl["Dzongkhag"] == sel].iloc[0]
    sel_series = (CLEANED[CLEANED["dzongkhag"] == sel].set_index("date")[TARGET].asfreq("MS"))
    sel_fc = fc19[fc19["Dzongkhag"] == sel].set_index("Month").sort_index()
    is_sarpang = sel == "SARPANG"

    mm = st.columns(4)
    kpi_card(mm, 0, "Selected model", row["Best model"],
             sub="Lowest RMSE on the 2025 holdout (Seasonal Naive wins within 5% — parsimony rule)."
                 if not is_sarpang else "No holdout test was possible (2022–24 records missing).",
             accent=A["blue"][2], small_value=True)
    kpi_card(mm, 1, "Test MAPE", "—" if is_sarpang else f"{row['Test MAPE (%)']:.2f}%",
             sub="Mean absolute percentage error on the real 2025 months.",
             accent=A["aqua"][2])
    kpi_card(mm, 2, "2026 forecast", f"{row['2026 forecast (GWh)']:.2f} GWh",
             sub="Sum of the twelve monthly 2026 values.", accent=A["violet"][2])
    kpi_card(mm, 3, "vs 2025 actual",
             "—" if pd.isna(row["YoY (%)"]) else f"{row['YoY (%)']:+.1f}%",
             sub="Change of the 2026 total against the 2025 actual.",
             accent=A["amber"][2])

    st.markdown(f"#### {title_case(sel)} — history and 2026 forecast")
    fig, ax = new_ax(11.5, 4.4, C)
    ax.plot(sel_series.index, sel_series.values, color=C["ink"], linewidth=1.8, label="Observed")
    ax.plot(sel_fc.index, sel_fc["Forecast"], color=C["s2"], linewidth=2, linestyle="--",
            label="2026 forecast")
    ax.fill_between(sel_fc.index, sel_fc["lower_80"], sel_fc["upper_80"], color=C["s2"], alpha=0.15,
                    linewidth=0, label="80% interval")
    ax.fill_between(sel_fc.index, sel_fc["lower_95"], sel_fc["upper_95"], color=C["s2"], alpha=0.07,
                    linewidth=0, label="95% interval")
    ax.axvline(pd.Timestamp("2026-01-01"), color=C["baseline"], linewidth=1, linestyle=":")
    finish(fig, ax, C, ylabel="GWh", legend=True, date_axis=True)

    left, right = st.columns(2)
    with left:
        if is_sarpang:
            info_card("Sarpang — why there is no backtest",
                      "Sarpang's 2022–2024 records are missing, so a 2025 holdout comparison would "
                      "be misleading. It is forecast with <b>Seasonal Naive</b> on its complete "
                      "2025 cycle only.", tint="amber")
        else:
            st.markdown("#### Backtest: the real 2025 months vs each model")
            bt = backtest_2025(sel, f"{len(CLEANED)}|{CLEANED['date'].max()}")
            fig, ax = new_ax(6.8, 4.2, C)
            ax.plot(bt["actual"].index, bt["actual"].values, color=C["ink"], linewidth=2.4,
                    label="Actual 2025")
            for m in MODELS:
                ax.plot(bt["preds"][m].index, bt["preds"][m].values, color=C[MODEL_COLOR[m]],
                        linewidth=2, linestyle="--", label=m)
            finish(fig, ax, C, ylabel="GWh", legend=True, date_axis=True)
    with right:
        st.markdown("#### 2026 month by month")
        tbl = sel_fc[["Forecast", "lower_80", "upper_80"]].copy()
        tbl.insert(0, "Month", [d.strftime("%b %Y") for d in tbl.index])
        st.dataframe(
            tbl.style.format({"Forecast": "{:.3f}", "lower_80": "{:.3f}", "upper_80": "{:.3f}"}).hide(axis="index"),
            use_container_width=True, height=330)
        st.download_button(f"Download {title_case(sel)} 2026 forecast (CSV)",
                           sel_fc.reset_index().to_csv(index=False).encode("utf-8"),
                           f"{sel.lower().replace(' ', '_')}_2026_forecast.csv", "text/csv")

    if not is_sarpang:
        left3, right3 = st.columns(2)
        with left3:
            st.markdown("#### Model accuracy on the 2025 holdout")
            ev = (D["metrics"][D["metrics"]["dzongkhag"] == sel]
                  .loc[:, ["model", "RMSE", "MAE", "MAPE", "MASE", "specification"]]
                  .set_index("model").reindex(MODELS))
            st.dataframe(ev.style
                         .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE": "{:.2f}", "MASE": "{:.2f}"})
                         .map(lambda v: mape_tone(v, C), subset=["MAPE"])
                         .hide(axis="index"), use_container_width=True)
            st.caption("Lower is better; MASE < 1 beats a seasonal-naive benchmark.")

            st.markdown("#### Stability across expanding-window validation")
            exp_dz = (D["exp"][D["exp"]["dzongkhag"] == sel]
                      .pivot(index="validation_year", columns="model", values="MAPE").reindex(columns=MODELS))
            fig, ax = new_ax(6.8, 3.2, C)
            x = np.arange(len(exp_dz.index))
            w = 0.26
            for i, m in enumerate(MODELS):
                ax.bar(x + (i - 1) * w, exp_dz[m].values, width=w, color=C[MODEL_COLOR[m]], label=m)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{y} holdout" for y in exp_dz.index])
            finish(fig, ax, C, ylabel="MAPE (%)", legend=True)
        with right3:
            st.markdown("#### Seasonal decomposition (additive, 12-month cycle)")
            if len(sel_series.dropna()) >= 24:
                dec = seasonal_decompose(sel_series.dropna(), model="additive", period=12)
                fig, axes = plt.subplots(3, 1, figsize=(6.8, 5.4), sharex=True)
                fig.patch.set_alpha(0.0)
                for axi, part, col, lab in zip(
                        axes, [dec.trend, dec.seasonal, dec.resid],
                        [C["s1"], C["s2"], C["muted"]], ["Trend", "Seasonal", "Residual"]):
                    axi.set_facecolor("none")
                    axi.plot(part.index, part.values, color=col, linewidth=1.8)
                    axi.set_ylabel(lab, color=C["ink2"], fontsize=9)
                    style_ax(axi, C)
                axes[2].tick_params(axis="x", colors=C["muted"])
                loc = mdates.AutoDateLocator()
                axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                info_card("Not enough data",
                          "This dzongkhag does not have enough contiguous months for a "
                          "seasonal decomposition.", tint="amber")

        with st.expander("Residual diagnostics for this dzongkhag (Ljung–Box on training residuals)"):
            dg = D["diag"][D["diag"]["dzongkhag"] == sel].set_index("model").reindex(MODELS)
            st.dataframe(dg.style
                         .format({"mean_residual": "{:.4f}", "ljung_box_pvalue": "{:.3f}"})
                         .map(lambda v: flag_tone(v, C), subset=["residual_autocorrelation_flag"])
                         .hide(axis="index"), use_container_width=True)
            st.caption("A *True* flag means leftover autocorrelation — the model didn't capture all structure.")

# ==============================================================================
# TAB 3 — MODEL DIAGNOSTICS
# ==============================================================================
with tab_diag:
    st.markdown("#### All models × all dzongkhags — 2025 holdout")
    ev_all = D["metrics"][["dzongkhag", "model", "RMSE", "MAE", "MAPE", "MASE", "specification"]]
    st.dataframe(ev_all.style
                 .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE": "{:.2f}", "MASE": "{:.2f}"})
                 .map(lambda v: mape_tone(v, C), subset=["MAPE"])
                 .hide(axis="index"), use_container_width=True, height=420)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Residual diagnostics")
        diag = D["diag"].copy()
        diag["autocorrelated"] = diag["residual_autocorrelation_flag"].map(
            {True: "Yes", False: "No", None: "—"})
        st.dataframe(diag[["dzongkhag", "model", "mean_residual", "ljung_box_pvalue", "autocorrelated"]]
                     .style.format({"mean_residual": "{:.4f}", "ljung_box_pvalue": "{:.3f}"})
                     .map(lambda v: flag_tone(v == "Yes", C), subset=["autocorrelated"])
                     .hide(axis="index"), use_container_width=True, height=420)
        st.caption("Ljung–Box p < 0.05 ⇒ leftover autocorrelation in training residuals.")
    with right:
        st.markdown("#### Expanding-window validation (mean MAPE by model and year)")
        exp_piv = (D["exp"].pivot_table(index="model", columns="validation_year", values="MAPE", aggfunc="mean")
                   .reindex(MODELS))
        st.dataframe(exp_piv.style.format("{:.2f}%").hide(axis="index"), use_container_width=True)
        st.caption("Untuned historical check: each year is forecast from everything before it, "
                   "so no information from 2025 leaks into these scores.")

        st.markdown("#### National: bottom-up vs direct")
        st.dataframe(D["comparison"].style
                     .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE": "{:.3f}", "MASE": "{:.3f}"})
                     .hide(axis="index"), use_container_width=True)
        st.caption("Bottom-up sums the 18 per-dzongkhag winners; direct models the aggregate series. "
                   "They land within ~1.4% RMSE of each other — the per-dzongkhag detail justifies the bottom-up route.")

        st.markdown("#### Direct national model — 2025 holdout metrics")
        direct_cols = [c for c in ["model", "RMSE", "MAE", "MAPE", "MASE"]
                       if c in D["direct_metrics"].columns]
        st.dataframe(D["direct_metrics"][direct_cols].style
                     .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE": "{:.2f}", "MASE": "{:.3f}"})
                     .hide(axis="index"), use_container_width=True)

    st.markdown("#### Two 2026 national views agree")
    fig, ax = new_ax(11.5, 4.0, C)
    ax.plot(D["bu"]["Month"], D["bu"]["Forecast"], color=C["s1"], linewidth=2, label="Bottom-up (18 dzongkhags)")
    ax.fill_between(D["bu"]["Month"], D["bu"]["lower_80"], D["bu"]["upper_80"], color=C["s1"],
                    alpha=0.13, linewidth=0, label="Bottom-up 80% interval")
    ax.plot(D["direct"]["Month"], D["direct"]["Forecast"], color=C["s2"], linewidth=2,
            linestyle="--", label="Direct national (SARIMA)")
    finish(fig, ax, C, ylabel="GWh", legend=True, date_axis=True)

# ==============================================================================
# TAB 4 — DATA & METHODOLOGY
# ==============================================================================
with tab_data:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Pipeline summary")
        info_card("How the analysis works", f"""
<ul>
<li><b>Target:</b> monthly urban residential consumption (GWh) per dzongkhag,
{CLEANED['date'].min():%b %Y} – {CLEANED['date'].max():%b %Y}.</li>
<li><b>Cleaning:</b> dates normalised to month-start; target coerced numeric; duplicates dropped;
Sarpang's 36-month gap (2022–24) <b>left missing, never imputed</b>; Lhuentse's Aug-2015 spike
retained and flagged ({D['quality']['lhuentse_aug_ratio_to_median']:.1f}× its median month).</li>
<li><b>Split:</b> train Jan 2015 – Dec 2024 (120 months); test = the real 2025 months,
withheld from fitting.</li>
<li><b>Models:</b> Seasonal Naive (baseline) · Holt-Winters (add/mul trend × seasonality, best AIC) ·
SARIMA with a small, interpretable order set — (0,1,1)(0,1,1)₁₂ / (1,1,1)(0,1,1)₁₂ by AIC.</li>
<li><b>Selection:</b> lowest RMSE on 2025 per dzongkhag, with a <b>5% parsimony rule</b> —
Seasonal Naive wins ties within 5%.</li>
<li><b>Final forecast:</b> the winning model is refit on the full 2015–2025 series before producing
2026 months with 80% / 95% intervals from training residuals.</li>
<li><b>National views:</b> a fixed <b>18-dzongkhag panel</b> (Sarpang excluded) plus a direct
aggregate model; the two 2026 national totals are compared explicitly.</li>
</ul>""", tint="blue")
        st.markdown("#### Data-driven seasonal groups")
        st.dataframe(D["groups"].rename(columns={
            "dzongkhag": "Dzongkhag", "mean_monthly_gwh": "Mean GWh / month",
            "peak_month": "Peak month", "seasonal_amplitude_ratio": "Amplitude ratio",
            "data_driven_group": "Group"}).style
            .format({"Mean GWh / month": "{:.3f}", "Amplitude ratio": "{:.2f}"}).hide(axis="index"),
            use_container_width=True)
    with right:
        st.markdown("#### Data-quality snapshot")
        q = D["quality"]
        info_card("Data quality", f"""
<ul>
<li>{q['rows']:,} rows × {q['columns']} columns, {len(q['dzongkhags'])} dzongkhags</li>
<li>Target missing values: <b>{q['target_missing']}</b></li>
<li>Duplicate rows in raw source: <b>{q['duplicate_rows_raw']}</b></li>
<li>Sarpang gap confirmed: <b>{q['sarpang_gap_confirmed']}</b>
({len(q['sarpang_missing_months'])} months)</li>
<li>Date handling: {q['date_normalisation']}</li>
</ul>""", tint="aqua")
        with st.expander("Full data-quality report (JSON)"):
            st.json(q)
    st.markdown("#### Cleaned data preview")
    st.dataframe(CLEANED.head(20), use_container_width=True)
    st.caption("**Adding new data:** paste new rows into `data/raw/electricity-consumption-monthly_v1.csv` "
               "(or point `SOURCE` in `run_analysis.py` at a new file) → run `python run_analysis.py` → "
               "reload this page. Every table, KPI card and chart reads the regenerated outputs and "
               "updates automatically.")
    st.caption("This dashboard reads the artefacts saved by `run_analysis.py` — regenerate them "
               "with `python run_analysis.py` and reload the page.")
