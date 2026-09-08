"""BIA405 mini-project: reproducible monthly electricity forecasting analysis.

Run: python outputs/run_analysis.py
The source CSV path is intentionally explicit so no manually copied source data is needed.
"""
from __future__ import annotations

import itertools
import json
import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

SOURCE = Path(r"D:\SEM VII\BIA405 Predictive Analytics in Finance and Business\Mini Project\electricity-consumption-monthly_v1.csv")
SCRIPT_DIR = Path(__file__).resolve().parent
# Running from the shared outputs folder creates the project subfolder; running
# from a packaged project folder keeps all generated artefacts together.
ROOT = SCRIPT_DIR / "electricity_forecasting_project" if SCRIPT_DIR.name == "outputs" else SCRIPT_DIR
DATA = ROOT / "data"
OUT = ROOT / "outputs"
EDA = OUT / "eda"
RESULTS = OUT / "model_results"
FORECASTS = OUT / "forecasts"
FIGURES = OUT / "figures"
TARGET = "consumption_urban_residents_GWh"
SEASON = 12


def make_dirs():
    for folder in (DATA / "raw", DATA / "cleaned", EDA, RESULTS, FORECASTS, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)


def load_and_clean() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(SOURCE)
    # The source switches from month-end to month-start dates in 2025.  Converting
    # each observation to its monthly Period removes that convention difference.
    parsed = pd.to_datetime(raw["date"], format="mixed", errors="coerce")
    df = raw.copy()
    df["date_original"] = df["date"]
    df["date"] = parsed.dt.to_period("M").dt.to_timestamp()
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df["dzongkhag"] = df.dzongkhag.str.strip().str.upper()
    df = df.sort_values(["dzongkhag", "date"]).reset_index(drop=True)

    target_missing = int(df[TARGET].isna().sum())
    series_counts = df.groupby("dzongkhag").size().sort_values()
    expected = pd.date_range(df.date.min(), df.date.max(), freq="MS")
    gaps = {}
    for name, group in df.groupby("dzongkhag"):
        absent = expected.difference(group.date)
        gaps[name] = [str(x.date()) for x in absent]
    sarpang_gap = gaps.get("SARPANG", [])
    lhuentse_aug = df.loc[(df.dzongkhag == "LHUENTSE") & (df.date == "2015-08-01"), TARGET]
    lhuentse_value = float(lhuentse_aug.iloc[0]) if len(lhuentse_aug) else np.nan
    lhuentse_series = df.loc[df.dzongkhag == "LHUENTSE", TARGET]
    quality = {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "min_date": str(df.date.min().date()), "max_date": str(df.date.max().date()),
        "dzongkhags": sorted(df.dzongkhag.unique().tolist()),
        "target_missing": target_missing, "duplicate_rows_raw": int(raw.duplicated().sum()),
        "duplicate_date_dzongkhag_raw": int(raw.duplicated(["date", "dzongkhag"]).sum()),
        "observations_by_dzongkhag": {k: int(v) for k, v in series_counts.items()},
        "missing_months_by_dzongkhag": {k: v for k, v in gaps.items() if v},
        "sarpang_gap_confirmed": len(sarpang_gap) == 36,
        "sarpang_missing_months": sarpang_gap,
        "lhuentse_aug_2015_gwh": lhuentse_value,
        "lhuentse_aug_ratio_to_median": float(lhuentse_value / lhuentse_series.median()),
        "date_normalisation": "All source dates converted to first day of their calendar month; original date retained.",
    }
    df.to_csv(DATA / "cleaned" / "electricity_monthly_cleaned.csv", index=False)
    with open(DATA / "cleaned" / "data_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)
    return df, quality


def national_panel(df: pd.DataFrame) -> pd.Series:
    # Sarpang is excluded rather than imputed. Including it would make an artificial
    # 2022--24 decline and 2025 rebound. Bottom-up and direct comparisons use this
    # same fixed 18-dzongkhag coverage.
    return df.query("dzongkhag != 'SARPANG'").groupby("date")[TARGET].sum().asfreq("MS")


def save_eda(df: pd.DataFrame, national: pd.Series):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5)); national.plot(ax=ax, color="#1565c0")
    ax.set(title="National Urban Residential Electricity Consumption (consistent 18-dzongkhag panel)", ylabel="GWh", xlabel="Month")
    fig.tight_layout(); fig.savefig(EDA / "national_monthly_timeseries.png", dpi=180); plt.close(fig)
    annual = national.resample("YS").sum()
    annual.index = annual.index.year
    fig, ax = plt.subplots(figsize=(10, 5)); annual.plot.bar(ax=ax, color="#2e7d32")
    ax.set(title="Annual National Urban Residential Consumption", ylabel="GWh", xlabel="Year")
    fig.tight_layout(); fig.savefig(EDA / "national_annual_totals.png", dpi=180); plt.close(fig)
    prof = national.groupby(national.index.month).mean()
    fig, ax = plt.subplots(figsize=(10, 5)); prof.plot(marker="o", ax=ax, color="#ef6c00")
    ax.set(title="Average National Seasonal Profile", ylabel="Mean GWh", xlabel="Month")
    ax.set_xticks(range(1, 13)); fig.tight_layout(); fig.savefig(EDA / "national_seasonal_profile.png", dpi=180); plt.close(fig)
    totals = df.groupby("dzongkhag")[TARGET].sum().sort_values()
    fig, ax = plt.subplots(figsize=(10, 7)); totals.plot.barh(ax=ax, color="#6a1b9a")
    ax.set(title="Urban Residential Consumption by Dzongkhag, 2015–2025", xlabel="GWh", ylabel="Dzongkhag")
    fig.tight_layout(); fig.savefig(EDA / "dzongkhag_total_consumption.png", dpi=180); plt.close(fig)
    pivot = national.to_frame("value").assign(year=lambda x: x.index.year, month=lambda x: x.index.month).pivot(index="year", columns="month", values="value")
    fig, ax = plt.subplots(figsize=(11, 5)); im=ax.imshow(pivot, aspect="auto", cmap="YlOrRd")
    ax.set(title="National Month-by-Year Seasonality", xlabel="Month", ylabel="Year", xticks=range(12), xticklabels=range(1,13), yticks=range(len(pivot)), yticklabels=pivot.index)
    fig.colorbar(im, ax=ax, label="GWh"); fig.tight_layout(); fig.savefig(EDA / "national_month_year_heatmap.png", dpi=180); plt.close(fig)
    # Small multiples retain a legible time-series view for all dzongkhags.
    groups = list(df.groupby("dzongkhag")); fig, axes = plt.subplots(5, 4, figsize=(15, 13), sharex=True)
    for ax, (name, group) in itertools.zip_longest(axes.flat, groups, fillvalue=(None, None)):
        if name is None: ax.axis("off"); continue
        ax.plot(group.date, group[TARGET], linewidth=1); ax.set_title(name, fontsize=9)
    fig.suptitle("Monthly Urban Residential Consumption by Dzongkhag", y=1.01); fig.tight_layout(); fig.savefig(EDA / "dzongkhag_timeseries_small_multiples.png", dpi=180); plt.close(fig)
    profiles = df.pivot_table(index="month", columns="dzongkhag", values=TARGET, aggfunc="mean")
    profiles.to_csv(EDA / "monthly_seasonal_profiles.csv")


def seasonal_groups(df: pd.DataFrame) -> pd.DataFrame:
    profile = df.pivot_table(index="dzongkhag", columns="month", values=TARGET, aggfunc="mean")
    yearly_mean = profile.mean(axis=1)
    peak_month = profile.idxmax(axis=1)
    strength = (profile.max(axis=1) - profile.min(axis=1)) / yearly_mean
    def label(month, amplitude):
        if amplitude < 0.15: return "Weakly seasonal"
        if month in (11, 12, 1, 2, 3): return "Winter-peaking"
        if month in (6, 7, 8, 9): return "Summer-peaking"
        return "Other/intermediate"
    result = pd.DataFrame({"mean_monthly_gwh": yearly_mean, "peak_month": peak_month, "seasonal_amplitude_ratio": strength})
    result["data_driven_group"] = [label(m, a) for m, a in zip(peak_month, strength)]
    return result.reset_index().sort_values(["data_driven_group", "dzongkhag"])


def metrics(actual, predicted, insample):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    err = actual - predicted
    scale = np.mean(np.abs(np.asarray(insample, float)[SEASON:] - np.asarray(insample, float)[:-SEASON]))
    return {"RMSE": float(np.sqrt(np.mean(err**2))), "MAE": float(np.mean(np.abs(err))),
            "MAPE": float(np.mean(np.abs(err) / np.maximum(np.abs(actual), 1e-6)) * 100),
            "MASE": float(np.mean(np.abs(err)) / scale) if scale > 0 else np.nan}


def snaive(train, horizon):
    return np.resize(train.iloc[-SEASON:].to_numpy(), horizon)


def hw_fit_forecast(train, horizon):
    candidates = []
    for trend, seasonal in [("add", "add"), (None, "add"), ("add", "mul"), (None, "mul")]:
        try:
            fit = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=SEASON,
                                       initialization_method="estimated").fit(optimized=True)
            candidates.append((fit.aic, fit))
        except Exception: pass
    if not candidates: raise RuntimeError("No Holt-Winters specification converged")
    fit = min(candidates, key=lambda x: x[0])[1]
    return np.asarray(fit.forecast(horizon)), fit, f"trend={fit.model.trend}; seasonal={fit.model.seasonal}"


def sarima_fit(train, horizon, selected_order=None):
    if selected_order is None:
        # A deliberately small, interpretable candidate set: seasonal differencing
        # (D=1) plus one non-seasonal difference (d=1), with sparse MA/AR terms.
        # This avoids a wasteful brute-force search on only 120 training months.
        candidate_orders = [(0, 1, 0, 1), (1, 1, 0, 1)]
        fits = []
        for p, q, P, Q in candidate_orders:
            try:
                fit = SARIMAX(train, order=(p, 1, q), seasonal_order=(P, 1, Q, SEASON), trend="c",
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=40)
                fits.append((fit.aic, (p, 1, q), (P, 1, Q, SEASON), fit))
            except Exception: pass
        if not fits: raise RuntimeError("No SARIMA specification converged")
        _, order, seasonal, fit = min(fits, key=lambda x: x[0])
    else:
        order, seasonal = selected_order
        fit = SARIMAX(train, order=order, seasonal_order=seasonal, trend="c", enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False, maxiter=50)
    return np.asarray(fit.forecast(horizon)), fit, (order, seasonal)


def interval_from_residuals(point, residuals):
    sd = float(np.nanstd(np.asarray(residuals, float), ddof=1))
    return {"lower_80": np.maximum(0, point - norm.ppf(.90)*sd), "upper_80": point + norm.ppf(.90)*sd,
            "lower_95": np.maximum(0, point - norm.ppf(.975)*sd), "upper_95": point + norm.ppf(.975)*sd}


def evaluate_one(name, series):
    train, test = series.loc[:"2024-12-01"], series.loc["2025-01-01":]
    rows, diagnostics, cache = [], [], {}
    models = {"Seasonal Naive": None, "Holt-Winters": None, "SARIMA": None}
    for model in models:
        try:
            if model == "Seasonal Naive":
                pred = snaive(train, len(test)); resid = train.iloc[SEASON:] - train.shift(SEASON).iloc[SEASON:]; spec = "seasonal period=12"
            elif model == "Holt-Winters": pred, fit, spec = hw_fit_forecast(train, len(test)); resid = fit.resid
            else: pred, fit, spec = sarima_fit(train, len(test)); resid = fit.resid; cache["sarima_order"] = spec
            row = {"dzongkhag": name, "model": model, **metrics(test, pred, train), "specification": str(spec)}
            rows.append(row)
            clean_resid = pd.Series(resid).dropna()
            lb_p = acorr_ljungbox(clean_resid, lags=[min(12, max(1, len(clean_resid)//5))], return_df=True)["lb_pvalue"].iloc[0] if len(clean_resid) > 10 else np.nan
            diagnostics.append({"dzongkhag": name, "model": model, "mean_residual": float(clean_resid.mean()), "ljung_box_pvalue": float(lb_p), "residual_autocorrelation_flag": bool(lb_p < .05) if pd.notna(lb_p) else None})
        except Exception as e: rows.append({"dzongkhag": name, "model": model, "error": str(e)})
    # expanding windows only refit the model specifications above; they are not used to tune on 2025.
    exp_rows = []
    for year in (2022, 2023, 2024):
        cutoff = f"{year-1}-12-01"; val = series.loc[f"{year}-01-01":f"{year}-12-01"]; tr = series.loc[:cutoff]
        for model in models:
            try:
                if model == "Seasonal Naive": pred = snaive(tr, 12)
                elif model == "Holt-Winters": pred, _, _ = hw_fit_forecast(tr, 12)
                else: pred, _, _ = sarima_fit(tr, 12, cache.get("sarima_order"))
                exp_rows.append({"dzongkhag": name, "validation_year": year, "model": model, **metrics(val, pred, tr)})
            except Exception as e: exp_rows.append({"dzongkhag": name, "validation_year": year, "model": model, "error": str(e)})
    return rows, diagnostics, exp_rows, cache


def final_forecast(name, series, winner, sarima_order=None):
    index = pd.date_range("2026-01-01", periods=12, freq="MS")
    if winner == "Seasonal Naive":
        point = snaive(series, 12); resid = series.iloc[SEASON:] - series.shift(SEASON).iloc[SEASON:]
    elif winner == "Holt-Winters": point, fit, _ = hw_fit_forecast(series, 12); resid = fit.resid
    else: point, fit, _ = sarima_fit(series, 12, sarima_order); resid = fit.resid
    intervals = interval_from_residuals(point, resid)
    return pd.DataFrame({"Month": index, "Dzongkhag": name, "Forecast": point, **intervals, "Selected_Model": winner})


def main():
    make_dirs(); shutil.copy2(SOURCE, DATA / "raw" / SOURCE.name)
    df, quality = load_and_clean(); national = national_panel(df); save_eda(df, national)
    group_table = seasonal_groups(df); group_table.to_csv(EDA / "data_driven_seasonal_groups.csv", index=False)
    all_rows, all_diag, all_exp, forecasts, selected_test_predictions = [], [], [], [], []
    # Sarpang cannot be fairly evaluated on the 2015–24 / 2025 split because all
    # same-month 2024 reference values are absent. For 2026 it uses 2025 Seasonal Naive.
    for name, group in df.groupby("dzongkhag"):
        series = group.set_index("date")[TARGET].asfreq("MS")
        if name == "SARPANG":
            forecasts.append(final_forecast(name, series.dropna(), "Seasonal Naive")); continue
        rows, diag, exp, cache = evaluate_one(name, series)
        all_rows += rows; all_diag += diag; all_exp += exp
        valid = [r for r in rows if "RMSE" in r]
        best_rmse = min(r["RMSE"] for r in valid)
        # 5% parsimony rule: choose seasonal naive when it is within 5% of the best.
        sn = next(r for r in valid if r["model"] == "Seasonal Naive")
        winner = "Seasonal Naive" if sn["RMSE"] <= best_rmse * 1.05 else min(valid, key=lambda r: r["RMSE"])["model"]
        test_train = series.loc[:"2024-12-01"]
        if winner == "Seasonal Naive": test_pred = snaive(test_train, 12)
        elif winner == "Holt-Winters": test_pred, _, _ = hw_fit_forecast(test_train, 12)
        else: test_pred, _, _ = sarima_fit(test_train, 12, cache.get("sarima_order"))
        selected_test_predictions.append(np.asarray(test_pred))
        forecasts.append(final_forecast(name, series, winner, cache.get("sarima_order")))
    eval_df, diag_df, exp_df = pd.DataFrame(all_rows), pd.DataFrame(all_diag), pd.DataFrame(all_exp)
    eval_df.to_csv(RESULTS / "test_2025_model_metrics.csv", index=False); diag_df.to_csv(RESULTS / "residual_diagnostics.csv", index=False); exp_df.to_csv(RESULTS / "expanding_window_metrics.csv", index=False)
    summary = eval_df.pivot(index="dzongkhag", columns="model", values="RMSE").reset_index()
    summary["Best Model"] = summary.apply(lambda r: "Seasonal Naive" if r["Seasonal Naive"] <= r[["Seasonal Naive", "Holt-Winters", "SARIMA"]].min()*1.05 else r[["Seasonal Naive", "Holt-Winters", "SARIMA"]].idxmin(), axis=1)
    summary = pd.concat([summary, pd.DataFrame([{ "dzongkhag": "SARPANG", "Seasonal Naive": np.nan, "Holt-Winters": np.nan, "SARIMA": np.nan, "Best Model": "Seasonal Naive (2026 only; 2025 test unavailable)" }])])
    summary.to_csv(RESULTS / "best_model_by_dzongkhag.csv", index=False)
    forecast_df = pd.concat(forecasts, ignore_index=True); forecast_df.to_csv(FORECASTS / "dzongkhag_2026_forecasts.csv", index=False)
    national_fc = forecast_df.groupby("Month")[["Forecast", "lower_80", "upper_80", "lower_95", "upper_95"]].sum().reset_index(); national_fc.to_csv(FORECASTS / "national_2026_bottom_up_forecast.csv", index=False)
    # Direct national model selected using the same test period, then refitted.
    direct_rows, _, _, direct_cache = evaluate_one("NATIONAL_18_DZONGKHAGS", national)
    direct_eval = pd.DataFrame(direct_rows); direct_eval.to_csv(RESULTS / "national_direct_2025_model_metrics.csv", index=False)
    dvalid = [r for r in direct_rows if "RMSE" in r]; dsn=next(r for r in dvalid if r["model"]=="Seasonal Naive"); dmin=min(r["RMSE"] for r in dvalid)
    dwinner="Seasonal Naive" if dsn["RMSE"]<=dmin*1.05 else min(dvalid,key=lambda r:r["RMSE"])["model"]
    direct_fc=final_forecast("NATIONAL_DIRECT_18_DZONGKHAGS",national,dwinner,direct_cache.get("sarima_order")); direct_fc.to_csv(FORECASTS / "national_2026_direct_forecast.csv",index=False)
    # Bottom-up comparison calculated against the fixed, Sarpang-excluded national 2025 actual.
    train_n, test_n = national.loc[:"2024-12-01"], national.loc["2025-01-01":]
    bottomup_selected = np.sum(selected_test_predictions, axis=0)
    direct_best = min(dvalid,key=lambda r:r["RMSE"])
    comparison = pd.DataFrame([{ "Approach":"Bottom-up selected dzongkhag models (18 dzongkhags)", **metrics(test_n,bottomup_selected,train_n)}, {"Approach":f"Direct national ({direct_best['model']})", **{k:direct_best[k] for k in ['RMSE','MAE','MAPE','MASE']}}])
    comparison.to_csv(RESULTS / "bottom_up_vs_direct_national_2025.csv",index=False)
    readme = """# BIA405 forecasting outputs\n\nThe national panel contains a fixed set of 18 dzongkhags: Sarpang is excluded because it has no 2022–2024 records. No values were imputed. Sarpang receives a 2026 seasonal-naive forecast from its complete 2025 cycle, but has no defensible 2025 holdout score. Lhuentse August 2015 is retained and flagged in data_quality_summary.json; it is a source observation, not silently deleted.\n\n`test_2025_model_metrics.csv` is the primary comparison. `expanding_window_metrics.csv` provides untuned historical validation. The national comparison currently uses a seasonal-naive bottom-up benchmark versus the selected direct national method, so its approach labels must be retained in any report.\n"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("Completed. Results saved to", ROOT)

if __name__ == "__main__": main()
