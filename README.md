# Bhutan Urban Residential Electricity — 2026 Forecast (BIA405)

A Streamlit dashboard for the BIA405 mini-project: monthly urban residential
electricity consumption forecasting for Bhutan's 19 dzongkhags
(Seasonal Naive vs Holt-Winters vs SARIMA, best model per dzongkhag,
validated against the real 2025 holdout).

**Live dashboard:** _paste your Streamlit Community Cloud link here after deploying_

## Run locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (all files, including `data/` and `outputs/` —
   the dashboard reads them at runtime and never refits the pipeline).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** →
   select the repo, branch `main`, main file `dashboard.py` → **Deploy**.

No app configuration is needed: everything the dashboard shows is pre-computed
and committed in `outputs/` and `data/cleaned/`.

## Refreshing the numbers

The dashboard does **not** refit models on load. To update it:

```bash
python run_analysis.py     # regenerates outputs/ and data/cleaned/
git add outputs data       # commit the regenerated artefacts
git push                   # Streamlit Cloud redeploys automatically
```

## Data notes

- The national panel uses a fixed set of 18 dzongkhags: **Sarpang is excluded**
  because it has no 2022–2024 records. No values were imputed. Sarpang receives
  a 2026 seasonal-naive forecast from its complete 2025 cycle but has no
  defensible 2025 holdout score.
- **Lhuentse August 2015** is retained and flagged in
  `data/cleaned/data_quality_summary.json`; it is a source observation, not
  silently deleted.
- `outputs/model_results/test_2025_model_metrics.csv` is the primary
  comparison; `expanding_window_metrics.csv` provides untuned historical
  validation. The national comparison uses a seasonal-naive bottom-up
  benchmark versus the selected direct national method.
