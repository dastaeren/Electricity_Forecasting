"""Recompute the 2025 bottom-up comparison from the already selected models."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analysis", HERE / "run_analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

df, _ = analysis.load_and_clean()
choices = pd.read_csv(analysis.RESULTS / "best_model_by_dzongkhag.csv").set_index("dzongkhag")["Best Model"]
predictions = []
for name, group in df.groupby("dzongkhag"):
    if name == "SARPANG":
        continue
    train = group.set_index("date")[analysis.TARGET].asfreq("MS").loc[:"2024-12-01"]
    winner = choices[name]
    if winner == "Seasonal Naive":
        pred = analysis.snaive(train, 12)
    elif winner == "Holt-Winters":
        pred, _, _ = analysis.hw_fit_forecast(train, 12)
    else:
        pred, _, _ = analysis.sarima_fit(train, 12)
    predictions.append(np.asarray(pred))

national = analysis.national_panel(df)
train_n, test_n = national.loc[:"2024-12-01"], national.loc["2025-01-01":]
direct = pd.read_csv(analysis.RESULTS / "national_direct_2025_model_metrics.csv")
direct_best = direct.loc[direct.RMSE.idxmin()]
comparison = pd.DataFrame([
    {"Approach": "Bottom-up selected dzongkhag models (18 dzongkhags)", **analysis.metrics(test_n, np.sum(predictions, axis=0), train_n)},
    {"Approach": f"Direct national ({direct_best['model']})", **direct_best[["RMSE", "MAE", "MAPE", "MASE"]].to_dict()},
])
comparison.to_csv(analysis.RESULTS / "bottom_up_vs_direct_national_2025.csv", index=False)
print(comparison.to_string(index=False))
