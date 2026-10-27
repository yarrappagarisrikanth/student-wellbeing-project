"""Create feature-importance reports for saved models."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data/processed_student_wellbeing.csv"; MODELS = ROOT / "models"; REPORTS = ROOT / "outputs/reports"
REPORTS.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA)
for target in ["burnout_risk", "distraction_risk", "overall_risk"]:
    model = joblib.load(MODELS / f"{target}.joblib")
    leak = ["student_id", "burnout_risk", "distraction_risk", "overall_risk", "burnout_score", "distraction_score", "overall_score"]
    X = df[[c for c in df.columns if c not in leak]]
    y = df[target]
    result = permutation_importance(model, X, y, n_repeats=5, random_state=42, scoring="f1_macro")
    names = model.named_steps["preprocessor"].get_feature_names_out()
    report = pd.DataFrame({"feature": names, "importance_mean": result.importances_mean, "importance_std": result.importances_std}).sort_values("importance_mean", ascending=False)
    report.to_csv(REPORTS / f"feature_importance_{target}.csv", index=False)
    print(f"Saved feature importance for {target}.")
