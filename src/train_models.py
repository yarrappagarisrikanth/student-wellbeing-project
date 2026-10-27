"""Train leakage-aware classifiers for the three risk targets."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data/processed_student_wellbeing.csv"; MODELS = ROOT / "models"; FIGURES = ROOT / "outputs/figures"
MODELS.mkdir(exist_ok=True, parents=True); FIGURES.mkdir(exist_ok=True, parents=True)
df = pd.read_csv(DATA)
targets = ["burnout_risk", "distraction_risk", "overall_risk"]
leak = targets + ["burnout_score", "distraction_score", "overall_score"]
features = [c for c in df.columns if c not in leak + ["student_id"]]
X = df[features]
cat = X.select_dtypes(include="object").columns.tolist(); num = [c for c in X.columns if c not in cat]
def make_preprocessor(sparse=True):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)
    return ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num), ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), cat)])
models = {
    "logistic": (LogisticRegression(max_iter=1500, class_weight="balanced"), True),
    "random_forest": (RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", min_samples_leaf=2), False),
    "hist_gradient_boosting": (HistGradientBoostingClassifier(max_iter=180, random_state=42), False),
}
all_metrics = {}
for target in targets:
    y = df[target]; Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    results = []
    for name, (estimator, sparse) in models.items():
        # HistGradientBoosting requires dense input; Random Forest also works reliably with dense input.
        pipe = Pipeline([("preprocessor", make_preprocessor(sparse=False)), ("model", estimator)])
        pipe.fit(Xtr, ytr); pred = pipe.predict(Xte)
        results.append((f1_score(yte, pred, average="macro"), name, pipe, pred))
    score, name, best, pred = max(results, key=lambda x: x[0])
    joblib.dump(best, MODELS / f"{target}.joblib")
    report = classification_report(yte, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=["Low", "Medium", "High"])
    plt.figure(figsize=(5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"]); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{target} confusion matrix"); plt.tight_layout(); plt.savefig(FIGURES / f"confusion_matrix_{target}.png", dpi=160); plt.close()
    all_metrics[target] = {"selected_model": name, "accuracy": accuracy_score(yte, pred), "macro_f1": score, "report": report, "features": features}
    print(f"{target}: {name}, accuracy={all_metrics[target]['accuracy']:.3f}, macro_f1={score:.3f}")
with open(MODELS / "metrics.json", "w", encoding="utf-8") as f: json.dump(all_metrics, f, indent=2, default=float)
print("Saved models, metrics, and confusion matrices.")
