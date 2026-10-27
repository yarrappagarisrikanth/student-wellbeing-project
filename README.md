# AI-Based Student Burnout and Social Media Distraction Risk Prediction System

Educational machine-learning mini-project that generates synthetic student lifestyle data, analyzes social-media distraction, predicts burnout/distraction/overall risk, and provides a Streamlit dashboard.

> **Important:** The data is synthetic and the predictions are not medical or psychological diagnoses. Use the project only for educational demonstration and early-warning experimentation.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/generate_datasets.py
python src/analyze_data.py
python src/train_models.py
streamlit run app/streamlit_app.py
```

The scripts create three connected datasets in `data/`, processed data and charts in `outputs/`, and trained models in `models/`. Dataset generation is deterministic with seed 42.

## Project structure

```text
data/                 Generated CSV files (ignored by git)
src/generate_datasets.py
src/analyze_data.py
src/train_models.py
app/streamlit_app.py
tests/test_pipeline.py
```

## Method

The project models three categorical targets: `burnout_risk`, `distraction_risk`, and `overall_risk`. Daily social-media records are aggregated by student, then merged with lifestyle and academic records. Training excludes target columns and directly calculated score columns to reduce leakage. Logistic Regression, Random Forest, and HistGradientBoosting are compared using macro F1.

## Limitations

Synthetic labels reflect transparent formulas rather than clinical assessments. Correlation is not causation. Real deployment would require validated instruments, privacy review, informed consent, fairness analysis, and qualified professional oversight.
