from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_source_files_exist():
    for path in ["src/generate_datasets.py", "src/analyze_data.py", "src/train_models.py", "src/explain_predictions.py", "app/streamlit_app.py"]:
        assert (ROOT / path).exists()

def test_generated_schema_when_available():
    lifestyle = ROOT / "data/student_lifestyle.csv"
    social = ROOT / "data/social_media_usage.csv"
    academic = ROOT / "data/academic_burnout.csv"
    if not all(path.exists() for path in [lifestyle, social, academic]):
        return
    life, usage, acad = map(pd.read_csv, [lifestyle, social, academic])
    assert len(life) == len(acad) > 0
    assert len(usage) == len(life) * 30
    assert life.student_id.is_unique and acad.student_id.is_unique
    assert set(life.student_id) == set(acad.student_id) == set(usage.student_id)
    assert {"burnout_risk", "distraction_risk", "overall_risk"}.issubset(acad.columns)

def test_processed_schema_when_available():
    path = ROOT / "data/processed_student_wellbeing.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    assert len(df) > 0 and df.student_id.is_unique
    assert {"burnout_risk", "distraction_risk", "overall_risk", "digital_distraction_index"}.issubset(df.columns)
