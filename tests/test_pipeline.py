from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_generated_files_exist():
    assert (ROOT / "src/generate_datasets.py").exists()
    assert (ROOT / "src/analyze_data.py").exists()
    assert (ROOT / "src/train_models.py").exists()

def test_processed_schema_when_available():
    path = ROOT / "data/processed_student_wellbeing.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    assert len(df) > 0
    assert {"student_id", "burnout_risk", "distraction_risk", "overall_risk"}.issubset(df.columns)
    assert df.student_id.is_unique
