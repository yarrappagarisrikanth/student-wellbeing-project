"""Generate deterministic synthetic datasets for the student wellbeing project."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
N_STUDENTS = 600
DAYS = 30
STUDENT_IDS = [f"STU_{i:04d}" for i in range(1, N_STUDENTS + 1)]


def bounded_int(values, low, high):
    return np.clip(np.rint(values), low, high).astype(int)


def risk_label(score):
    return np.select([score < 35, score < 65], ["Low", "Medium"], default="High")


lifestyle = pd.DataFrame({
    "student_id": STUDENT_IDS,
    "age": RNG.integers(18, 25, N_STUDENTS),
    "academic_year": RNG.choice(["First Year", "Second Year", "Third Year", "Fourth Year"], N_STUDENTS),
    "course_stream": RNG.choice(["Computer Science", "Information Technology", "Electronics", "Mechanical", "Commerce", "Science", "Arts"], N_STUDENTS),
    "sleep_hours": np.round(np.clip(RNG.normal(6.8, 1.1, N_STUDENTS), 3.5, 9.5), 1),
    "study_hours": np.round(np.clip(RNG.normal(4.5, 1.7, N_STUDENTS), 0.5, 10), 1),
    "screen_time_hours": np.round(np.clip(RNG.normal(5.8, 2.0, N_STUDENTS), 1, 13), 1),
    "physical_activity_hours": np.round(np.clip(RNG.normal(0.8, 0.6, N_STUDENTS), 0, 4), 1),
    "attendance_percentage": bounded_int(RNG.normal(78, 12, N_STUDENTS), 40, 100),
    "average_score": bounded_int(RNG.normal(69, 14, N_STUDENTS), 25, 100),
    "social_support_score": bounded_int(RNG.normal(6.5, 2, N_STUDENTS), 1, 10),
    "stress_level": bounded_int(RNG.normal(5.5, 2.2, N_STUDENTS), 1, 10),
})
lifestyle.to_csv(DATA / "student_lifestyle.csv", index=False)

app_names = ["instagram", "youtube", "tiktok", "whatsapp", "facebook", "gaming"]
app_shares = np.array([0.22, 0.25, 0.16, 0.14, 0.05, 0.12])
app_stds = np.array([12, 15, 12, 10, 7, 15])
usage_rows = []
for i, student_id in enumerate(STUDENT_IDS):
    for date in pd.date_range("2025-01-01", periods=DAYS, freq="D"):
        total_minutes = lifestyle.loc[i, "screen_time_hours"] * 60 * (1.15 if date.dayofweek >= 5 else 0.95)
        app_values = np.maximum(0, RNG.normal(total_minutes * app_shares, app_stds))
        entertainment = float(app_values.sum())
        sessions = int(np.clip(np.rint(RNG.normal(4.5, 1.5)), 1, 10))
        completion_probability = np.clip(0.95 - entertainment / 1000, 0.25, 0.95)
        completed = int(RNG.binomial(sessions, completion_probability))
        interruptions = int(np.clip(np.rint(RNG.normal(entertainment * 0.025, 2)), 0, 30))
        usage_rows.append([
            student_id, date.date(), *np.round(app_values, 1),
            int(np.clip(np.rint(RNG.normal(45 + entertainment * 0.08, 18)), 5, 250)),
            int(np.clip(np.rint(RNG.normal(30 + entertainment * 0.12, 25)), 5, 300)),
            interruptions, round(max(0, RNG.normal(entertainment * 0.11, 10)), 1),
            sessions, completed,
            int(np.clip(np.rint(entertainment / 90 + interruptions * 0.25 + RNG.normal(0, 0.8)), 1, 10)),
        ])
usage_columns = ["student_id", "date"] + [f"{name}_minutes" for name in app_names] + [
    "notifications_count", "phone_pickups", "study_interruptions", "late_night_usage_minutes",
    "study_session_count", "completed_study_sessions", "self_reported_distraction",
]
social = pd.DataFrame(usage_rows, columns=usage_columns)
social.to_csv(DATA / "social_media_usage.csv", index=False)

app_minutes = social[[f"{name}_minutes" for name in app_names]].sum(axis=1)
summary = social.assign(total_social_minutes=app_minutes).groupby("student_id").mean(numeric_only=True).reset_index()
academic = lifestyle[["student_id", "sleep_hours", "physical_activity_hours", "stress_level"]].merge(summary, on="student_id")
academic["assignments_per_week"] = bounded_int(RNG.normal(5.5, 2.2, N_STUDENTS), 1, 14)
academic["pending_assignments"] = bounded_int(RNG.normal(2.5, 1.5, N_STUDENTS), 0, 10)
academic["exams_next_30_days"] = bounded_int(RNG.normal(2.5, 1.5, N_STUDENTS), 0, 8)
academic["deadline_pressure"] = bounded_int(academic.assignments_per_week * 0.55 + academic.pending_assignments * 0.6 + academic.exams_next_30_days * 0.35 + RNG.normal(0, 1, N_STUDENTS), 1, 10)
academic["motivation_level"] = bounded_int(7 - academic.stress_level * 0.22 - academic.self_reported_distraction * 0.12 + RNG.normal(0, 1, N_STUDENTS), 1, 10)
academic["exhaustion_level"] = bounded_int(academic.stress_level * 0.55 + academic.deadline_pressure * 0.25 + academic.late_night_usage_minutes / 80 - academic.sleep_hours * 0.35 + RNG.normal(0, 1, N_STUDENTS), 1, 10)
academic["concentration_level"] = bounded_int(8 - academic.self_reported_distraction * 0.3 - academic.study_interruptions * 0.15 + academic.sleep_hours * 0.15 + RNG.normal(0, 0.8, N_STUDENTS), 1, 10)

distraction_raw = (
    academic.self_reported_distraction / 10 * 0.35
    + np.clip(academic.total_social_minutes / 500, 0, 1) * 0.25
    + np.clip(academic.phone_pickups / 180, 0, 1) * 0.15
    + np.clip(academic.study_interruptions / 15, 0, 1) * 0.15
    + np.clip(academic.late_night_usage_minutes / 120, 0, 1) * 0.10
)
academic["distraction_score"] = np.round(np.clip(distraction_raw * 100, 0, 100), 2)
workload = np.clip((academic.assignments_per_week + academic.pending_assignments * 1.5 + academic.exams_next_30_days) / 25, 0, 1)
sleep_risk = np.clip((8 - academic.sleep_hours) / 5, 0, 1)
stress = academic.stress_level / 10
exhaustion = academic.exhaustion_level / 10
concentration_risk = (10 - academic.concentration_level) / 9
motivation_risk = (10 - academic.motivation_level) / 9
academic["burnout_score"] = np.round(np.clip(100 * (
    sleep_risk * 0.18 + workload * 0.18 + stress * 0.18 + exhaustion * 0.18
    + concentration_risk * 0.10 + motivation_risk * 0.08 + academic.distraction_score / 100 * 0.10
), 0, 100), 2)
academic["overall_score"] = np.round(academic.burnout_score * 0.65 + academic.distraction_score * 0.35, 2)
academic["burnout_risk"] = risk_label(academic.burnout_score)
academic["distraction_risk"] = risk_label(academic.distraction_score)
academic["overall_risk"] = risk_label(academic.overall_score)
keep = ["student_id", "assignments_per_week", "pending_assignments", "exams_next_30_days", "deadline_pressure", "motivation_level", "exhaustion_level", "concentration_level", "burnout_score", "burnout_risk", "distraction_score", "distraction_risk", "overall_score", "overall_risk"]
academic[keep].to_csv(DATA / "academic_burnout.csv", index=False)
print(f"Generated {N_STUDENTS} students, {len(social)} daily usage records, and {N_STUDENTS} academic records.")
print("Burnout labels:", academic["burnout_risk"].value_counts().to_dict())
print("Distraction labels:", academic["distraction_risk"].value_counts().to_dict())
