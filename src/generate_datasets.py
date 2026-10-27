"""Generate deterministic synthetic datasets for the student wellbeing project."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)
N = 600
DAYS = 30
IDS = [f"STU_{i:04d}" for i in range(1, N + 1)]

def integer(x, lo, hi):
    return np.clip(np.rint(x), lo, hi).astype(int)

def risk(score):
    return np.where(score < 35, "Low", np.where(score < 65, "Medium", "High"))

lifestyle = pd.DataFrame({
    "student_id": IDS,
    "age": RNG.integers(18, 25, N),
    "academic_year": RNG.choice(["First Year", "Second Year", "Third Year", "Fourth Year"], N),
    "course_stream": RNG.choice(["Computer Science", "Information Technology", "Electronics", "Mechanical", "Commerce", "Science", "Arts"], N),
    "sleep_hours": np.round(np.clip(RNG.normal(6.8, 1.1, N), 3.5, 9.5), 1),
    "study_hours": np.round(np.clip(RNG.normal(4.5, 1.7, N), .5, 10), 1),
    "screen_time_hours": np.round(np.clip(RNG.normal(5.8, 2, N), 1, 13), 1),
    "physical_activity_hours": np.round(np.clip(RNG.normal(.8, .6, N), 0, 4), 1),
    "attendance_percentage": integer(RNG.normal(78, 12, N), 40, 100),
    "average_score": integer(RNG.normal(69, 14, N), 25, 100),
    "social_support_score": integer(RNG.normal(6.5, 2, N), 1, 10),
    "stress_level": integer(RNG.normal(5.5, 2.2, N), 1, 10),
})
lifestyle.to_csv(DATA / "student_lifestyle.csv", index=False)

rows = []
for i, sid in enumerate(IDS):
    for date in pd.date_range("2025-01-01", periods=DAYS):
        base = lifestyle.loc[i, "screen_time_hours"] * 60 * (1.15 if date.dayofweek >= 5 else .95)
        apps = np.maximum(0, RNG.normal([.22, .25, .16, .14, .05, .12], [12, 15, 12, 10, 7, 15]) * base)
        entertainment = apps.sum()
        sessions = int(np.clip(round(RNG.normal(4.5, 1.5)), 1, 10))
        completed = int(RNG.binomial(sessions, np.clip(.95 - entertainment / 1000, .25, .95)))
        interruptions = int(np.clip(round(RNG.normal(entertainment * .025, 2)), 0, 30))
        rows.append([sid, date.date(), *np.round(apps, 1), int(np.clip(round(RNG.normal(45 + entertainment*.08, 18)), 5, 250)), int(np.clip(round(RNG.normal(30 + entertainment*.12, 25)), 5, 300)), interruptions, round(max(0, RNG.normal(entertainment*.11, 10)), 1), sessions, completed, int(np.clip(round(entertainment/90 + interruptions*.25 + RNG.normal(0, .8)), 1, 10))])
cols = ["student_id", "date", "instagram_minutes", "youtube_minutes", "tiktok_minutes", "whatsapp_minutes", "facebook_minutes", "gaming_minutes", "notifications_count", "phone_pickups", "study_interruptions", "late_night_usage_minutes", "study_session_count", "completed_study_sessions", "self_reported_distraction"]
social = pd.DataFrame(rows, columns=cols)
social.to_csv(DATA / "social_media_usage.csv", index=False)

s = social.assign(total_social_minutes=social[cols[2:8]].sum(axis=1)).groupby("student_id").mean(numeric_only=True).reset_index()
academic = lifestyle[["student_id", "sleep_hours", "physical_activity_hours", "stress_level"]].merge(s, on="student_id")
academic["assignments_per_week"] = integer(RNG.normal(5.5, 2.2, N), 1, 14)
academic["pending_assignments"] = integer(RNG.normal(2.5, 1.5, N), 0, 10)
academic["exams_next_30_days"] = integer(RNG.normal(2.5, 1.5, N), 0, 8)
academic["deadline_pressure"] = integer(academic.assignments_per_week*.55 + academic.pending_assignments*.6 + academic.exams_next_30_days*.35 + RNG.normal(0, 1, N), 1, 10)
academic["motivation_level"] = integer(7 - academic.stress_level*.22 - academic.self_reported_distraction*.12 + RNG.normal(0, 1, N), 1, 10)
academic["exhaustion_level"] = integer(academic.stress_level*.55 + academic.deadline_pressure*.25 + academic.late_night_usage_minutes/80 - academic.sleep_hours*.35 + RNG.normal(0, 1, N), 1, 10)
academic["concentration_level"] = integer(8 - academic.self_reported_distraction*.3 - academic.study_interruptions*.15 + academic.sleep_hours*.15 + RNG.normal(0, .8, N), 1, 10)
academic["distraction_score"] = np.clip(academic.self_reported_distraction*5 + academic.study_interruptions*1.5 + academic.phone_pickups/30 + academic.late_night_usage_minutes/20 - academic.completed_study_sessions*2, 0, 100)
academic["burnout_score"] = np.clip((7.5-academic.sleep_hours).clip(lower=0)*7 + academic.deadline_pressure*4 + academic.pending_assignments*2 + academic.exams_next_30_days*1.5 + academic.stress_level*4 + academic.exhaustion_level*4 + (10-academic.concentration_level)*3 + (10-academic.motivation_level)*2 + academic.distraction_score*.2, 0, 100)
academic["overall_score"] = academic.burnout_score*.65 + academic.distraction_score*.35
academic["burnout_risk"] = risk(academic.burnout_score)
academic["distraction_risk"] = risk(academic.distraction_score)
academic["overall_risk"] = risk(academic.overall_score)
keep = ["student_id", "assignments_per_week", "pending_assignments", "exams_next_30_days", "deadline_pressure", "motivation_level", "exhaustion_level", "concentration_level", "burnout_score", "burnout_risk", "distraction_score", "distraction_risk", "overall_score", "overall_risk"]
academic[keep].to_csv(DATA / "academic_burnout.csv", index=False)
print(f"Generated {N} students, {len(social)} daily usage records, and {N} academic records.")
