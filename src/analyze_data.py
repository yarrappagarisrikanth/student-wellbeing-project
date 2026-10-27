"""Merge generated data, engineer features, and create exploratory charts."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, REPORT = ROOT / "data", ROOT / "outputs/figures", ROOT / "outputs/reports"
FIG.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
life = pd.read_csv(DATA / "student_lifestyle.csv")
social = pd.read_csv(DATA / "social_media_usage.csv")
acad = pd.read_csv(DATA / "academic_burnout.csv")
apps = ["instagram_minutes", "youtube_minutes", "tiktok_minutes", "whatsapp_minutes", "facebook_minutes", "gaming_minutes"]
social["total_social_media_minutes"] = social[apps].sum(axis=1)
social["study_completion_percentage"] = social["completed_study_sessions"] / social["study_session_count"].clip(lower=1) * 100
agg = social.groupby("student_id").agg({**{a: "mean" for a in apps}, "total_social_media_minutes": ["mean", "max"], "notifications_count": "mean", "phone_pickups": "mean", "study_interruptions": "mean", "late_night_usage_minutes": "mean", "study_session_count": "mean", "completed_study_sessions": "mean", "study_completion_percentage": "mean", "self_reported_distraction": "mean"})
agg.columns = ["_".join(c).strip("_") for c in agg.columns]
agg = agg.reset_index()
df = life.merge(agg, on="student_id").merge(acad, on="student_id")
df["social_media_percentage_of_screen_time"] = df.total_social_media_minutes_mean / (df.screen_time_hours * 60) * 100
df["sleep_deficit_hours"] = (8 - df.sleep_hours).clip(lower=0)
df["academic_workload_score"] = df.assignments_per_week*2 + df.pending_assignments*3 + df.exams_next_30_days*2 + df.deadline_pressure*2
df["digital_distraction_index"] = df.total_social_media_minutes_mean*.08 + df.phone_pickups_mean*.08 + df.notifications_count_mean*.03 + df.study_interruptions_mean*2 + df.late_night_usage_minutes_mean*.1
df["study_efficiency_score"] = df.completed_study_sessions_mean / df.study_session_count_mean.clip(lower=1) * 100
df["wellbeing_score"] = (df.sleep_hours*8 + df.physical_activity_hours*5 + df.social_support_score*4 + df.motivation_level*4 + df.concentration_level*4 - df.stress_level*5 - df.exhaustion_level*5).clip(0, 100)
df.to_csv(DATA / "processed_student_wellbeing.csv", index=False)
pd.DataFrame({"metric": ["students", "avg_sleep", "avg_social_minutes", "avg_burnout", "avg_distraction"], "value": [len(df), df.sleep_hours.mean(), df.total_social_media_minutes_mean.mean(), df.burnout_score.mean(), df.distraction_score.mean()]}).to_csv(REPORT / "summary_report.csv", index=False)
sns.set_theme(style="whitegrid")
for col, name in [("burnout_risk", "burnout_risk.png"), ("distraction_risk", "distraction_risk.png"), ("overall_risk", "overall_risk.png")]:
    plt.figure(figsize=(7, 4)); sns.countplot(data=df, x=col, order=["Low", "Medium", "High"]); plt.title(col.replace("_", " ").title()); plt.tight_layout(); plt.savefig(FIG/name, dpi=160); plt.close()
plt.figure(figsize=(8, 5)); sns.scatterplot(data=df, x="total_social_media_minutes_mean", y="burnout_score", hue="burnout_risk"); plt.tight_layout(); plt.savefig(FIG/"social_media_vs_burnout.png", dpi=160); plt.close()
plt.figure(figsize=(8, 5)); sns.heatmap(df[["sleep_hours", "stress_level", "total_social_media_minutes_mean", "phone_pickups_mean", "academic_workload_score", "burnout_score", "distraction_score"]].corr(), annot=True, cmap="coolwarm", fmt=".2f"); plt.tight_layout(); plt.savefig(FIG/"correlation_heatmap.png", dpi=160); plt.close()
print(f"Saved processed data with {len(df)} rows and {len(df.columns)} columns.")
