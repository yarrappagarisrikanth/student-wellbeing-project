"""Streamlit dashboard for the educational project."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"; MODELS = ROOT / "models"
st.set_page_config(page_title="Student Wellbeing AI", layout="wide")
st.title("Student Burnout and Social Media Distraction Risk")
st.warning("Educational demonstration only. Synthetic data and predictions are not medical or psychological diagnoses.")
processed = DATA / "processed_student_wellbeing.csv"
if not processed.exists():
    st.error("Processed data is missing. Run: python src/generate_datasets.py && python src/analyze_data.py")
    st.stop()
df = pd.read_csv(processed)
with st.sidebar:
    page = st.radio("Page", ["Overview", "Analysis", "Individual prediction"])
if page == "Overview":
    a, b, c, d = st.columns(4)
    a.metric("Students", len(df)); b.metric("Average sleep", f"{df.sleep_hours.mean():.1f} h"); c.metric("Average social media", f"{df.total_social_media_minutes_mean.mean():.0f} min"); d.metric("High burnout", int((df.burnout_risk == "High").sum()))
    st.subheader("Risk distributions")
    st.bar_chart(pd.DataFrame({"Burnout": df["burnout_risk"].value_counts(), "Distraction": df["distraction_risk"].value_counts()}).fillna(0))
elif page == "Analysis":
    st.subheader("Social-media usage and burnout")
    st.scatter_chart(df, x="total_social_media_minutes_mean", y="burnout_score", color="burnout_risk")
    st.subheader("Average app usage")
    app_cols = [c for c in df.columns if c.startswith("".join(["instagram", "_minutes_mean"]))]
    app_cols = [c for c in df.columns if c in {f"{app}_minutes_mean" for app in ["instagram", "youtube", "tiktok", "whatsapp", "facebook", "gaming"]}]
    if app_cols: st.bar_chart(df[app_cols].mean().rename(lambda x: x.replace("_minutes_mean", "")))
    st.dataframe(df[["student_id", "burnout_risk", "distraction_risk", "overall_risk", "sleep_hours", "stress_level"]].head(100), use_container_width=True)
else:
    st.subheader("Predict risk for a hypothetical student")
    model_paths = {target: MODELS / f"{target}.joblib" for target in ["burnout_risk", "distraction_risk", "overall_risk"]}
    if not all(path.exists() for path in model_paths.values()):
        st.error("Models are missing. Run: python src/train_models.py")
        st.stop()
    models = {target: joblib.load(path) for target, path in model_paths.items()}
    c1, c2, c3 = st.columns(3)
    with c1:
        sleep = st.slider("Sleep hours", 3.0, 10.0, 6.5); study = st.slider("Study hours", .5, 12.0, 4.0); screen = st.slider("Screen time hours", 1.0, 14.0, 6.0); stress = st.slider("Stress level", 1, 10, 6); physical = st.slider("Physical activity hours", 0.0, 4.0, .8); support = st.slider("Social support", 1, 10, 6)
    with c2:
        attendance = st.slider("Attendance percentage", 40, 100, 78); score = st.slider("Average score", 25, 100, 70); assignments = st.slider("Assignments/week", 1, 14, 5); pending = st.slider("Pending assignments", 0, 10, 2); exams = st.slider("Exams next 30 days", 0, 8, 2); deadline = st.slider("Deadline pressure", 1, 10, 5)
    with c3:
        social = st.slider("Social-media minutes/day", 0, 900, 240); pickups = st.slider("Phone pickups/day", 5, 300, 70); interruptions = st.slider("Study interruptions/day", 0, 30, 5); late = st.slider("Late-night usage minutes", 0, 300, 30); motivation = st.slider("Motivation", 1, 10, 6); exhaustion = st.slider("Exhaustion", 1, 10, 5); concentration = st.slider("Concentration", 1, 10, 6)
    row = df.drop(columns=["student_id", "burnout_risk", "distraction_risk", "overall_risk", "burnout_score", "distraction_score", "overall_score"], errors="ignore").median(numeric_only=True).to_frame().T
    values = {"sleep_hours": sleep, "study_hours": study, "screen_time_hours": screen, "stress_level": stress, "physical_activity_hours": physical, "social_support_score": support, "attendance_percentage": attendance, "average_score": score, "assignments_per_week": assignments, "pending_assignments": pending, "exams_next_30_days": exams, "deadline_pressure": deadline, "total_social_media_minutes_mean": social, "phone_pickups_mean": pickups, "study_interruptions_mean": interruptions, "late_night_usage_minutes_mean": late, "motivation_level": motivation, "exhaustion_level": exhaustion, "concentration_level": concentration}
    for col, value in values.items():
        if col in row.columns: row[col] = value
    # The training pipeline includes categorical columns; use their most common values.
    for col in ["academic_year", "course_stream"]:
        if col in [x for x in df.columns if x not in ["student_id", "burnout_risk", "distraction_risk", "overall_risk", "burnout_score", "distraction_score", "overall_score"]]: row[col] = df[col].mode().iat[0]
    if st.button("Predict"):
        results = {target: model.predict(row)[0] for target, model in models.items()}
        cols = st.columns(3)
        for col, (target, result) in zip(cols, results.items()): col.metric(target.replace("_", " ").title(), result)
        st.info("General suggestions: schedule focused study blocks, reduce non-essential notifications, protect sleep, and contact a trusted support person if stress feels difficult to manage.")
