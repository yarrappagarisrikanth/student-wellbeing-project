"""Streamlit dashboard for the educational project."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT/"data"; MODELS = ROOT/"models"
st.set_page_config(page_title="Student Wellbeing AI", layout="wide")
st.title("Student Burnout and Social Media Distraction Risk")
st.warning("Educational demonstration only. Synthetic data and predictions are not medical or psychological diagnoses.")
processed = DATA/"processed_student_wellbeing.csv"
if not processed.exists():
    st.error("Processed data is missing. Run: python src/generate_datasets.py && python src/analyze_data.py")
    st.stop()
df = pd.read_csv(processed)
with st.sidebar:
    page = st.radio("Page", ["Overview", "Analysis", "Individual prediction"])
if page == "Overview":
    a,b,c,d = st.columns(4); a.metric("Students", len(df)); b.metric("Avg sleep", f"{df.sleep_hours.mean():.1f} h"); c.metric("Avg social media", f"{df.total_social_media_minutes_mean.mean():.0f} min"); d.metric("High burnout", int((df.burnout_risk=="High").sum()))
    st.bar_chart(df["burnout_risk"].value_counts()); st.bar_chart(df["distraction_risk"].value_counts())
elif page == "Analysis":
    st.subheader("Social-media and burnout relationship")
    st.scatter_chart(df, x="total_social_media_minutes_mean", y="burnout_score", color="burnout_risk")
    st.subheader("Average app usage")
    app_cols = [c for c in df.columns if c.startswith("average_") and c.endswith("_minutes_mean")]
    if app_cols: st.bar_chart(df[app_cols].mean().sort_values())
    st.dataframe(df[["student_id", "burnout_risk", "distraction_risk", "overall_risk", "sleep_hours", "stress_level"]].head(100), use_container_width=True)
else:
    st.subheader("Predict risk for a hypothetical student")
    models = {}
    missing = []
    for target in ["burnout_risk", "distraction_risk", "overall_risk"]:
        path = MODELS/f"{target}.joblib"
        if path.exists(): models[target] = joblib.load(path)
        else: missing.append(target)
    if missing:
        st.error("Models missing. Run: python src/train_models.py")
        st.stop()
    c1,c2,c3 = st.columns(3)
    with c1: sleep = st.slider("Sleep hours", 3.0, 10.0, 6.5); study = st.slider("Study hours", .5, 12.0, 4.0); screen = st.slider("Screen time hours", 1.0, 14.0, 6.0); stress = st.slider("Stress level", 1, 10, 6)
    with c2: assignments = st.slider("Assignments/week", 1, 14, 5); pending = st.slider("Pending assignments", 0, 10, 2); exams = st.slider("Exams next 30 days", 0, 8, 2); deadline = st.slider("Deadline pressure", 1, 10, 5)
    with c3: social = st.slider("Social-media minutes/day", 0, 900, 240); pickups = st.slider("Phone pickups/day", 5, 300, 70); interruptions = st.slider("Study interruptions/day", 0, 30, 5); late = st.slider("Late-night usage minutes", 0, 300, 30)
    row = df.drop(columns=["burnout_risk", "distraction_risk", "overall_risk", "burnout_score", "distraction_score", "overall_score", "student_id"]).median(numeric_only=True).to_frame().T
    for col, val in {"sleep_hours":sleep,"study_hours":study,"screen_time_hours":screen,"stress_level":stress,"assignments_per_week":assignments,"pending_assignments":pending,"exams_next_30_days":exams,"deadline_pressure":deadline,"total_social_media_minutes_mean":social,"phone_pickups_mean":pickups,"study_interruptions_mean":interruptions,"late_night_usage_minutes_mean":late}.items():
        if col in row: row[col] = val
    for target, model in models.items(): st.metric(target.replace("_", " ").title(), model.predict(row)[0])
    st.info("General suggestions: schedule focused study blocks, reduce non-essential notifications, protect sleep, and contact a trusted support person if stress feels difficult to manage.")
