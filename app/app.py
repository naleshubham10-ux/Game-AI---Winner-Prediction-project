"""
Streamlit web app - PUBG Win Probability Predictor.

    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import MODEL_DIR, REPORT_DIR  # noqa: E402
from predict import predict_dataframe  # noqa: E402

st.set_page_config(page_title="PUBG Winner Prediction", page_icon="🎯", layout="wide")
st.title("🎯 PUBG Game Winner Prediction")
st.caption("PRCP-1012 · predicts winPlacePerc (0 = last place, 1 = chicken dinner)")

if not (MODEL_DIR / "best_model.joblib").exists():
    st.error("No trained model found. Run `python src/train.py` first.")
    st.stop()

meta = json.loads((MODEL_DIR / "metadata.json").read_text())
c1, c2, c3 = st.columns(3)
c1.metric("Production model", meta["best_model"])
c2.metric("Training groups", f"{meta['n_groups_train']:,}")
c3.metric("Features", meta["n_features"])
if meta.get("synthetic_data"):
    st.warning("Model was trained on the synthetic sample. Retrain with train_V2.csv for real use.")

tab1, tab2, tab3 = st.tabs(["Single player", "Batch (CSV)", "Model report"])

with tab1:
    st.subheader("Estimate one player's win probability")
    player = joblib.load(MODEL_DIR / "player_model.joblib")
    cols = st.columns(3)
    v = {}
    with cols[0]:
        v["kills"] = st.number_input("Kills", 0, 40, 2)
        v["headshotKills"] = st.number_input("Headshot kills", 0, 40, 0)
        v["damageDealt"] = st.number_input("Damage dealt", 0.0, 5000.0, 200.0)
        v["DBNOs"] = st.number_input("Knock-downs (DBNOs)", 0, 40, 1)
        v["assists"] = st.number_input("Assists", 0, 20, 0)
        v["killStreaks"] = st.number_input("Kill streaks", 0, 20, 1)
        v["longestKill"] = st.number_input("Longest kill (m)", 0.0, 1000.0, 30.0)
    with cols[1]:
        v["walkDistance"] = st.number_input("Walk distance (m)", 0.0, 10000.0, 1500.0)
        v["rideDistance"] = st.number_input("Ride distance (m)", 0.0, 20000.0, 0.0)
        v["swimDistance"] = st.number_input("Swim distance (m)", 0.0, 2000.0, 0.0)
        v["boosts"] = st.number_input("Boosts used", 0, 30, 2)
        v["heals"] = st.number_input("Heals used", 0, 40, 2)
        v["weaponsAcquired"] = st.number_input("Weapons acquired", 0, 80, 4)
        v["revives"] = st.number_input("Revives", 0, 20, 0)
    with cols[2]:
        v["killPlace"] = st.number_input("Kill place (rank in match)", 1, 100, 30)
        v["matchDuration"] = st.number_input("Match duration (s)", 100, 2400, 1600)
        v["playersJoined"] = st.number_input("Players in match", 1, 100, 95)
        v["teamKills"] = st.number_input("Team kills", 0, 10, 0)
        v["vehicleDestroys"] = st.number_input("Vehicles destroyed", 0, 10, 0)
    v["totalDistance"] = v["walkDistance"] + v["rideDistance"] + v["swimDistance"]
    v["healsAndBoosts"] = v["heals"] + v["boosts"]
    if st.button("Predict", type="primary"):
        x = pd.DataFrame([v])[player["features"]]
        p = float(min(max(player["model"].predict(x)[0], 0), 1))
        st.metric("Predicted winPlacePerc", f"{p:.3f}")
        st.progress(p)
        st.write(f"Expected to finish ahead of about **{p*100:.0f}%** of the groups in the match.")

with tab2:
    st.subheader("Predict a whole match file")
    st.write("Upload a CSV with the PUBG schema (e.g. `test_V2.csv`). Full matches give the best "
             "results because the model compares groups inside each match.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        with st.spinner("Predicting..."):
            preds = predict_dataframe(df)
        st.success(f"{len(preds):,} predictions")
        st.dataframe(preds.head(200), width="stretch")
        st.download_button("Download predictions.csv", preds.to_csv(index=False), "predictions.csv")

with tab3:
    comp = REPORT_DIR / "model_comparison.csv"
    if comp.exists():
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv(comp), width="stretch")
    fi = REPORT_DIR / "feature_importance.csv"
    if fi.exists():
        st.subheader("Top factors affecting win probability")
        imp = pd.read_csv(fi, index_col=0).head(20)
        st.bar_chart(imp)
