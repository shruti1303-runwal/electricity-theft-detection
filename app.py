"""
Streamlit demo: upload smart-meter data, get a ranked list of suspected thieves.

Run:  streamlit run app.py
"""
import json
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, "src")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from preprocess import build_inputs  # noqa: E402


def customer_chart(series):
    """Daily kWh plus a 7-day average, with axes fixed to the data (no zoom/pan)."""
    s = pd.Series(np.asarray(series, dtype=float))
    df = pd.DataFrame({"day": np.arange(1, len(s) + 1), "daily kWh": s,
                       "7-day average": s.rolling(7, min_periods=1).mean()})
    long = df.melt("day", var_name="series", value_name="kWh")
    y_max = float(np.nanmax(s)) * 1.05 if np.isfinite(np.nanmax(s)) else 1.0
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("day:Q", title="Day", scale=alt.Scale(domain=[1, len(s)], nice=False)),
            y=alt.Y("kWh:Q", title="kWh", scale=alt.Scale(domain=[0, y_max], nice=False)),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=["daily kWh", "7-day average"],
                                            range=["#9ab8e0", "#d1495b"])),
            opacity=alt.condition(alt.datum.series == "daily kWh", alt.value(0.7), alt.value(1.0)),
        )
        .properties(height=320)
    )

st.set_page_config(page_title="Electricity Theft Detection", layout="wide")


@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model("models/theft_model.keras")
    cfg = json.load(open("models/config.json"))
    return model, cfg


st.title("⚡ Electricity Theft Detection")
st.caption("Wide & Deep CNN on daily smart-meter readings · TensorFlow")

if not os.path.exists("models/theft_model.keras"):
    st.error("No trained model found. Run `python src/train.py` first.")
    st.stop()

model, cfg = load_model()
n_days = cfg["n_days"]

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold", 0.05, 0.95, float(cfg["threshold"]), 0.01,
                          help="Higher = fewer false alarms, but more missed thieves")
    if os.path.exists("reports/metrics.json"):
        m = json.load(open("reports/metrics.json"))["dnn_tuned_threshold"]
        st.metric("Test accuracy", f"{m['accuracy']:.1%}")
        st.metric("Test AUC", f"{m['auc']:.3f}")
        st.metric("Recall (thieves caught)", f"{m['recall']:.1%}")

uploaded = st.file_uploader("Upload a CSV (CONS_NO, [FLAG], then one column per day)", type="csv")
use_sample = st.button("…or use 300 sample customers")

df = None
if uploaded:
    df = pd.read_csv(uploaded)
elif use_sample or st.session_state.get("sample"):
    st.session_state["sample"] = True
    path = "data/synthetic_sgcc.csv" if os.path.exists("data/synthetic_sgcc.csv") else "data/sample_customers.csv"
    df = pd.read_csv(path, nrows=300)

if df is not None:
    ids = df["CONS_NO"].astype(str).values
    flags = df["FLAG"].values if "FLAG" in df else None
    X = df.drop(columns=[c for c in ["CONS_NO", "FLAG"] if c in df]).values.astype("float32")
    if X.shape[1] != n_days:
        st.error(f"The model was trained on {n_days} days per customer, this file has {X.shape[1]}.")
        st.stop()

    daily, weekly, stats = build_inputs(X)
    probs = model.predict({"daily": daily, "weekly": weekly, "stats": stats}, verbose=0).ravel()

    out = pd.DataFrame({"customer": ids, "theft_probability": probs.round(3),
                        "flagged": probs >= threshold})
    if flags is not None:
        out["actual_label"] = np.where(flags == 1, "theft", "normal")
    out = out.sort_values("theft_probability", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers scored", len(out))
    c2.metric("Flagged for inspection", int(out.flagged.sum()))
    if flags is not None:
        hits = ((out.flagged) & (out.actual_label == "theft")).sum()
        c3.metric("Flag precision", f"{hits / max(out.flagged.sum(), 1):.0%}")

    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Inspection priority list")
        st.dataframe(out, height=420, width="stretch")
        st.download_button("Download CSV", out.to_csv(index=False), "flagged_customers.csv")
    with right:
        st.subheader("Inspect a customer")
        cid = st.selectbox("Customer", out.customer.tolist())
        row = np.where(ids == cid)[0][0]
        st.altair_chart(customer_chart(X[row]), width="stretch")
        st.write(f"Theft probability: **{probs[row]:.1%}**")
        st.caption("Look for sudden permanent drops, flat-lined readings, or many zero days.")
