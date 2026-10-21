"""Streamlit app: interactive Titanic survival prediction.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "titanic_model.joblib"
METRICS_PATH = ROOT / "model" / "metrics.json"
FIG = ROOT / "figures"

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())


if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

model = load_model()
metrics = load_metrics()

st.title("🚢 Titanic Survival Predictor")
st.caption(f"Model: {metrics['selected_model']} (tuned, scikit-learn pipeline)")

tab_pred, tab_models, tab_eda = st.tabs(["Predict", "Model performance", "Data exploration"])

with tab_pred:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Passenger details")
        sex = st.radio("Sex", ["female", "male"], horizontal=True)
        title_opts = ["Mr", "Mrs", "Miss", "Master", "Rare"]
        default_title = "Mr" if sex == "male" else "Mrs"
        title = st.selectbox(
            "Title", title_opts, index=title_opts.index(default_title),
            help="'Master' = young boy; 'Rare' = Dr, Rev, Col, Lady, etc.",
        )
        pclass = st.selectbox("Ticket class", [1, 2, 3], index=2)
        age_known = st.checkbox("Age known", value=True)
        age = st.slider("Age", 0, 80, 30) if age_known else None
        fare = st.number_input("Total ticket fare (£)", 0.0, 520.0, float(round(metrics["defaults"]["Fare"], 1)))
        embarked = st.selectbox(
            "Port of embarkation", ["S", "C", "Q"],
            format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x],
        )
        family = st.slider("Family size aboard (including the passenger)", 1, 11, 1)
        has_cabin = st.checkbox("Cabin number recorded", value=(pclass == 1))

    # Build one row with the exact schema the model was trained on
    capped_fare = min(fare, 249.0)  # training clipped fares at the 99th percentile
    row = pd.DataFrame([{
        "Age": age if age is not None else float("nan"),
        "Fare": capped_fare,
        "FamilySize": family,
        "FarePerPerson": capped_fare / family,
        "Pclass": str(pclass),
        "Sex": sex,
        "Embarked": embarked,
        "Title": title,
        "IsAlone": str(int(family == 1)),
        "HasCabin": str(int(has_cabin)),
    }])[metrics["features"]]

    with right:
        st.subheader("Prediction")
        prob = float(model.predict_proba(row)[0, 1])
        label = "Survived" if prob >= 0.5 else "Did not survive"
        st.metric("Survival probability", f"{prob:.1%}")
        st.progress(prob)
        if prob >= 0.5:
            st.success(f"Predicted outcome: **{label}**")
        else:
            st.error(f"Predicted outcome: **{label}**")
        if (sex == "male" and title in ("Mrs", "Miss")) or (sex == "female" and title in ("Mr", "Master")):
            st.warning("Sex and title look inconsistent - the prediction may be unreliable.")
        with st.expander("Model input"):
            st.dataframe(row.T.astype(str).rename(columns={0: "value"}))

with tab_models:
    st.subheader("Held-out test set performance")
    tm = metrics["test_metrics"]
    cols = st.columns(5)
    for c, k in zip(cols, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
        c.metric(k, f"{tm[k]:.3f}")
    st.markdown("**All tuned models (test set)**")
    st.dataframe(pd.DataFrame(metrics["test_all_models"]).round(4), hide_index=True)
    st.markdown("**Model comparison (5-fold cross-validation, before tuning)**")
    st.dataframe(pd.DataFrame(metrics["cv_comparison"]).round(4), hide_index=True)
    c1, c2 = st.columns(2)
    c1.image(str(FIG / "evaluation.png"), caption="Confusion matrix and ROC curves")
    c2.image(str(FIG / "feature_importance.png"), caption="Permutation feature importance")

with tab_eda:
    st.image(str(FIG / "eda_overview.png"), caption="Exploratory analysis")
    c1, c2 = st.columns(2)
    c1.image(str(FIG / "correlation_heatmap.png"))
    c2.image(str(FIG / "missing_values.png"))
