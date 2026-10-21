"""Streamlit app: interactive Titanic survival prediction.

Run: streamlit run app.py
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

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        color: #e5e7eb;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .title-wrap {
        padding: 1rem 0 0.5rem 0;
    }
    .hero-box {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    .prediction-box {
        background: linear-gradient(135deg, rgba(14,116,144,0.18), rgba(15,23,42,0.9));
        border: 1px solid rgba(94, 234, 212, 0.35);
        border-radius: 18px;
        padding: 1.4rem 1.2rem;
        margin-top: 0.25rem;
        box-shadow: 0 12px 28px rgba(13, 148, 136, 0.12);
    }
    .prediction-value {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.05em;
        margin-top: 0.3rem;
        color: #a7f3d0;
    }
    .info-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-top: 0.7rem;
    }
    .kpi {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        text-align: center;
        height: 100%;
    }
    .kpi-label {
        color: #cbd5e1;
        font-size: 0.8rem;
        display: block;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .sidebar .block-container {
        background: rgba(15, 23, 42, 0.9);
    }
    div[data-testid="stHorizontalBlock"] > div {
        gap: 0.8rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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

st.markdown(
    """
    <div class="title-wrap">
        <h1 style="margin-bottom:0; color:#f8fafc;">🚢 Titanic Survival Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"Selected model: {metrics['selected_model']} • Tuned scikit-learn pipeline • Real-time survival estimate")

with st.sidebar:
    st.markdown("### Model summary")
    st.markdown(
        f"""
        <div class="info-card">
            <strong>Selected model:</strong><br>
            {metrics['selected_model']}<br><br>
            <strong>Test Accuracy:</strong> {metrics['test_metrics']['Accuracy']:.3f}<br>
            <strong>ROC-AUC:</strong> {metrics['test_metrics']['ROC-AUC']:.3f}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Feature set")
    st.code(", ".join(metrics["features"]))


tab_pred, tab_models, tab_eda = st.tabs(["Predict", "Model performance", "Data exploration"])

with tab_pred:
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Passenger details")
        sex = st.radio("Sex", ["female", "male"], horizontal=True)

        title_opts = ["Mr", "Mrs", "Miss", "Master", "Rare"]
        default_title = "Mr" if sex == "male" else "Mrs"
        title = st.selectbox(
            "Title",
            title_opts,
            index=title_opts.index(default_title),
            help="'Master' = young boy; 'Rare' = Dr, Rev, Col, Lady, etc.",
        )

        pclass = st.selectbox("Ticket class", [1, 2, 3], index=2)
        age_known = st.checkbox("Age known", value=True)
        age = st.slider("Age", 0, 80, 30) if age_known else None
        fare = st.number_input(
            "Total ticket fare (£)",
            0.0,
            520.0,
            float(round(metrics["defaults"]["Fare"], 1)),
        )
        embarked = st.selectbox(
            "Port of embarkation",
            ["S", "C", "Q"],
            format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x],
        )
        family = st.slider("Family size aboard (including the passenger)", 1, 11, 1)
        has_cabin = st.checkbox("Cabin number recorded", value=(pclass == 1))

        with st.expander("How the model sees this passenger"):
            st.caption("The input row is matched to the exact feature schema used during training.")
            st.code(
                "Age, Fare, FamilySize, FarePerPerson, Pclass, Sex, Embarked, Title, IsAlone, HasCabin",
                language="text",
            )

    with right:
        capped_fare = min(fare, 249.0)
        row = pd.DataFrame([
            {
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
            }
        ])[metrics["features"]]

        prob = float(model.predict_proba(row)[0, 1])
        label = "Survived" if prob >= 0.5 else "Did not survive"

        st.markdown("### Prediction")
        st.markdown(
            f"""
            <div class="prediction-box">
                <div style="color:#cbd5e1; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em;">Predicted outcome</div>
                <div class="prediction-value">{prob:.1%}</div>
                <div style="font-size:1.2rem; font-weight:600; margin-top:0.4rem;">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(prob)

        if prob >= 0.5:
            st.success(f"**Predicted outcome:** {label}")
        else:
            st.error(f"**Predicted outcome:** {label}")

        if (sex == "male" and title in ("Mrs", "Miss")) or (sex == "female" and title in ("Mr", "Master")):
            st.warning("Sex and title look inconsistent — the prediction may be less reliable.")

        st.markdown("### Passenger summary")
        summary = {
            "Sex": sex,
            "Title": title,
            "Ticket class": pclass,
            "Age": "Unknown" if age is None else age,
            "Fare": f"£{fare:.2f}",
            "Embarked": embarked,
            "Family size": family,
            "Cabin recorded": "Yes" if has_cabin else "No",
        }
        for key, value in summary.items():
            st.markdown(f"- **{key}:** {value}")

with tab_models:
    st.subheader("Held-out test set performance")
    test_metrics = metrics["test_metrics"]
    cols = st.columns(5)
    for c, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
        with c:
            st.markdown(
                f"""
                <div class="kpi">
                    <span class="kpi-label">{key}</span>
                    <span class="kpi-value">{test_metrics[key]:.3f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Model comparison")
    st.dataframe(pd.DataFrame(metrics["test_all_models"]).round(4), hide_index=True, use_container_width=True)

    st.markdown("### Cross-validation comparison")
    st.dataframe(pd.DataFrame(metrics["cv_comparison"]).round(4), hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.image(str(FIG / "evaluation.png"), caption="Confusion matrix and ROC curves")
    c2.image(str(FIG / "feature_importance.png"), caption="Permutation feature importance")

with tab_eda:
    st.subheader("Exploratory analysis")
    st.image(str(FIG / "eda_overview.png"), caption="Summary of survival patterns")

    c1, c2 = st.columns(2)
    c1.image(str(FIG / "correlation_heatmap.png"), caption="Correlation overview")
    c2.image(str(FIG / "missing_values.png"), caption="Missing value profile")

    if (FIG / "feature_importance.png").exists():
        st.image(str(FIG / "feature_importance.png"), caption="Feature importance")

    st.caption("This dashboard uses the same dataset and modeling logic as the training pipeline to ensure consistency between model development and prediction output.")
