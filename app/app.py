"""
Streamlit App for ML Model Deployment
=====================================

This is your Streamlit application that deploys both your regression and
classification models. Users can input feature values and get predictions.

HOW TO RUN LOCALLY:
    streamlit run app/app.py

HOW TO DEPLOY TO STREAMLIT CLOUD:
    1. Push your code to GitHub
    2. Go to share.streamlit.io
    3. Connect your GitHub repo
    4. Set the main file path to: app/app.py
    5. Deploy!

WHAT YOU NEED TO CUSTOMIZE:
    1. Update the page title and description
    2. Update feature input fields to match YOUR features
    3. Update the model paths if you changed them
    4. Customize the styling if desired

Author: [Kalpana Selva]
Dataset:[Concrete Compressive Strength Data Set](https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set)
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTS
# =============================================================================
STRENGTH_ICONS = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}
BAR_COLORS     = {"Low": "#e74c3c", "Medium": "#f39c12", "High": "#2ecc71"}
GAUGE_MAX_MPA  = 90

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource
def load_models():
    """Load all saved models and artifacts."""
    base_path = Path(__file__).parent.parent / "models"

    try:
        with open(base_path / "model_metrics.json") as f:
            metrics = json.load(f)

        return {
            "regression_model":        joblib.load(base_path / "regression_model.pkl"),
            "regression_scaler":       joblib.load(base_path / "regression_scaler.pkl"),
            "regression_features":     joblib.load(base_path / "regression_features.pkl"),
            "classification_model":    joblib.load(base_path / "classification_model.pkl"),
            "classification_scaler":   joblib.load(base_path / "classification_scaler.pkl"),
            "classification_features": joblib.load(base_path / "classification_features.pkl"),
            "label_encoder":           joblib.load(base_path / "label_encoder.pkl"),
            "binning_info":            joblib.load(base_path / "binning_info.pkl"),
            "metrics":                 metrics,
        }
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Make sure you've trained and saved your models in the notebooks first!")
        st.stop()


def engineer_features(raw):
    """Compute engineered features from raw mix design inputs.

    The models were trained on engineered features (ratios, log-transform),
    not the raw ingredient quantities. This function derives those features
    so users can enter intuitive mix design values.
    """
    cement, slag, fly_ash = raw["cement"], raw["slag"], raw["fly_ash"]
    water, sp             = raw["water"], raw["superplasticizer"]
    coarse_agg, fine_agg  = raw["coarse_agg"], raw["fine_agg"]
    age                   = raw["age"]

    total_binder    = cement + slag + fly_ash
    total_aggregate = coarse_agg + fine_agg

    return {
        "cement":                 cement,
        "blast_furnace_slag":     slag,
        "fly_ash":                fly_ash,
        "water":                  water,
        "superplasticizer":       sp,
        "coarse_aggregate":       coarse_agg,
        "fine_aggregate":         fine_agg,
        "age":                    age,
        "water_cement_ratio":     water / cement                if cement > 0       else 0,
        "total_binder":           total_binder,
        "cement_ratio":           cement / total_binder         if total_binder > 0 else 0,
        "water_binder_ratio":     water  / total_binder         if total_binder > 0 else 0,
        "aggregate_binder_ratio": total_aggregate / total_binder if total_binder > 0 else 0,
        "fine_coarse_ratio":      fine_agg / coarse_agg         if coarse_agg > 0   else 0,
        "log_age":                np.log1p(age),
    }


def build_input(features_dict, feature_list):
    """Select model features from the full features dict and return a DataFrame."""
    return pd.DataFrame([{f: features_dict[f] for f in feature_list}])


def make_regression_prediction(models, input_data):
    """Make a regression prediction."""
    input_scaled = models["regression_scaler"].transform(input_data)
    return models["regression_model"].predict(input_scaled)[0]


def make_classification_prediction(models, input_data):
    """Make a classification prediction."""
    input_scaled = models["classification_scaler"].transform(input_data)
    encoded      = models["classification_model"].predict(input_scaled)[0]
    proba        = models["classification_model"].predict_proba(input_scaled)[0]
    label        = models["label_encoder"].inverse_transform([encoded])[0]
    classes      = models["label_encoder"].classes_
    return label, encoded, proba, classes


def render_mix_design_inputs(key_prefix):
    """Render the 8 raw concrete mix design input fields and return their values."""
    col1, col2 = st.columns(2)
    with col1:
        cement     = st.number_input("Cement (kg/m³)",             0.0,   600.0,  300.0, 1.0, key=f"{key_prefix}_cement")
        fly_ash    = st.number_input("Fly Ash (kg/m³)",            0.0,   200.0,  0.0,   1.0, key=f"{key_prefix}_fly_ash")
        sp         = st.number_input("Superplasticizer (kg/m³)",   0.0,   35.0,   6.0,   0.1, key=f"{key_prefix}_sp")
        coarse_agg = st.number_input("Coarse Aggregate (kg/m³)",   700.0, 1200.0, 950.0, 1.0, key=f"{key_prefix}_coarse")
    with col2:
        slag       = st.number_input("Blast Furnace Slag (kg/m³)", 0.0,   400.0,  0.0,   1.0, key=f"{key_prefix}_slag")
        water      = st.number_input("Water (kg/m³)",              100.0, 250.0,  180.0, 1.0, key=f"{key_prefix}_water")
        fine_agg   = st.number_input("Fine Aggregate (kg/m³)",     500.0, 1000.0, 750.0, 1.0, key=f"{key_prefix}_fine")
        age        = st.number_input("Age (days)",                  1,     365,    28,    1,   key=f"{key_prefix}_age")
    return {
        "cement": cement, "slag": slag, "fly_ash": fly_ash,
        "water": water, "superplasticizer": sp,
        "coarse_agg": coarse_agg, "fine_agg": fine_agg, "age": age,
    }


# =============================================================================
# SIDEBAR — Navigation
# =============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "📈 Regression Model", "🏷️ Classification Model"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This app deploys machine learning models trained on the
    **Concrete Compressive Strength** dataset (UCI).

    - **Regression**: Predicts compressive strength (MPa)
    - **Classification**: Predicts strength category (Low / Medium / High)
    """
)
st.sidebar.markdown("**Built by:** Kalpana Selva")

# =============================================================================
# LOAD MODELS (shared across all pages)
# =============================================================================
models = load_models()

# =============================================================================
# HOME PAGE
# =============================================================================
if page == "🏠 Home":
    st.title("🏗️ Concrete Compressive Strength Predictor")
    st.markdown("### Welcome!")
    st.write(
        """
        This application allows you to make predictions about concrete compressive strength
        using trained machine learning models.

        **What you can do:**
        - 📈 **Regression Model**: Predict the exact compressive strength in MPa
        - 🏷️ **Classification Model**: Predict the strength category (Low / Medium / High)

        Use the sidebar to navigate between models.
        """
    )

    st.markdown("---")
    st.markdown("### About This Project")

    reg_metrics = models["metrics"]["regression"]
    clf_metrics = models["metrics"]["classification"]

    st.write(
        f"""
        **Dataset:** Concrete Compressive Strength — UCI ML Repository (1,030 samples)

        **Problem Statement:** Predict the compressive strength of concrete mixtures based on
        their ingredient proportions and curing age. Accurate strength prediction reduces
        costly physical testing and enables better mix design decisions.

        **Input Features (8 raw mix design parameters):**
        - Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer (kg/m³)
        - Coarse Aggregate, Fine Aggregate (kg/m³)
        - Age (days)

        **Engineered Features:** Water-binder ratio, total binder, cement ratio,
        aggregate-binder ratio, and log-transformed age — derived automatically from the
        raw inputs above.

        **Models Used:**
        - Regression: Gradient Boosting Regressor (Test R² = {reg_metrics['test_r2']}, RMSE = {reg_metrics['test_rmse']} MPa)
        - Classification: Gradient Boosting Classifier (Test Accuracy = {clf_metrics['test_accuracy'] * 100:.2f}%)
        """
    )

# =============================================================================
# REGRESSION PAGE
# =============================================================================
elif page == "📈 Regression Model":
    st.title("📈 Regression Prediction")
    st.write("Enter concrete mix design parameters to predict compressive strength (MPa).")

    st.markdown("---")
    st.markdown("### Enter Mix Design Parameters")

    raw_inputs = render_mix_design_inputs(key_prefix="reg")
    feats      = engineer_features(raw_inputs)

    st.markdown("---")

    if st.button("🔮 Make Regression Prediction", type="primary"):
        X        = build_input(feats, models["regression_features"])
        strength = make_regression_prediction(models, X)

        reg_metrics = models["metrics"]["regression"]
        bin_info    = models["binning_info"]
        low_th      = bin_info["bins"][1]
        high_th     = bin_info["bins"][2]

        st.success(f"### Predicted Compressive Strength: {strength:.2f} MPa")

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Strength", f"{strength:.2f} MPa")
        col2.metric("Test R²",            f"{reg_metrics['test_r2']:.4f}")
        col3.metric("Test RMSE",          f"{reg_metrics['test_rmse']} MPa")

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = float(strength),
            title = {"text": "Compressive Strength (MPa)", "font": {"size": 16}},
            number= {"suffix": " MPa", "font": {"size": 28}},
            gauge = {
                "axis":  {"range": [0, GAUGE_MAX_MPA], "tickwidth": 1},
                "bar":   {"color": "steelblue", "thickness": 0.25},
                "steps": [
                    {"range": [0,       low_th],        "color": "#ffe0e0"},
                    {"range": [low_th,  high_th],        "color": "#fff8e0"},
                    {"range": [high_th, GAUGE_MAX_MPA],  "color": "#e0f7e0"},
                ],
                "threshold": {
                    "line":      {"color": "navy", "width": 4},
                    "thickness": 0.75,
                    "value":     float(strength),
                },
            },
        ))
        fig.update_layout(height=300, margin=dict(t=60, b=20, l=40, r=40))
        st.plotly_chart(fig, use_container_width=True)

        if strength < low_th:
            st.info(f"**Low strength** ({strength:.2f} MPa < {low_th:.2f} MPa) — suitable for non-structural use (fill, footpaths, kerbs).")
        elif strength < high_th:
            st.success(f"**Normal strength** ({strength:.2f} MPa) — suitable for general structural use (beams, slabs, columns).")
        else:
            st.success(f"**High strength** ({strength:.2f} MPa > {high_th:.2f} MPa) — suitable for demanding structures (bridges, high-rise columns).")

        with st.expander("View Input Summary"):
            st.dataframe(pd.DataFrame([raw_inputs]), use_container_width=True, hide_index=True)

        with st.expander("View Engineered Features"):
            reg_features = models["regression_features"]
            ef_df = pd.DataFrame([{
                "Feature":       k,
                "Value":         round(v, 4),
                "Used by Model": "✓" if k in reg_features else "",
            } for k, v in feats.items()])
            st.dataframe(ef_df, use_container_width=True, hide_index=True)

# =============================================================================
# CLASSIFICATION PAGE
# =============================================================================
elif page == "🏷️ Classification Model":
    st.title("🏷️ Classification Prediction")
    st.write("Enter concrete mix design parameters to predict the strength category.")

    class_labels = models["label_encoder"].classes_
    st.info(f"**Possible Categories:** {', '.join(class_labels)}")

    bin_info = models["binning_info"]
    with st.expander("How were categories created?"):
        st.write(f"Original target: **{bin_info['original_target']}**")
        st.write("Categories were created by quantile-based binning (25th / 75th percentile) on training data:")
        for i, label in enumerate(bin_info["labels"]):
            if i == 0:
                st.write(f"- **{label}**: < {bin_info['bins'][i + 1]:.2f} MPa")
            elif i == len(bin_info["labels"]) - 1:
                st.write(f"- **{label}**: ≥ {bin_info['bins'][i]:.2f} MPa")
            else:
                st.write(f"- **{label}**: {bin_info['bins'][i]:.2f} – {bin_info['bins'][i + 1]:.2f} MPa")

    st.markdown("---")
    st.markdown("### Enter Mix Design Parameters")

    raw_inputs = render_mix_design_inputs(key_prefix="clf")
    feats      = engineer_features(raw_inputs)

    st.markdown("---")

    if st.button("🔮 Make Classification Prediction", type="primary"):
        X                              = build_input(feats, models["classification_features"])
        label, encoded, proba, classes = make_classification_prediction(models, X)

        clf_metrics = models["metrics"]["classification"]
        icon        = STRENGTH_ICONS.get(label, "⚪")

        st.success(f"### Predicted Category: {icon} {label}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Category", f"{icon} {label}")
        col2.metric("Test Accuracy",      f"{clf_metrics['test_accuracy'] * 100:.2f}%")
        col3.metric("F1 (weighted)",      f"{clf_metrics['test_f1_weighted']:.4f}")

        fig = go.Figure()
        for cls, prob in zip(classes, proba):
            fig.add_bar(
                x=[cls], y=[prob * 100],
                marker_color=BAR_COLORS.get(cls, "steelblue"),
                name=cls,
                text=[f"{prob * 100:.1f}%"],
                textposition="outside",
            )
        fig.update_layout(
            title      = "Class Probabilities (%)",
            yaxis      = dict(title="Probability (%)", range=[0, 110]),
            xaxis      = dict(title="Strength Category"),
            showlegend = False,
            height     = 320,
            margin     = dict(t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        prob_cols = st.columns(len(classes))
        for i, cls in enumerate(classes):
            delta = "← predicted" if cls == label else ""
            prob_cols[i].metric(cls, f"{proba[i] * 100:.1f}%", delta=delta)

        with st.expander("View Input Summary"):
            st.dataframe(pd.DataFrame([raw_inputs]), use_container_width=True, hide_index=True)

        with st.expander("View Engineered Features"):
            clf_features = models["classification_features"]
            ef_df = pd.DataFrame([{
                "Feature":       k,
                "Value":         round(v, 4),
                "Used by Model": "✓" if k in clf_features else "",
            } for k, v in feats.items()])
            st.dataframe(ef_df, use_container_width=True, hide_index=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Concrete Compressive Strength Dataset · UCI ML Repository ·
        Built by Kalpana Selva | Full Stack Academy AI & ML Bootcamp
    </div>
    """,
    unsafe_allow_html=True,
)
