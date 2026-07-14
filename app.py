import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("model_info.pkl", "rb") as f:
    model_info = pickle.load(f)

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# ---------- custom styling ----------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* animated gradient app background */
.stApp {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #3a0d12);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* hero title */
.big-title {
    font-family: 'Poppins', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff4b4b, #ff8b8b, #ffb199);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
    animation: heartbeat 1.8s ease-in-out infinite;
}
@keyframes heartbeat {
    0%, 100% { transform: scale(1); }
    15% { transform: scale(1.04); }
    30% { transform: scale(1); }
    45% { transform: scale(1.03); }
    60% { transform: scale(1); }
}
.subtitle {
    color: #d8d8f0cc;
    font-size: 1.05rem;
    margin-top: -8px;
}

/* glass cards everywhere */
[data-testid="stVerticalBlockBorderWrapper"], .metric-card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

/* sidebar glass */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.85);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: #f0f0ff !important;
}

/* metric widgets */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.2);
    transition: transform 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: #ff4b4b66;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,0.04);
    padding: 6px;
    border-radius: 14px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #d8d8f0;
    font-weight: 600;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #ff4b4b, #ff8b8b) !important;
    color: white !important;
}

/* primary button glow */
button[kind="primary"] {
    background: linear-gradient(90deg, #ff4b4b, #ff7b7b) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    box-shadow: 0 0 20px rgba(255,75,75,0.45);
    transition: all 0.25s ease;
}
button[kind="primary"]:hover {
    box-shadow: 0 0 30px rgba(255,75,75,0.75);
    transform: translateY(-2px) scale(1.01);
}

/* headings glow */
h2, h3 {
    color: #f5f5ff !important;
}

/* badges */
.risk-badge-high {
    background: linear-gradient(90deg, #ff4b4b33, #ff4b4b11);
    color: #ff7b7b;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    border: 1px solid #ff4b4b55;
    display: inline-block;
}
.risk-badge-ok {
    background: linear-gradient(90deg, #21c35433, #21c35411);
    color: #4ee08a;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    border: 1px solid #21c35455;
    display: inline-block;
}

/* dataframes / expanders rounding */
[data-testid="stExpander"], .stDataFrame {
    border-radius: 14px !important;
    overflow: hidden;
}

/* scrollbar */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #ff4b4b88; border-radius: 10px; }

/* divider glow */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #ff4b4b88, transparent);
}
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### ❤️ About this model")
    st.metric("Model", "K-Nearest Neighbors")
    st.metric("Accuracy", f"{round(model_info['accuracy'] * 100, 2)}%")
    if "best_k" in model_info:
        st.metric("Best K", model_info["best_k"])
    st.markdown("---")
    st.markdown("**Features used:**")
    st.caption(", ".join(model_info["columns"]))
    st.markdown("---")
    st.caption(
        "This tool is for educational purposes as part of a DecodeLabs "
        "internship project and is not a substitute for professional "
        "medical advice."
    )
    st.markdown("---")
    st.caption(f"Session predictions made: **{len(st.session_state.history)}**")

# ---------- header ----------
st.markdown('<p class="big-title">❤️ Heart Disease Risk Predictor</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="subtitle">Powered by a <b>KNN classifier</b> trained on the UCI Heart '
    f'Disease dataset — <b>{round(model_info["accuracy"] * 100, 2)}%</b> test accuracy. '
    'Fill in the patient\'s details below to get an instant risk estimate.</p>',
    unsafe_allow_html=True
)
st.write("")

tab_predict, tab_insights, tab_history, tab_about = st.tabs(
    ["🔍 Predict", "📊 Model Insights", "🕓 History", "ℹ️ About"]
)

# =====================================================
# TAB 1: PREDICT
# =====================================================
with tab_predict:
    st.subheader("Enter Patient Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demographics**")
        age = st.slider("Age", 20, 90, 50)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type",
            options=["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
        )
        trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 220, 130)

    with col2:
        st.markdown("**🩸 Bloodwork**")
        chol = st.slider("Cholesterol (chol)", 80, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["No", "Yes"])
        thalch = st.slider("Max Heart Rate Achieved (thalch)", 60, 220, 150)
        exang = st.selectbox("Exercise-Induced Angina?", options=["No", "Yes"])

    with col3:
        st.markdown("**📈 ECG / Stress Test**")
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", options=["upsloping", "flat", "downsloping"])
        ca = st.slider("Number of Major Vessels (ca)", 0, 3, 0)
        thal = st.selectbox("Thalassemia (thal)", options=["normal", "fixed defect", "reversable defect"])

    input_row = {
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "cp": encoders["cp"].transform([cp])[0],
        "thalch": thalch,
        "age": age,
        "sex": encoders["sex"].transform([sex])[0],
        "ca": ca,
        "slope": encoders["slope"].transform([slope])[0],
        "thal": encoders["thal"].transform([thal])[0],
        "trestbps": trestbps,
        "fbs": 1 if fbs == "Yes" else 0,
        "chol": chol,
    }

    st.divider()
    predict_clicked = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

    if predict_clicked:
        input_data = pd.DataFrame([input_row])[model_info["columns"]]
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        risk_pct = round(probability[1] * 100, 2)

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "age": age, "sex": sex, "risk_%": risk_pct,
            "result": "High Risk" if prediction == 1 else "Low Risk"
        })

        st.subheader("Result")
        r1, r2 = st.columns([1, 1.3])

        with r1:
            if prediction == 1:
                st.error("⚠️ Higher risk of heart disease detected.")
            else:
                st.success("✅ Lower risk of heart disease detected.")
            st.metric("Disease Probability", f"{risk_pct}%")
            st.metric("No-Disease Probability", f"{round(probability[0]*100, 2)}%")
            st.progress(float(probability[1]))

        with r2:
            fig, ax = plt.subplots(figsize=(4, 3.2), subplot_kw={"aspect": "equal"})
            colors = ["#21c354", "#ff4b4b"]
            ax.pie(
                probability, labels=["No Disease", "Disease"],
                autopct="%1.1f%%", colors=colors, startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2}
            )
            ax.set_title("Risk Breakdown")
            st.pyplot(fig)

        # flag concerning inputs
        st.markdown("#### ⚑ Risk Factor Flags")
        flags = []
        if chol > 240:
            flags.append(("High Cholesterol", chol))
        if trestbps > 140:
            flags.append(("High Resting BP", trestbps))
        if oldpeak > 2:
            flags.append(("High ST Depression", oldpeak))
        if exang == "Yes":
            flags.append(("Exercise-Induced Angina", "Yes"))
        if ca > 0:
            flags.append(("Major Vessels Blocked", ca))

        if flags:
            badge_cols = st.columns(len(flags))
            for c, (label, val) in zip(badge_cols, flags):
                c.markdown(f'<span class="risk-badge-high">⚠️ {label}: {val}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="risk-badge-ok">✅ No major risk factors flagged</span>', unsafe_allow_html=True)

        st.caption(
            "Remember: this is a machine learning estimate based on patterns "
            "in historical data, not a medical diagnosis."
        )

        with st.expander("See raw input sent to the model"):
            st.dataframe(input_data, use_container_width=True)

# =====================================================
# TAB 2: MODEL INSIGHTS
# =====================================================
with tab_insights:
    st.subheader("How the model was trained")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Type", "KNN")
    c2.metric("Accuracy", f"{round(model_info['accuracy']*100, 2)}%")
    c3.metric("Features Used", len(model_info["columns"]))

    st.markdown("#### Training Graphs")
    g1, g2 = st.columns(2)
    try:
        with g1:
            st.image("graph_class_balance.png", caption="Class Balance")
            st.image("graph_correlation_heatmap.png", caption="Correlation Heatmap")
        with g2:
            st.image("graph_age_distribution.png", caption="Age Distribution")
            st.image("graph_confusion_matrix.png", caption="Confusion Matrix")
    except Exception:
        st.info("Training graphs not found in this folder — run train_model.py first.")

# =====================================================
# TAB 3: HISTORY
# =====================================================
with tab_history:
    st.subheader("Predictions made this session")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        st.download_button(
            "⬇️ Download history as CSV",
            hist_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv"
        )
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet — go to the Predict tab and try one.")

# =====================================================
# TAB 4: ABOUT
# =====================================================
with tab_about:
    st.subheader("About this project")
    st.write("""
    This app estimates heart disease risk using a **K-Nearest Neighbors (KNN)**
    model trained on the UCI Heart Disease dataset (920 patients across
    Cleveland, Hungary, Switzerland, and VA Long Beach).

    **Pipeline:**
    - Cleaned missing values and unrealistic outliers
    - Encoded categorical fields (sex, chest pain type, slope, thalassemia)
    - Selected features based on correlation with the target
    - Tuned K and picked the best-performing value
    - Deployed here with Streamlit
    """)
    st.caption(
        "Built by Talha Gillani — DecodeLabs AI Engineering Internship, Batch 2026"
    )
