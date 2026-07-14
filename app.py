import streamlit as st
import pandas as pd
import pickle

with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("model_info.pkl", "rb") as f:
    model_info = pickle.load(f)

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Predictor")
st.write(
    f"This app uses a **K-Nearest Neighbors (KNN)** model "
    f"(trained accuracy: {round(model_info['accuracy'] * 100, 2)}%) "
    "to estimate heart disease risk based on patient health data."
)
st.caption(
    "This tool is for educational purposes as part of a DecodeLabs internship "
    "project and is not a substitute for professional medical advice."
)

st.divider()

st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 90, 50)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cp = st.selectbox(
        "Chest Pain Type",
        options=["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
    )
    trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 220, 130)
    chol = st.slider("Cholesterol (chol)", 80, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["No", "Yes"])

with col2:
    thalch = st.slider("Max Heart Rate Achieved (thalch)", 60, 220, 150)
    exang = st.selectbox("Exercise-Induced Angina?", options=["No", "Yes"])
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

if st.button("Predict Risk", type="primary"):

    input_data = pd.DataFrame([input_row])[model_info["columns"]]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Result")

    if prediction == 1:
        st.error("⚠️ Higher risk of heart disease detected.")
        st.write(f"Model confidence: {round(probability[1] * 100, 2)}%")
    else:
        st.success("✅ Lower risk of heart disease detected.")
        st.write(f"Model confidence: {round(probability[0] * 100, 2)}%")

    st.caption(
        "Remember: this is a machine learning estimate based on patterns "
        "in historical data, not a medical diagnosis."
    )

st.divider()
st.caption("Built by Talha Gillani — DecodeLabs AI Engineering Internship, Batch 2026")
