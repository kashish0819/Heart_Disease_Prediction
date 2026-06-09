import streamlit as st
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# Load Model
model = joblib.load("heart_model.pkl")

# Title
st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details below to predict heart disease risk.")

# Inputs

age = st.number_input(
    "Age",
    min_value=20,
    max_value=100,
    value=50
)

sex = st.selectbox(
    "Gender",
    ["Female", "Male"]
)
sex = 0 if sex == "Female" else 1

cp_option = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ]
)

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}

cp = cp_map[cp_option]

trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol Level (mg/dl)",
    min_value=100,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["No", "Yes"]
)
fbs = 1 if fbs == "Yes" else 0

restecg = st.selectbox(
    "Resting ECG Results",
    [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ]
)

restecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

restecg = restecg_map[restecg]

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=60,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)
exang = 1 if exang == "Yes" else 0

oldpeak = st.number_input(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

slope = slope_map[slope]

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal_option = st.selectbox(
    "Thalassemia",
    [
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ]
)

thal_map = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

thal = thal_map[thal_option]

# Prediction Button
if st.button("🔍 Predict Heart Disease Risk"):

    data = np.array([[age,
                      sex,
                      cp,
                      trestbps,
                      chol,
                      fbs,
                      restecg,
                      thalach,
                      exang,
                      oldpeak,
                      slope,
                      ca,
                      thal]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
        st.warning("Please consult a healthcare professional for further evaluation.")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.balloons()

# Footer
st.markdown("---")
st.caption("Developed using Machine Learning (Random Forest Classifier)")