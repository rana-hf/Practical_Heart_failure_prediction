import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('heart_model.pkl')
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('selector.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, selector, model_columns

model, scaler, selector, model_columns = load_assets()

# --- 2. UI DESIGN ---
st.title("Heart Failure Prediction App ❤️")
st.write("Enter patient data to predict heart disease risk.")

# --- 3. USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 0, 600, 200)
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)

with col2:
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

# --- 4. PREPROCESSING ---
def preprocess():
    # إنشاء DataFrame ببيانات المستخدم
    data = {
        'Age': [age], 'RestingBP': [resting_bp], 'Cholesterol': [cholesterol],
        'FastingBS': [fbs], 'MaxHR': [max_hr], 'Oldpeak': [oldpeak],
        'Sex': [1 if sex == "M" else 0],
        'ExerciseAngina': [1 if exercise_angina == "Y" else 0]
    }
    input_df = pd.DataFrame(data)

    # معالجة الأعمدة اللي عملنا لها Get Dummies في الكولاب
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    if f"RestingECG_{resting_ecg}" in model_columns: input_df[f"RestingECG_{resting_ecg}"] = 1
    if f"ST_Slope_{st_slope}" in model_columns: input_df[f"ST_Slope_{st_slope}"] = 1
    if f"ChestPainType_{chest_pain}" in model_columns: input_df[f"ChestPainType_{chest_pain}"] = 1

    # ترتيب الأعمدة
    input_df = input_df[model_columns]
    
    # Scaling & Selection
    scaled = scaler.transform(input_df)
    selected = selector.transform(scaled)
    return selected

# --- 5. PREDICTION ---
if st.button("Predict Result"):
    input_data = preprocess()
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ Warning: High Risk of Heart Disease")
    else:
        st.success("✅ Good News: Low Risk of Heart Disease")
