import numpy as np 
import pandas as pd 
import pickle
import os 
import streamlit as st

base_path = os.path.dirname(__file__)

# Construct absolute paths
scaler_path = os.path.join(base_path, "Models", "standard.pkl")
model_path = os.path.join(base_path, "Models", "logisticModel.pkl")
## Load Pickel files 
scaler = None
model = None
try:
    with open(scaler_path,'rb') as f:
        scaler = pickle.load(f)
    with open(model_path,"rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'Models/standard.pkl' and 'Models/logisticModel.pkl' exist.")

## App Header 
st.title("🔭 Magic Gamma Telescope Prediction 🛰️")

st.markdown("Enter The Magic Gamma Telescope Features Values To Predicty The Classifiction")


with st.sidebar:
    st.title("👨🏻‍💻 Developer Info")
    st.write("**Name:** Rasheed Ahmad")
    st.write("**Role:** ML Engineer")
    st.divider()
    st.subheader("📊 Model Accuracy")
    st.metric(label="Model Accuracy",value="79.08%",delta="0.05")
    st.info("Algorithm: Logistic Regression")


def get_user_input():
    col1,col2 = st.columns(2)
    with col1:
        fLength = st.number_input("Enter fLength", value=0.0)
        fwidth = st.number_input("Enter fWidth", value=0.0)
        fSize = st.number_input("Enter fSize", value=0.0)
        fConc = st.number_input("Enter fConc", value=0.0)
        fConc1 = st.number_input("Enter fConc1", value=0.0)
    with col2:
        fAsym = st.number_input("Enter fAsym", value=0.0)
        fM3Long = st.number_input("Enter fM3Long", value=0.0)
        fM3Trans = st.number_input("Enter fM3Trans", value=0.0)
        fAlpha = st.number_input("Enter fAlpha", value=0.0)
        fDist = st.number_input("Enter fDist", value=0.0)

    data = np.array([[fLength,fwidth,fSize,fConc,fConc1,fAsym,fM3Long,fM3Trans,fAlpha,fDist]])
    scaled = scaler.transform(data)
    return scaled

input_data = get_user_input()

if st.button("Predict 🚀"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.divider()
    if prediction[0] == 1:
        st.success(f"### Prediction: **Hadron (Background)** 🌑")
    else:
        st.error(f"### Prediction: **Gamma (Signal)** 🌟")
    
    st.write(f"**Confidence Score:** {np.max(probability)*100:.2f}%")