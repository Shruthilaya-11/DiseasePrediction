import streamlit as st
import pandas as pd
from joblib import load
import os

st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")
st.title("Multiple Disease Prediction")

MODEL_DIR = r"C:\Users\Shruthilaya\GUVI\output\models"  

@st.cache_resource
def load_models():
    models = {}
    for key in ["parkinsons", "kidney", "liver"]:
        path = os.path.join(MODEL_DIR, f"{key}.joblib")
        if os.path.exists(path):
            try:
                models[key] = load(path)
            except Exception:
                models[key] = None
        else:
            models[key] = None
    return models

models = load_models()

def model_predict_ui(title, model_key, sample_inputs: dict = None):
    st.header(title)
    model = models.get(model_key)
    if model is None:
        st.warning(f"Model for '{model_key}' not found. Train and place at {MODEL_DIR}/{model_key}.joblib")
        return

    st.write("Enter values (leave blank to use defaults):")
    if sample_inputs is None:
        try:
            st.info("No sample schema provided. The app will ask for a CSV input matching training columns.")
            csv = st.file_uploader("Upload a single-row CSV with features (columns must match training columns)", type=["csv"])
            if csv is not None:
                df = pd.read_csv(csv)
                try:
                    prob = float(model.predict_proba(df)[:,1][0])
                except Exception:
                    score = model.decision_function(df)[0]
                    prob = (score - score) / (1 + abs(score)) 
                st.metric("Predicted Risk", f"{prob:.2%}")
            return
        except Exception:
            st.error("Couldn't infer columns. Upload a single-row CSV.")
            return

    with st.form(f"form_{model_key}"):
        inputs = {}
        cols = st.columns(3)
        keys = list(sample_inputs.keys())
        for i, k in enumerate(keys):
            with cols[i % 3]:
                v = sample_inputs[k]
                if isinstance(v, (int, float)):
                    inputs[k] = st.number_input(k, value=float(v))
                elif isinstance(v, list):
                    inputs[k] = st.selectbox(k, v)
                else:
                    inputs[k] = st.text_input(k, value=str(v))
        submitted = st.form_submit_button("Predict")
    if submitted:
        df = pd.DataFrame([inputs])
        try:
            prob = float(models[model_key].predict_proba(df)[:,1][0])
        except Exception:
            try:
                score = models[model_key].decision_function(df)
                smin, smax = score.min(), score.max()
                prob = float((score - smin) / (smax - smin + 1e-9))
            except Exception:
                prob = 0.0
        st.metric("Predicted Risk", f"{prob:.2%}")
        if prob >= 0.7:
            st.error("High risk — consider consulting a healthcare professional.")
        elif prob >= 0.4:
            st.warning("Moderate risk.")
        else:
            st.success("Low risk.")

sample_parkinsons = {
    "MDVP:Fo(Hz)": 120.0,
    "MDVP:Fhi(Hz)": 150.0,
    "MDVP:Flo(Hz)": 90.0,
    "MDVP:Jitter(%)": 0.01,
    "MDVP:Shimmer": 0.05
}
sample_kidney = {
    "age": 45, "bp": 80, "sg": 1.02, "al": 1, "su": 0,
    "rbc": ["normal", "abnormal"], "pc": ["normal", "abnormal"]
}
sample_liver = {
    "Age": 45, "Gender": ["Male", "Female"], "Total_Bilirubin": 1.0,
    "Alkaline_Phosphotase": 200, "Alamine_Aminotransferase": 30,
    "Aspartate_Aminotransferase": 35, "Albumin": 3.5, "A/G_Ratio": 1.0
}

st.sidebar.title("Pick disease")
choice = st.sidebar.selectbox("Disease", ["Parkinsons", "Kidney", "Liver"])
if choice == "Parkinsons":
    model_predict_ui("Parkinson's Prediction", "parkinsons", sample_parkinsons)
elif choice == "Kidney":
    model_predict_ui("Kidney Disease Prediction", "kidney", sample_kidney)
else:
    model_predict_ui("Liver Disease Prediction", "liver", sample_liver)
