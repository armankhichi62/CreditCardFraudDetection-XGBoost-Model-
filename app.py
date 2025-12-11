import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

st.set_page_config(page_title="Fraud Detection - XGBoost", layout="centered")

# Load artifacts
@st.cache_resource
def load_artifacts():
booster = xgb.Booster()
booster.load_model("xgb_model.json")

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)

    return model, scaler, encoders, features

model, scaler, encoders, features = load_artifacts()

numeric_features = features["numeric"]
categorical_features = features["categorical"]

st.title("Credit Card Fraud Detection (XGBoost)")
st.write("Enter transaction details to predict whether the transaction is fraudulent.")

with st.form("input_form"):
    st.subheader("Numeric Features")
    num_inputs = {}
    cols = st.columns(2)
    for i, col in enumerate(numeric_features):
        with cols[i % 2]:
            num_inputs[col] = st.number_input(col, value=0.0)

    st.subheader("Categorical Features")
    cat_inputs = {}
    for col in categorical_features:
        cat_inputs[col] = st.text_input(col, value="")

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Scale numeric inputs
        X_num = np.array([[num_inputs[c] for c in numeric_features]])
        X_num_scaled = scaler.transform(X_num)

        # Encode categorical
        cat_encoded = []
        for col in categorical_features:
            le = encoders[col]
            val = str(cat_inputs[col])
            if val in le.classes_:
                enc = int(le.transform([val])[0])
            else:
                enc = int(le.transform([le.classes_[0]])[0])
            cat_encoded.append(enc)

        final_input = np.hstack([X_num_scaled, np.array([cat_encoded])])

        # Predict
        prob = float(model.predict_proba(final_input)[0][1])
        pred = 1 if prob > 0.5 else 0

        st.success("Prediction Complete")
        st.write(f"**Fraud Probability:** {prob:.4f}")
        st.write(f"**Prediction:** {'FRAUD' if pred==1 else 'LEGIT'}")

    except Exception as e:
        st.error(f"Error: {e}")
