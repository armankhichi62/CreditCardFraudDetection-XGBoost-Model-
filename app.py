import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

st.set_page_config(page_title="Fraud Detection - XGBoost", layout="centered")

# Load all artifacts (model + preprocessors)
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

    return booster, scaler, encoders, features


# ---- LOAD MODEL & PREPROCESSORS ----
booster, scaler, encoders, features = load_artifacts()

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
        # ---- PREPARE NUMERIC FEATURES ----
        X_num = np.array([[num_inputs[c] for c in numeric_features]])
        X_num_scaled = scaler.transform(X_num)

        # ---- ENCODE CATEGORICAL FEATURES ----
        cat_encoded = []
        for col in categorical_features:
            le = encoders[col]
            val = str(cat_inputs[col])

            if val in le.classes_:
                enc = int(le.transform([val])[0])
            else:
                # fallback for unseen category
                enc = int(le.transform([le.classes_[0]])[0])

            cat_encoded.append(enc)

        # ---- FINAL INPUT VECTOR ----
        final_input = np.hstack([X_num_scaled, np.array([cat_encoded])])

        # ---- XGBOOST PREDICTION ----
        dtest = xgb.DMatrix(final_input)
        prob = float(booster.predict(dtest)[0])
        pred = 1 if prob > 0.5 else 0

        # ---- DISPLAY RESULTS ----
        st.success("Prediction Complete")
        st.write(f"**Fraud Probability:** {prob:.4f}")
        st.write(f"**Prediction:** {'FRAUD' if pred==1 else 'LEGIT'}")

    except Exception as e:
        st.error(f"Error: {e}")
