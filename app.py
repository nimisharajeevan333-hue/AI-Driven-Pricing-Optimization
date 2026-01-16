import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="AI Pricing Optimization", layout="centered")
st.title("💰 AI-Driven Pricing Optimization")

# ---------- Load model & features ----------
@st.cache_resource
def load_model():
    with open("pricing_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_features():
    with open("model_features.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
feature_names = load_features()   # ALL features used in training

# ---------- UI ----------
st.header("🔢 Enter Main Inputs")

# Choose ONLY 5 important features to show
ui_features = feature_names[:5]

default_values = [120, 250, 300, 10, 500]

user_inputs = {}

for i, feature in enumerate(ui_features):
    user_inputs[feature] = st.number_input(
        label=feature,
        value=float(default_values[i])
    )

# ---------- Build FULL input dataframe ----------
input_data = {}

for feature in feature_names:
    if feature in user_inputs:
        input_data[feature] = user_inputs[feature]
    else:
        input_data[feature] = 0.0   # fill missing features safely

input_df = pd.DataFrame([input_data])

# ---------- Prediction ----------
if st.button("📈 Predict Optimal Price"):
    prediction = model.predict(input_df)
    st.success(f"✅ Predicted Price: ₹ {round(prediction[0], 2)}")
