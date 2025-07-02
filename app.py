# 📦 Import required libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# 🧠 Load trained model and required input features
regressor = joblib.load("pollution_model.pkl")
feature_columns = joblib.load("model_columns.pkl")

# 🎨 Streamlit UI Setup
st.set_page_config(page_title="Water Pollution Forecast", layout="centered")
st.markdown(
    """
    <style>
    .main { font-family: 'Segoe UI', sans-serif; }
    .title { color: #2C3E50; text-align: center; }
    .footer { margin-top: 30px; font-size: 12px; color: gray; text-align: center; }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 class='title'>🌊 Water Pollutants Forecast System</h1>", unsafe_allow_html=True)
st.markdown("Enter the year and station ID to estimate key water pollutants.", unsafe_allow_html=True)
st.write("---")

# 📝 Input from user
selected_year = st.number_input("📅 Select Year", min_value=2000, max_value=2100, value=2025)
input_station = st.text_input("🏷️ Enter Station ID", value='1')

# 🚀 Predict button
if st.button("🔍 Predict Levels"):
    if not input_station.strip():
        st.warning("⚠️ Please enter a valid Station ID.")
    else:
        # Prepare input for model
        user_input_df = pd.DataFrame({'year': [selected_year], 'id': [input_station]})
        encoded_input = pd.get_dummies(user_input_df, columns=['id'])

        # Align with expected model features
        for col in feature_columns:
            if col not in encoded_input.columns:
                encoded_input[col] = 0
        encoded_input = encoded_input[feature_columns]

        # Get prediction
        try:
            result = regressor.predict(encoded_input)[0]
            pollutant_labels = ['Oxygen (O₂)', 'Nitrate (NO₃)', 'Nitrite (NO₂)', 'Sulfate (SO₄)', 'Phosphate (PO₄)', 'Chloride (Cl)']

            st.success(f"✅ Predicted pollutant levels for Station ID `{input_station}` in {selected_year}:")
            for label, value in zip(pollutant_labels, result):
                st.markdown(f"**{label}**: `{value:.2f}` mg/L")
        except Exception as e:
            st.error(f"🚫 Error during prediction: {str(e)}")

# Footer
st.markdown("<p class='footer'>Developed using Streamlit · Model by your team</p>", unsafe_allow_html=True)
