import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("ids_model.pkl")

st.title("🔐 AI-Powered Intrusion Detection System (IDS)")
st.write("Enter 41 comma-separated numeric features to check for attack detection:")

user_input = st.text_input("Input Features (Comma Separated)")

if st.button("Detect"):
    try:
        values = np.array([float(x) for x in user_input.split(',')]).reshape(1, -1)
        prediction = model.predict(values)
        if prediction[0] == "normal":
            st.success("✅ Result: Normal Traffic")
        else:
            st.error("🚨 Alert: Attack Detected!")
    except Exception as e:
        st.warning("⚠️ Please enter exactly 41 valid numbers.")
