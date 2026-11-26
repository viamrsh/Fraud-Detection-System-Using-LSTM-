import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================
# LOAD MODEL
# ============================
model = tf.keras.models.load_model("lstm_fraud_model.keras")

st.set_page_config(page_title="Fraud Detection (LSTM)", layout="centered")

st.title("💳 Fraud Transaction Detection")
st.write("Deep Learning (LSTM) based sequential fraud detection")

# ============================
# USER INPUT
# ============================
st.subheader("Enter last 10 transaction amounts")

tx_amounts = []
flags = []

for i in range(10):
    amt = st.number_input(f"Transaction {i+1} Amount", min_value=0.0, step=1.0)
    tx_amounts.append(amt)
    flags.append(1 if amt > 220 else 0)

# ============================
# PREDICTION
# ============================
if st.button("Detect Fraud"):
    scaler = StandardScaler()
    X = np.column_stack([tx_amounts, flags])
    X = scaler.fit_transform(X)
    X = X.reshape(1, 10, 2)

    prob = model.predict(X)[0][0]

    if prob > 0.5:
        st.error(f"🚨 FRAUD DETECTED\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ LEGITIMATE TRANSACTION\n\nProbability: {prob:.2f}")

    # ============================
    # VISUALIZATION
    # ============================
    st.subheader("Transaction Pattern")

    fig, ax = plt.subplots()
    ax.plot(tx_amounts, marker='o')
    ax.axhline(220, color='r', linestyle='--', label='Fraud Threshold')
    ax.set_xlabel("Transaction Index")
    ax.set_ylabel("Amount")
    ax.legend()

    st.pyplot(fig)
