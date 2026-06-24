import streamlit as st
import pandas as pd
import numpy as np
from config import engine

st.title("Price Prediction (Next 5 Minutes)")

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data(ttl=30)
def load_data():
    query = "SELECT * FROM crypto_prices ORDER BY created_at DESC"
    return pd.read_sql(query, engine)

df = load_data()

# -----------------------
# FILTER BITCOIN (SAFE)
# -----------------------
btc = df[df["coin"].str.lower() == "bitcoin"].copy()

# -----------------------
# SAFETY CHECK (IMPORTANT)
# -----------------------
if btc.empty:
    st.warning("No Bitcoin data available yet.")
    st.stop()

btc = btc.sort_values("created_at")

# -----------------------
# SIMPLE FORECAST MODEL
# -----------------------
btc["price_change"] = btc["price"].diff()

avg_change = btc["price_change"].mean()

last_price = btc["price"].iloc[-1]

# fallback safety if NaN
if np.isnan(avg_change):
    avg_change = 0

predicted_price = last_price + avg_change

# -----------------------
# OUTPUT
# -----------------------
st.subheader("Prediction Result")

st.metric("Current Price", round(last_price, 2))
st.metric("Predicted Price (5 min)", round(predicted_price, 2))

st.write("""
This is a simple statistical forecast using average price movement.

Later upgrades:
- Moving Average Model
- Prophet forecasting
- LSTM neural network
- Real-time streaming prediction
""")