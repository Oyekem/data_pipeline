import streamlit as st
import joblib
import pandas as pd
import numpy as np
from config import engine
from ml_model import create_features, backtest

st.title("Price Prediction (Next 5 Minutes)")

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data(ttl=30)
def load_data():
    query = "SELECT * FROM crypto_prices ORDER BY created_at DESC"
    return pd.read_sql(query, engine)

df = load_data()

if df.empty:
    st.warning("No data available")
    st.stop()

df["created_at"] = pd.to_datetime(df["created_at"])
df["price"] = pd.to_numeric(df["price"], errors="coerce")


# -----------------------
# COIN SELECTOR (FIXED)
# -----------------------
coins = sorted(df["coin"].dropna().unique())

selected_coin = st.selectbox("Select Coin", coins)

coin_df = df[df["coin"] == selected_coin].copy()
coin_df = coin_df.sort_values("created_at")

if coin_df.empty:
    st.warning("No data for selected coin")
    st.stop()

# -----------------------
# BASELINE MODEL
# -----------------------
st.subheader("Baseline Model")

coin_df["price_change"] = coin_df["price"].diff()
avg_change = coin_df["price_change"].mean()

if np.isnan(avg_change):
    avg_change = 0

last_price = coin_df["price"].iloc[-1]
simple_pred = last_price + avg_change

col1, col2 = st.columns(2)
col1.metric("Current Price", round(last_price, 2))
col2.metric("Simple Prediction", round(simple_pred, 2))


# -----------------------
# FEATURE ENGINEERING
# -----------------------
ml_df = create_features(coin_df)

if ml_df.empty:
    st.warning("Not enough data for ML model")
    st.stop()

features = ["lag1", "lag2", "lag3", "ma5", "std5"]


# -----------------------
# LOAD MODEL (ONLY SOURCE OF TRUTH)
# -----------------------
try:
    model = joblib.load("models/rf_model.pkl")
except FileNotFoundError:
    st.error("Model not found. Run train_model.py first.")
    st.stop()


# -----------------------
# PREDICTION
# -----------------------
latest = ml_df.iloc[-1]

input_data = np.array([[latest[f] for f in features]])

ml_pred = model.predict(input_data)[0]


st.subheader("ML Model Prediction")

col1, col2 = st.columns(2)
col1.metric("ML Prediction", round(ml_pred, 2))
col2.metric("Model", "Random Forest")


# -----------------------
# SIGNAL ENGINE (SINGLE SOURCE)
# -----------------------
def generate_signal(current_price, predicted_price):
    diff = predicted_price - current_price

    if diff > current_price * 0.002:
        return "🟢 BUY"
    elif diff < -current_price * 0.002:
        return "🔴 SELL"
    else:
        return "🟡 HOLD"


signal = generate_signal(last_price, ml_pred)

st.subheader("Trading Signal")

col1, col2 = st.columns(2)
col1.metric("Signal", signal)
col2.metric("Predicted Price", round(ml_pred, 2))


# -----------------------
# CONFIDENCE RANGE
# -----------------------
volatility = coin_df["price"].pct_change().std()

upper = ml_pred * (1 + volatility)
lower = ml_pred * (1 - volatility)

st.write(f"Confidence Range: {lower:.2f} - {upper:.2f}")


# -----------------------
# MODEL COMPARISON
# -----------------------
st.subheader("Model Comparison")

compare_df = pd.DataFrame({
    "Model": ["Baseline", "ML"],
    "Prediction": [simple_pred, ml_pred]
})

st.dataframe(compare_df)


# -----------------------
# BACKTEST (CLEAN)
# -----------------------
mae, rmse = backtest(model, coin_df)

st.subheader("Model Performance")

col1, col2 = st.columns(2)
col1.metric("MAE", round(mae, 4))
col2.metric("RMSE", round(rmse, 4))


# -----------------------
# INSIGHT
# -----------------------
st.subheader("Insight")

if ml_pred > simple_pred:
    st.success("ML model is more bullish than baseline")
else:
    st.warning("ML model is more conservative than baseline")


st.caption(f"Last updated: {coin_df['created_at'].max()}")