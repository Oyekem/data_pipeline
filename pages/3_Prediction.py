import streamlit as st
import joblib
import pandas as pd
import numpy as np

from sqlalchemy import text
from config import engine
from ml_model import create_features, backtest
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# -------------------------
# TITLE
# -------------------------
st.title("Crypto Price Prediction System")

# -------------------------
# LIVE AUTO REFRESH
# -------------------------
st_autorefresh(interval=30000, key="refresh")

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:10px;">
        <span style="color:green; font-weight:bold;">🟢 LIVE</span>
        <span style="font-size:12px;">Auto-refreshing every 30s</span>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data(ttl=30)
def load_data():
    return pd.read_sql(
        "SELECT * FROM crypto_prices ORDER BY created_at DESC",
        engine
    )

df = load_data()

if df.empty:
    st.warning("No data available")
    st.stop()

df["created_at"] = pd.to_datetime(df["created_at"])
df["price"] = pd.to_numeric(df["price"], errors="coerce")

df = df.sort_values("created_at").tail(200)

# -------------------------
# COIN SELECTOR
# -------------------------
coins = sorted(df["coin"].dropna().unique())

selected_coin = st.selectbox("Select Coin", coins)

coin_df = df[df["coin"] == selected_coin].copy()
coin_df = coin_df.sort_values("created_at").tail(150)

if coin_df.empty:
    st.warning("No data for selected coin")
    st.stop()

# -------------------------
# BASELINE MODEL
# -------------------------
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

# -------------------------
# FEATURE ENGINEERING (SINGLE SOURCE OF TRUTH)
# -------------------------
ml_df = create_features(coin_df)

if ml_df.empty:
    st.warning("Not enough data for ML prediction")
    st.stop()

# -------------------------
# LOAD MODELS
# -------------------------
coin = selected_coin.lower()

try:
    rf_model = joblib.load(f"models/{coin}_rf.pkl")
    xgb_model = joblib.load(f"models/{coin}_xgb.pkl")

except FileNotFoundError:
    st.error("Model files missing. Run train_model.py first.")
    st.stop()

# -------------------------
# SAFE FEATURE SELECTION (CRITICAL FIX)
# -------------------------
model_features = rf_model.feature_names_in_.tolist()

missing = [f for f in model_features if f not in ml_df.columns]

if missing:
    st.error(f"Missing features in dataset: {missing}")
    st.stop()

latest = ml_df.iloc[-1]

input_data = latest[model_features].values.reshape(1, -1)

# -------------------------
# PREDICTIONS
# -------------------------
rf_pred = rf_model.predict(input_data)[0]
xgb_pred = xgb_model.predict(input_data)[0]

# -------------------------
# BACKTEST
# -------------------------
rf_mae, _ = backtest(rf_model, coin_df)
xgb_mae, _ = backtest(xgb_model, coin_df)

rf_weight = 1 / (rf_mae + 1e-6)
xgb_weight = 1 / (xgb_mae + 1e-6)

total = rf_weight + xgb_weight
rf_weight /= total
xgb_weight /= total

avg_pred = (rf_pred * rf_weight) + (xgb_pred * xgb_weight)

# -------------------------
# SIGNAL ENGINE
# -------------------------
def generate_signal(current_price, predicted_price):
    diff = predicted_price - current_price

    if diff > current_price * 0.002:
        return "🟢 BUY"
    elif diff < -current_price * 0.002:
        return "🔴 SELL"
    return "🟡 HOLD"

signal = generate_signal(last_price, avg_pred)

confidence = abs(rf_pred - xgb_pred) / (last_price + 1e-6)

st.metric("Confidence Spread", round(confidence, 4))

# -------------------------
# OUTPUT
# -------------------------
st.subheader("Trading Signal")

col1, col2 = st.columns(2)
col1.metric("Signal", signal)
col2.metric("Predicted Price", round(avg_pred, 2))

st.caption(
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

# -------------------------
# CONFIDENCE BAND
# -------------------------
volatility = coin_df["price"].pct_change().std()

upper = avg_pred * (1 + volatility)
lower = avg_pred * (1 - volatility)

band_df = pd.DataFrame(
    {"Price": [lower, avg_pred, upper]},
    index=["Lower", "Prediction", "Upper"]
)

st.subheader("Prediction Confidence Band")
st.line_chart(band_df)

# -------------------------
# MODEL COMPARISON
# -------------------------
compare_df = pd.DataFrame({
    "Model": ["Baseline", "Random Forest", "XGBoost", "Ensemble"],
    "Prediction": [simple_pred, rf_pred, xgb_pred, avg_pred]
})

st.subheader("Model Comparison")
st.dataframe(compare_df)

# -------------------------
# PERFORMANCE
# -------------------------
mae, rmse = backtest(rf_model, coin_df)

st.subheader("Model Performance")

col1, col2 = st.columns(2)
col1.metric("MAE", round(mae, 4))
col2.metric("RMSE", round(rmse, 4))

# -------------------------
# SAVE PREDICTIONS
# -------------------------
with engine.begin() as conn:
    conn.execute(
        text("""
        INSERT INTO prediction_history
        (prediction_time, coin, predicted_price, actual_price, signal)
        VALUES (NOW(), :coin, :predicted_price, :actual_price, :signal)
        """),
        {
            "coin": selected_coin,
            "predicted_price": float(avg_pred),
            "actual_price": float(last_price),
            "signal": signal
        }
    )

# -------------------------
# HISTORY
# -------------------------
history = pd.read_sql(
    """
    SELECT *
    FROM prediction_history
    ORDER BY prediction_time DESC
    LIMIT 50
    """,
    engine
)

history["error"] = abs(history["predicted_price"] - history["actual_price"])

st.subheader("Prediction History")
st.dataframe(history)

st.metric("Average Prediction Error", round(history["error"].mean(), 2))

# -------------------------
# INSIGHT
# -------------------------
st.subheader("Insight")

if avg_pred > simple_pred:
    st.success("Ensemble model is more bullish than baseline")
else:
    st.warning("Ensemble model is more conservative than baseline")

st.caption(f"Last updated: {coin_df['created_at'].max()}")