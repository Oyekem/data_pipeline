import streamlit as st
import pandas as pd
import numpy as np
from config import engine
from sklearn.ensemble import RandomForestRegressor

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

if btc.empty:
    st.warning("No Bitcoin data available yet.")
    st.stop()

btc = btc.sort_values("created_at")


# =========================================================
# MODEL 1: SIMPLE STATISTICAL MODEL
# =========================================================
st.subheader("Baseline Model (Simple Trend)")

btc["price_change"] = btc["price"].diff()
avg_change = btc["price_change"].mean()

if np.isnan(avg_change):
    avg_change = 0

last_price = btc["price"].iloc[-1]
simple_pred = last_price + avg_change

col1, col2 = st.columns(2)

col1.metric("Current Price", round(last_price, 2))
col2.metric("Simple Prediction", round(simple_pred, 2))


# =========================================================
# MODEL 2: MACHINE LEARNING MODEL
# =========================================================
st.subheader("ML Model (Random Forest Forecast)")


# -----------------------
# FEATURE ENGINEERING
# -----------------------
ml_df = btc.copy()

ml_df["lag1"] = ml_df["price"].shift(1)
ml_df["lag2"] = ml_df["price"].shift(2)
ml_df["lag3"] = ml_df["price"].shift(3)

ml_df["ma5"] = ml_df["price"].rolling(5).mean()
ml_df["std5"] = ml_df["price"].rolling(5).std()

ml_df = ml_df.dropna()


# -----------------------
# TRAIN MODEL
# -----------------------
X = ml_df[["lag1", "lag2", "lag3", "ma5", "std5"]]
y = ml_df["price"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)


# -----------------------
# PREDICT
# -----------------------
latest = ml_df.iloc[-1]

input_data = np.array([[
    latest["lag1"],
    latest["lag2"],
    latest["lag3"],
    latest["ma5"],
    latest["std5"]
]])

ml_pred = model.predict(input_data)[0]


col1, col2 = st.columns(2)

col1.metric("ML Prediction", round(ml_pred, 2))
col2.metric("Model Type", "Random Forest")


# =========================================================
# SIGNAL ENGINE (COMBINED)
# =========================================================
st.subheader("Trading Signal Engine")

diff = ml_pred - last_price
pct = diff / last_price

if pct > 0.01:
    signal = "🟢 BUY"
elif pct < -0.01:
    signal = "🔴 SELL"
else:
    signal = "🟡 HOLD"

st.metric("Signal", signal)


# =========================================================
# CONFIDENCE RANGE
# =========================================================
volatility = btc["price"].pct_change().std()

upper = ml_pred * (1 + volatility)
lower = ml_pred * (1 - volatility)

st.write(f"Confidence Range: {lower:.2f} - {upper:.2f}")


# =========================================================
# MODEL COMPARISON
# =========================================================
st.subheader("Model Comparison")

compare_df = pd.DataFrame({
    "Model": ["Simple Average", "ML Model"],
    "Prediction": [simple_pred, ml_pred]
})

st.dataframe(compare_df)


# =========================================================
# QUICK INSIGHT
# =========================================================
st.subheader("Insight")

if ml_pred > simple_pred:
    st.success("ML model is more bullish than baseline model")
else:
    st.warning("ML model is more conservative than baseline")


st.caption(f"Last updated: {btc['created_at'].max()}")