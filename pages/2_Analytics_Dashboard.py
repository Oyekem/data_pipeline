import streamlit as st
import pandas as pd
import numpy as np

from config import engine
from streamlit_autorefresh import st_autorefresh
from ml_model import create_features

# -------------------------
# LIVE REFRESH SYSTEM
# -------------------------
st_autorefresh(interval=30000, key="refresh")

st.title("Crypto Trading Intelligence System")
st.markdown("LIVE Dashboard (Auto-refresh every 30s)")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data(ttl=30)
def load_data():
    query = "SELECT * FROM crypto_prices ORDER BY created_at ASC"
    return pd.read_sql(query, engine)

df = load_data()

if df.empty:
    st.warning("No data yet")
    st.stop()

df["created_at"] = pd.to_datetime(df["created_at"])
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# =====================================================
# DATE RANGE FILTER (ADDED HERE)
# =====================================================
min_date = df["created_at"].min().date()
max_date = df["created_at"].max().date()

start_date, end_date = st.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

df = df[
    (df["created_at"].dt.date >= start_date) &
    (df["created_at"].dt.date <= end_date)
]

if df.empty:
    st.warning("No data in selected date range")
    st.stop()

# -------------------------
# COIN SELECTION
# -------------------------
coins = df["coin"].dropna().unique()
selected_coin = st.selectbox("Select Coin", coins)

coin_df = df[df["coin"] == selected_coin].copy()
coin_df = coin_df.sort_values("created_at")

if coin_df.empty:
    st.warning("No data for selected coin")
    st.stop()

# -------------------------
# FEATURE ENGINEERING
# -------------------------
coin_df = create_features(coin_df)

coin_df["ma7"] = coin_df["price"].rolling(7).mean()
coin_df["ema7"] = coin_df["price"].ewm(span=7).mean()

coin_df = coin_df.dropna()

# -------------------------
# CORE METRICS
# -------------------------
last_price = coin_df["price"].iloc[-1]

coin_df["returns"] = coin_df["price"].pct_change()
volatility = coin_df["returns"].std()

heat = (
    "🟢 Low" if volatility < 0.01 else
    "🟡 Medium" if volatility < 0.03 else
    "🔴 High"
)

col1, col2, col3 = st.columns(3)
col1.metric("Price", round(last_price, 2))
col2.metric("Volatility", round(volatility, 6))
col3.metric("Market Heat", heat)

# -------------------------
# PRICE TREND
# -------------------------
st.subheader("Price Trend")
st.line_chart(
    coin_df.set_index("created_at")[["price", "ma7", "ema7"]].tail(100)
)

# -------------------------
# RSI (SAFE)
# -------------------------
st.subheader("RSI (14)")
if "rsi" in coin_df.columns:
    st.line_chart(coin_df.set_index("created_at")["rsi"].tail(100))
else:
    st.warning("RSI not available")

# -------------------------
# MACD (SAFE)
# -------------------------
st.subheader("MACD")
if "macd" in coin_df.columns and "macd_signal" in coin_df.columns:
    st.line_chart(
        coin_df.set_index("created_at")[["macd", "macd_signal"]].tail(100)
    )
else:
    st.warning("MACD not available")

# -------------------------
# BOLLINGER
# -------------------------
st.subheader("Bollinger Bands")
if all(x in coin_df.columns for x in ["bb_upper", "bb_lower"]):
    st.line_chart(
        coin_df.set_index("created_at")[["price", "bb_upper", "bb_lower"]].tail(100)
    )
else:
    st.warning("Bollinger Bands not available")

# -------------------------
# EMA TREND
# -------------------------
st.subheader("EMA Trend")
if "ema12" in coin_df.columns and "ema26" in coin_df.columns:
    st.line_chart(
        coin_df.set_index("created_at")[["price", "ema12", "ema26"]].tail(100)
    )

# -------------------------
# MULTI-COIN COMPARISON
# -------------------------
selected_coins = st.multiselect(
    "Compare coins",
    coins,
    default=list(coins[:2])
)

compare_df = df[df["coin"].isin(selected_coins)]

pivot = compare_df.pivot_table(
    index="created_at",
    columns="coin",
    values="price"
).dropna(how="all")

if not pivot.empty:
    normalized = pivot / pivot.iloc[0] * 100
    st.subheader("Multi-Coin Performance (Normalized)")
    st.line_chart(normalized.tail(200))

# -------------------------
# PREDICTION LAYER
# -------------------------
st.subheader(" Prediction Quality Layer")

simple_pred = last_price + coin_df["price"].diff().mean()

def multi_step_forecast(series, steps=5):
    preds = []
    data = series.copy()

    for _ in range(steps):
        diff = data.diff().mean()
        next_val = data.iloc[-1] + diff
        preds.append(next_val)
        data = pd.concat([data, pd.Series([next_val])], ignore_index=True)

    return preds

forecast_5 = multi_step_forecast(coin_df["price"], 5)

residuals = coin_df["price"].diff()
std_error = residuals.std()

upper = simple_pred + (1.96 * std_error)
lower = simple_pred - (1.96 * std_error)

confidence_score = 1 / (std_error + 1e-6)
confidence_score = min(confidence_score / 100, 1)

# -------------------------
# DISPLAY
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Next Price", round(simple_pred, 2))
col2.metric("Confidence", round(confidence_score, 3))
col3.metric("Volatility", heat)

st.write(f"Confidence Interval: {lower:.2f} - {upper:.2f}")

st.subheader("Forecast (5-step)")
st.line_chart(pd.DataFrame(forecast_5, columns=["Forecast"]))

# -------------------------
# RAW DATA
# -------------------------
st.subheader("Latest Data")
st.dataframe(df.tail(20))