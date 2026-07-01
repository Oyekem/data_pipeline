import streamlit as st
import pandas as pd
import numpy as np

from utils.logger import logger
from config import engine
from streamlit_autorefresh import st_autorefresh
from ml_model import create_features

logger.info("Dashboard started")

# -------------------------
# AUTO REFRESH
# -------------------------
st_autorefresh(interval=30000, key="refresh")

st.title("📊 Crypto Trading Intelligence System")
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
    st.warning("No data available")
    st.stop()

df["created_at"] = pd.to_datetime(df["created_at"])
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# -------------------------
# DATE FILTER
# -------------------------
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
    st.warning("No data in selected range")
    st.stop()

# -------------------------
# COIN FILTER
# -------------------------
coins = df["coin"].dropna().unique()
selected_coin = st.selectbox("Select Coin", coins)

coin_df = df[df["coin"] == selected_coin].copy()
coin_df = coin_df.sort_values("created_at")

if coin_df.empty:
    st.warning("No data for selected coin")
    st.stop()

# -------------------------
# FEATURES
# -------------------------
coin_df = create_features(coin_df)

coin_df["ma7"] = coin_df["price"].rolling(7).mean()
coin_df["ema7"] = coin_df["price"].ewm(span=7).mean()
coin_df = coin_df.dropna()

# -------------------------
# METRICS
# -------------------------
last_price = coin_df["price"].iloc[-1]

volatility = coin_df["price"].pct_change().std()

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
# CHARTS
# -------------------------
st.subheader("Price Trend")
st.line_chart(coin_df.set_index("created_at")[["price", "ma7", "ema7"]].tail(100))

st.subheader("RSI")
if "rsi" in coin_df.columns:
    st.line_chart(coin_df.set_index("created_at")["rsi"].tail(100))

st.subheader("MACD")
if {"macd", "macd_signal"}.issubset(coin_df.columns):
    st.line_chart(coin_df.set_index("created_at")[["macd", "macd_signal"]].tail(100))

st.subheader("Bollinger Bands")
if {"bb_upper", "bb_lower"}.issubset(coin_df.columns):
    st.line_chart(coin_df.set_index("created_at")[["price", "bb_upper", "bb_lower"]].tail(100))

st.subheader("EMA Trend")
if {"ema12", "ema26"}.issubset(coin_df.columns):
    st.line_chart(coin_df.set_index("created_at")[["price", "ema12", "ema26"]].tail(100))

# -------------------------
# MULTI-COIN
# -------------------------
selected_coins = st.multiselect("Compare coins", coins, default=list(coins[:2]))

compare_df = df[df["coin"].isin(selected_coins)]

pivot = compare_df.pivot_table(
    index="created_at",
    columns="coin",
    values="price"
).dropna(how="all")

if not pivot.empty:
    normalized = pivot / pivot.iloc[0] * 100
    st.subheader("Multi-Coin Performance")
    st.line_chart(normalized.tail(200))

# -------------------------
# RAW DATA
# -------------------------
st.subheader("Latest Data")
st.dataframe(df.tail(20))