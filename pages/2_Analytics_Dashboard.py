import streamlit as st
import pandas as pd
from config import engine

# -------------------------
# AUTO REFRESH (SAFE)
# -------------------------
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=30000, key="refresh")


# -------------------------
# TITLE
# -------------------------
st.title("Crypto Analytics Dashboard")


# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data(ttl=30)
def load_data():
    query = "SELECT * FROM crypto_prices ORDER BY created_at DESC"
    return pd.read_sql(query, engine)

df = load_data()


# -------------------------
# SAFETY CHECK
# -------------------------
if df.empty:
    st.warning("No data available yet. Waiting for pipeline...")
    st.stop()


# -------------------------
# CLEANING
# -------------------------
df["created_at"] = pd.to_datetime(df["created_at"])
df["price"] = pd.to_numeric(df["price"], errors="coerce")


# -------------------------
# GLOBAL METRICS
# -------------------------
latest_time = df["created_at"].max()

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Latest Update", latest_time.strftime("%Y-%m-%d %H:%M:%S"))
col3.metric("Coins Tracked", df["coin"].nunique())


# -------------------------
# COIN SELECTOR
# -------------------------
coins = df["coin"].dropna().unique()
selected_coin = st.selectbox("Select Coin", coins)


# -------------------------
# FILTER DATA
# -------------------------
coin_df = df[df["coin"] == selected_coin].copy()
coin_df = coin_df.sort_values("created_at")


# -------------------------
# SAFETY FOR EMPTY COIN DATA
# -------------------------
if coin_df.empty:
    st.warning("No data for selected coin")
    st.stop()


# -------------------------
# PRICE CHART
# -------------------------
st.subheader(f"{selected_coin} Price Trend")

st.line_chart(
    coin_df.set_index("created_at")["price"]
)


# -------------------------
# MOVING AVERAGE
# -------------------------
coin_df["ma7"] = coin_df["price"].rolling(window=7).mean()

st.subheader("Moving Average (7-period)")

st.line_chart(
    coin_df.set_index("created_at")[["price", "ma7"]]
)


# -------------------------
# EMA
# -------------------------
coin_df["ema"] = coin_df["price"].ewm(span=7).mean()


# -------------------------
# VOLATILITY
# -------------------------
coin_df["returns"] = coin_df["price"].pct_change()
volatility = coin_df["returns"].std()


# -------------------------
# MARKET HEAT
# -------------------------
if pd.isna(volatility):
    heat = "N/A"
elif volatility < 0.01:
    heat = "Low"
elif volatility < 0.03:
    heat = "Medium"
else:
    heat = "High"


# -------------------------
# MARKET OVERVIEW
# -------------------------
st.subheader("Market Overview")

last_price = coin_df["price"].iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric("Price", round(last_price, 2))
col2.metric("Volatility", round(volatility, 6) if not pd.isna(volatility) else 0)
col3.metric("Market Heat", heat)


# -------------------------
# PREDICTION MODELS
# -------------------------
coin_df["rolling_mean"] = coin_df["price"].rolling(window=5).mean()

pred_rolling = coin_df["rolling_mean"].iloc[-1]
pred_ema = coin_df["ema"].iloc[-1]

std = coin_df["price"].std()

upper = last_price + std if not pd.isna(std) else last_price
lower = last_price - std if not pd.isna(std) else last_price


st.subheader("Prediction Models")

col1, col2 = st.columns(2)

col1.metric(
    "Rolling Forecast",
    round(pred_rolling, 2) if not pd.isna(pred_rolling) else "N/A"
)

col2.metric(
    "EMA Forecast",
    round(pred_ema, 2)
)

st.write(f"Confidence Range: {lower:.2f} - {upper:.2f}")

# -------------------------
# MULTI-COIN COMPARISON (USER SELECTABLE)
# -------------------------
st.subheader("Multi-Coin Comparison")

# Get available coins
all_coins = df["coin"].dropna().unique()

# User selects coins
selected_coins = st.multiselect(
    "Select coins to compare",
    options=all_coins,
    default=list(all_coins[:2])
)

# Safety check
if not selected_coins:
    st.warning("Please select at least one coin to compare")
    st.stop()

# Filter data
compare_df = df[df["coin"].isin(selected_coins)].copy()

# Ensure time order
compare_df["created_at"] = pd.to_datetime(compare_df["created_at"])
compare_df = compare_df.sort_values("created_at")

# Pivot
pivot = compare_df.pivot_table(
    index="created_at",
    columns="coin",
    values="price"
)

# Clean missing values
pivot = pivot.dropna(how="all")

# -------------------------
# NORMALIZATION (SAFE VERSION)
# -------------------------
# Avoid crash if first row has NaN
pivot_clean = pivot.dropna()

if not pivot_clean.empty:
    normalized = pivot_clean / pivot_clean.iloc[0] * 100

    st.caption("Normalized performance (starting at 100)")
    st.line_chart(normalized)
else:
    st.warning("Not enough data to normalize comparison")


# -------------------------
# PIPELINE HEALTH
# -------------------------
st.subheader("Pipeline Health Monitor")

log_query = """
SELECT * FROM pipeline_runs
ORDER BY run_time DESC
LIMIT 10
"""

logs = pd.read_sql(log_query, engine)

if logs.empty:
    st.warning("No pipeline logs yet")
else:
    st.dataframe(logs)

    latest = logs.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Last Status", latest["status"])
    col2.metric("Last Runtime (s)", round(latest["runtime_seconds"], 2))
    col3.metric("Last Run Time", latest["run_time"])


# -------------------------
# RAW DATA
# -------------------------
st.subheader("Latest Data")

st.dataframe(df.head(20))