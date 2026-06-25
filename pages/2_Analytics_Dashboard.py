import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import engine
from streamlit_autorefresh import st_autorefresh

# -------------------------
# LIVE REFRESH
# -------------------------
st_autorefresh(interval=30000, key="refresh")

st.title("Crypto Trading Intelligence System")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data(ttl=30)
def load_data():
    query = "SELECT * FROM crypto_prices ORDER BY created_at DESC"
    return pd.read_sql(query, engine)

@st.cache_data(ttl=30)
def load_logs():
    return pd.read_sql("SELECT * FROM pipeline_runs ORDER BY run_time DESC", engine)

df = load_data()
logs = load_logs()

if df.empty:
    st.warning("No data yet")
    st.stop()

df["created_at"] = pd.to_datetime(df["created_at"])
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# -------------------------
# PIPELINE HEALTH
# -------------------------
st.subheader("⚙️ Pipeline Health")

if not logs.empty:
    latest_log = logs.iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", latest_log["status"])
    col2.metric("Runtime", round(latest_log["runtime_seconds"], 2))
    col3.metric("Runs", len(logs))

    success_rate = (logs["status"] == "success").mean() * 100
    st.metric("Success Rate", f"{success_rate:.2f}%")
else:
    st.warning("No logs yet")

# -------------------------
# COIN SELECTOR
# -------------------------
coins = df["coin"].dropna().unique()
selected_coin = st.selectbox("Select Coin", coins)

coin_df = df[df["coin"] == selected_coin].copy()
coin_df = coin_df.sort_values("created_at")

if coin_df.empty:
    st.warning("No data for selected coin")
    st.stop()

# -------------------------
# METRICS
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
# MOVING AVERAGE + EMA
# -------------------------
coin_df["ma7"] = coin_df["price"].rolling(7).mean()
coin_df["ema7"] = coin_df["price"].ewm(span=7).mean()

st.line_chart(
    coin_df.set_index("created_at")[["price", "ma7", "ema7"]]
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
    st.line_chart(normalized)

# -------------------------
# FEATURE ENGINEERING
# -------------------------
ml = coin_df.copy()

ml["lag1"] = ml["price"].shift(1)
ml["lag2"] = ml["price"].shift(2)
ml["lag3"] = ml["price"].shift(3)
ml["ma5"] = ml["price"].rolling(5).mean()
ml["std5"] = ml["price"].rolling(5).std()

ml = ml.dropna()

next_pred = None
model = None

features = ["lag1", "lag2", "lag3", "ma5", "std5"]

# -------------------------
# ML MODEL + BACKTEST
# -------------------------
st.subheader("🤖 ML Prediction + Backtest")

if len(ml) > 50:

    split = int(len(ml) * 0.8)

    train = ml.iloc[:split]
    test = ml.iloc[split:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train[features], train["price"])

    preds = model.predict(test[features])

    mae = mean_absolute_error(test["price"], preds)
    rmse = np.sqrt(mean_squared_error(test["price"], preds))

    st.metric("MAE", round(mae, 4))
    st.metric("RMSE", round(rmse, 4))

    # next prediction
    latest = ml.iloc[-1]
    next_pred = model.predict([[latest[f] for f in features]])[0]

    st.metric("Next Price Prediction", round(next_pred, 2))

else:
    st.warning("Not enough data for ML model")

# -------------------------
# SIGNAL ENGINE (SINGLE SOURCE OF TRUTH)
# -------------------------
def generate_signal(current_price, predicted_price):
    diff = predicted_price - current_price
    threshold = current_price * 0.002

    if diff > threshold:
        return "🟢 BUY"
    elif diff < -threshold:
        return "🔴 SELL"
    else:
        return "🟡 HOLD"

# -------------------------
# SIGNAL DISPLAY
# -------------------------
st.subheader("Trading Signal Engine")

if next_pred is not None:
    signal = generate_signal(last_price, next_pred)

    col1, col2 = st.columns(2)
    col1.metric("Signal", signal)
    col2.metric("Predicted Price", round(next_pred, 2))
else:
    st.info("Signal unavailable (not enough training data)")

# -------------------------
# RAW DATA
# -------------------------
st.subheader("Latest Data")
st.dataframe(df.head(20))