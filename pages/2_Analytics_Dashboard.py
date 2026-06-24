import streamlit as st
import pandas as pd
from config import engine

st.title("Crypto Analytics Dashboard")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data(ttl=30)
def load_data():
    query = "SELECT * FROM crypto_prices ORDER BY created_at DESC"
    df = pd.read_sql(query, engine)
    return df

df = load_data()

# -------------------------
# SAFETY CHECK
# -------------------------
if df.empty:
    st.warning("No data available yet. Waiting for pipeline...")
    st.stop()


# -------------------------
# BASIC CLEANING
# -------------------------
df["created_at"] = pd.to_datetime(df["created_at"])
df["price"] = pd.to_numeric(df["price"], errors="coerce")


# -------------------------
# METRICS SECTION
# -------------------------
latest_time = df["created_at"].max()

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Latest Update", latest_time.strftime("%Y-%m-%d %H:%M:%S"))
col3.metric("Coins Tracked", df["coin"].nunique())

# -------------------------
# FILTER BITCOIN
# -------------------------
btc = df[df["coin"].str.lower() == "bitcoin"].sort_values("created_at")

st.subheader("Bitcoin Price Trend")

st.line_chart(
    btc.set_index("created_at")["price"]
)

# -------------------------
# PIPELINE HEALTH
# -------------------------
st.subheader("Pipeline Health")

if len(df) > 0:
    st.success("Pipeline is running and receiving data")
else:
    st.error("No data detected")

# -------------------------
# PIPELINE HEALTH MONITOR
# -------------------------
st.subheader("⚙️ Pipeline Health Monitor")

log_query = """
SELECT * FROM pipeline_runs
ORDER BY run_time DESC
LIMIT 10
"""

logs = pd.read_sql(log_query, engine)

st.dataframe(logs)

if logs.empty:
    st.warning("No pipeline logs yet")
    st.stop()

latest = logs.iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric("Last Status", latest["status"])
col2.metric("Last Runtime (s)", round(latest["runtime_seconds"], 2))
col3.metric("Last Run Time", latest["run_time"])



# -------------------------
# RAW DATA TABLE
# -------------------------
st.subheader("📋 Latest Data")

st.dataframe(df.head(20))