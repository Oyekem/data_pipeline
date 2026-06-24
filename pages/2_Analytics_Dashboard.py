import streamlit as st
import pandas as pd
from config import engine

st.title("Crypto Analytics Dashboard")

@st.cache_data(ttl=30)
def load_data():
    query = "SELECT * FROM crypto_prices ORDER BY created_at DESC"
    return pd.read_sql(query, engine)

df = load_data()

# -----------------------
# LIVE METRICS
# -----------------------
st.subheader("Pipeline Health")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Latest Price Rows", df.shape[0])
col3.metric("Last Update", df["created_at"].max())

# -----------------------
# TABLE
# -----------------------
st.subheader("Live Data")
st.dataframe(df)

# -----------------------
# PRICE TREND
# -----------------------
st.subheader("Price Trend (Bitcoin Example)")

btc = df[df["coin"] == "bitcoin"]

st.line_chart(btc.set_index("created_at")["price"])