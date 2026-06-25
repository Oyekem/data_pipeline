
import streamlit as st
import joblib
import pandas as pd
import numpy as np

from sqlalchemy import text
from config import engine
from ml_model import create_features, backtest
from streamlit_autorefresh import st_autorefresh
from datetime import datetime




st.title("Crypto Price Prediction System")




# LIVE AUTO REFRESH
st_autorefresh(interval=30000, key="refresh")

# LIVE BADGE
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:10px;">
        <span style="color:green; font-weight:bold;">🟢 LIVE</span>
        <span style="font-size:12px;">Auto-refreshing every 30s</span>
    </div>
    """,
    unsafe_allow_html=True
)


df = df.tail(200)   # keeps dashboard fast + smooth


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




coins = sorted(df["coin"].dropna().unique())

selected_coin = st.selectbox(
    "Select Coin",
    coins
)

coin_df = (
    df[df["coin"] == selected_coin]
    .copy()
    .sort_values("created_at")
)

if coin_df.empty:
    st.warning("No data for selected coin")
    st.stop()


coin_df = coin_df.tail(150)




st.subheader("Baseline Model")

coin_df["price_change"] = coin_df["price"].diff()

avg_change = coin_df["price_change"].mean()

if np.isnan(avg_change):
    avg_change = 0

last_price = coin_df["price"].iloc[-1]
simple_pred = last_price + avg_change

col1, col2 = st.columns(2)

col1.metric(
    "Current Price",
    round(last_price, 2)
)

col2.metric(
    "Simple Prediction",
    round(simple_pred, 2)
)





ml_df = create_features(coin_df)

if ml_df.empty:
    st.warning("Not enough data for ML prediction")
    st.stop()

features = [
    "lag1",
    "lag2",
    "lag3",
    "ma5",
    "std5"
]



coin = selected_coin.lower()


try:
    rf_model = joblib.load(
        f"models/{coin}_rf.pkl"
    )

    xgb_model = joblib.load(
        f"models/{coin}_xgb.pkl"
    )

except FileNotFoundError:
    st.error(
        "Model files missing. Run train_model.py first."
    )
    st.stop()







latest = ml_df.iloc[-1]

input_data = np.array([[
    latest["lag1"],
    latest["lag2"],
    latest["lag3"],
    latest["ma5"],
    latest["std5"]
]])





rf_pred = rf_model.predict(input_data)[0]
xgb_pred = xgb_model.predict(input_data)[0]

# use real performance (BEST PRACTICE)
rf_mae, _ = backtest(rf_model, coin_df)
xgb_mae, _ = backtest(xgb_model, coin_df)

rf_weight = 1 / (rf_mae + 1e-6)
xgb_weight = 1 / (xgb_mae + 1e-6)

total = rf_weight + xgb_weight
rf_weight /= total
xgb_weight /= total

avg_pred = (rf_pred * rf_weight) + (xgb_pred * xgb_weight)




st.subheader("Ensemble Prediction (Weighted)")

col1, col2, col3 = st.columns(3)

col1.metric("Random Forest", round(rf_pred, 2))
col2.metric("XGBoost", round(xgb_pred, 2))
col3.metric("Final Ensemble", round(avg_pred, 2))

st.write(f"RF weight: {rf_weight:.2f}")
st.write(f"XGB weight: {xgb_weight:.2f}")





st.subheader("Feature Importance")

fi_df = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_
})

fi_df = fi_df.sort_values(
    "importance",
    ascending=True
)

st.bar_chart(
    fi_df.set_index("feature")
)






def generate_signal(
    current_price,
    predicted_price
):
    diff = predicted_price - current_price

    if diff > current_price * 0.002:
        return "🟢 BUY"

    elif diff < -current_price * 0.002:
        return "🔴 SELL"

    return "🟡 HOLD"

signal = generate_signal(last_price, avg_pred)
confidence = abs(rf_pred - xgb_pred) / last_price

st.metric("Confidence Spread", round(confidence, 4))






st.subheader("Trading Signal")

col1, col2 = st.columns(2)

col1.metric(
    "Signal",
    signal
)

col2.metric(
    "Predicted Price",
    round(avg_pred, 2)
)




st.caption(
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)




st.subheader(
    "Prediction Confidence Band"
)

volatility = (
    coin_df["price"]
    .pct_change()
    .std()
)

upper = avg_pred * (1 + volatility)
lower = avg_pred * (1 - volatility)

band_df = pd.DataFrame({
    "Price": [
        lower,
        avg_pred,
        upper
    ]
},
index=[
    "Lower",
    "Prediction",
    "Upper"
])

st.line_chart(band_df)







st.subheader("Model Comparison")

compare_df = pd.DataFrame({
    "Model": [
        "Baseline",
        "Random Forest",
        "XGBoost",
        "Ensemble"
    ],
    "Prediction": [
        simple_pred,
        rf_pred,
        xgb_pred,
        avg_pred
    ]
})

st.dataframe(compare_df)








mae, rmse = backtest(
    rf_model,
    coin_df
)

st.subheader(
    "Model Performance"
)

col1, col2 = st.columns(2)

col1.metric(
    "MAE",
    round(mae, 4)
)

col2.metric(
    "RMSE",
    round(rmse, 4)
)






with engine.begin() as conn:

    conn.execute(
        text("""
        INSERT INTO prediction_history
        (
            prediction_time,
            coin,
            predicted_price,
            actual_price,
            signal
        )
        VALUES
        (
            NOW(),
            :coin,
            :predicted_price,
            :actual_price,
            :signal
        )
        """),
        {
            "coin": selected_coin,
            "predicted_price": float(avg_pred),
            "actual_price": float(last_price),
            "signal": signal
        }
    )








st.subheader(
    "Prediction History"
)

history = pd.read_sql(
    """
    SELECT *
    FROM prediction_history
    ORDER BY prediction_time DESC
    LIMIT 50
    """,
    engine
)

history["error"] = (
    abs(
        history["predicted_price"]
        - history["actual_price"]
    )
)

st.dataframe(history)

st.metric(
    "Average Prediction Error",
    round(
        history["error"].mean(),
        2
    )
)






st.subheader("Insight")

if avg_pred > simple_pred:
    st.success(
        "Ensemble model is more bullish than baseline"
    )
else:
    st.warning(
        "Ensemble model is more conservative than baseline"
    )

st.caption(
    f"Last updated: {coin_df['created_at'].max()}"
)