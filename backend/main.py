from fastapi import FastAPI
from config import engine
import pandas as pd


app = FastAPI(title = "Crypto Intelligence API")



@app.get("/latest")
def get_latest():
    df = pd.read_sql(
        "SELECT * FROM crypto_prices ORDER BY created_at DESC LIMIT 1",
        engine
    )
    return df.to_dict(orient="records")





@app.get("/coins")
def get_coins():
    df = pd.read_sql(
        "SELECT DISTINCT coin FROM crypto_prices",
        engine
    )
    return df["coin"].tolist()





@app.get("/history/{coin}")
def get_history(coin: str):
    df = pd.read_sql(
        f"""
        SELECT * FROM crypto_prices
        WHERE coin = '{coin}'
        ORDER BY created_at DESC
        LIMIT 200
        """,
        engine
    )
    return df.to_dict(orient="records")





@app.get("/metrics/{coin}")
def get_metrics(coin: str):
    df = pd.read_sql(
        f"""
        SELECT * FROM crypto_prices
        WHERE coin = '{coin}'
        """,
        engine
    )

    return {
        "rows": len(df),
        "latest_price": float(df["price"].iloc[-1])
    }






@app.get("/health")
def health():
    return {"status": "ok"}






import joblib
import numpy as np
from ml_model import create_features

rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")


@app.get("/predictions/{coin}")
def predict(coin: str):
    df = pd.read_sql(
        f"""
        SELECT * FROM crypto_prices
        WHERE coin = '{coin}'
        ORDER BY created_at
        """,
        engine
    )

    df = create_features(df)

    latest = df.iloc[-1]

    input_data = np.array([[
        latest["lag1"],
        latest["lag2"],
        latest["lag3"],
        latest["ma5"],
        latest["std5"]
    ]])

    rf_pred = rf_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_data)[0]

    final = (rf_pred + xgb_pred) / 2

    return {
        "coin": coin,
        "rf_prediction": float(rf_pred),
        "xgb_prediction": float(xgb_pred),
        "final_prediction": float(final)
    }







