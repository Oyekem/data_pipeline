import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "models/rf_model.pkl"

# -----------------------
# BACKTEST
# -----------------------

def backtest(model, df):
    df = create_features(df)

    if len(df) < 50:
        return 0, 0

    features = ["lag1", "lag2", "lag3", "ma5", "std5"]

    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    # ❌ DO NOT retrain model here
    preds = model.predict(test[features])

    mae = mean_absolute_error(test["price"], preds)
    rmse = np.sqrt(mean_squared_error(test["price"], preds))

    return mae, rmse



# -----------------------
# FEATURE ENGINEERING
# -----------------------
def create_features(df):
    df = df.copy()

# ------------------------- 
# Basic Lag Features 
# -------------------------

    df["lag1"] = df["price"].shift(1)
    df["lag2"] = df["price"].shift(2)
    df["lag3"] = df["price"].shift(3)


# ------------------------- 
# Moving Averages 
# -------------------------

    df["ma5"] = df["price"].rolling(5).mean()
    df["ma10"] = df["price"].rolling(10).mean()


# ------------------------- 
# EMA 
# -------------------------
    df["ema12"] = df["price"].ewm(span=12, adjust=False).mean() 
    df["ema26"] = df["price"].ewm(span=26, adjust=False).mean()




# ------------------------- 
# MACD 
# ------------------------- 
    df["macd"] = df["ema12"] - df["ema26"] 
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()



# ------------------------- 
# Bollinger Bands 
# -------------------------

    rolling_mean = df["price"].rolling(20).mean() 
    rolling_std = df["price"].rolling(20).std() 

    df["bb_upper"] = rolling_mean + (2 * rolling_std) 
    df["bb_lower"] = rolling_mean - (2 * rolling_std)



# ------------------------- 
# RSI 
# -------------------------


    delta = df["price"].diff() 

    gain = delta.clip(lower=0) 
    loss = -delta.clip(upper=0) 

    avg_gain = gain.rolling(14).mean() 
    avg_loss = loss.rolling(14).mean() 

    rs = avg_gain / (avg_loss + 1e-9) 

    df["rsi"] = 100 - (100 / (1 + rs))


# ------------------------- 
# Momentum 
# -------------------------
    
    df["momentum"] = df["price"] - df["price"].shift(10)



# ------------------------- 
# Volatility 
# -------------------------

    df["std5"] = df["price"].rolling(5).std()


    df = df.dropna()
    return df


# -----------------------
# TRAIN MODEL
# -----------------------
def train_model(df):
    df = create_features(df)

    X = df[["lag1", "lag2", "lag3", "ma5", "std5"]]
    y = df["price"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model


# -----------------------
# LOAD MODEL
# -----------------------
def load_model():
    return joblib.load(MODEL_PATH)


# -----------------------
# PREDICT
# -----------------------
def predict_next(model, df):
    df = create_features(df)

    if len(df) == 0:
        return None

    latest = df.iloc[-1]

    X_input = np.array([[ 
        latest["lag1"],
        latest["lag2"],
        latest["lag3"],
        latest["ma5"],
        latest["std5"]
    ]])

    return model.predict(X_input)[0]
