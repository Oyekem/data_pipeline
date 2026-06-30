import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.logger import logger

try:
    logger.info("Feature engineering module loaded")
except:
    pass

# =====================================================
# FEATURE ENGINEERING
# =====================================================
def create_features(df):
    df = df.copy()

    # LAG FEATURES
    df["lag1"] = df["price"].shift(1)
    df["lag2"] = df["price"].shift(2)
    df["lag3"] = df["price"].shift(3)

    # MOVING AVERAGES
    df["ma5"] = df["price"].rolling(5).mean()
    df["ma10"] = df["price"].rolling(10).mean()

    # EMA
    df["ema12"] = df["price"].ewm(span=12).mean()
    df["ema26"] = df["price"].ewm(span=26).mean()

    # MACD
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # BOLLINGER BANDS
    df["bb_mid"] = df["price"].rolling(20).mean()
    df["bb_std"] = df["price"].rolling(20).std()

    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    # RSI
    delta = df["price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MOMENTUM
    df["momentum"] = df["price"] - df["price"].shift(10)

    # VOLATILITY
    df["std5"] = df["price"].rolling(5).std()

    df = df.dropna()

    if len(df) < 20:
        return pd.DataFrame()


# =====================================================
# FEATURE LIST (SINGLE SOURCE OF TRUTH)
# =====================================================
FEATURES = [
    "lag1",
    "lag2",
    "lag3",
    "ma5",
    "ma10",
    "ema12",
    "ema26",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_lower",
    "rsi",
    "momentum",
    "std5"
]


# =====================================================
# BACKTEST
# =====================================================
def backtest(model, df):
    df = create_features(df)

    if len(df) < 50:
        return 0, 0

    split = int(len(df) * 0.8)

    test = df.iloc[split:]

    preds = model.predict(test[FEATURES])

    mae = mean_absolute_error(test["price"], preds)
    rmse = np.sqrt(mean_squared_error(test["price"], preds))

    return mae, rmse


# =====================================================
# PREDICTION HELPERS
# =====================================================
def predict_next(model, df):
    df = create_features(df)

    if df.empty:
        return None

    latest = df.iloc[-1]

    X_input = latest[FEATURES].to_numpy().reshape(1, -1)

    return model.predict(X_input)[0]