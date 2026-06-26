import os
import joblib
import pandas as pd
import numpy as np

from config import engine

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# -------------------
# SETUP
# -------------------
os.makedirs("models", exist_ok=True)

# -------------------
# LOAD DATA
# -------------------
df = pd.read_sql("SELECT * FROM crypto_prices", engine)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])

# -------------------
# FEATURES
# -------------------
def create_features(data):
    data = data.copy()

    data["lag1"] = data["price"].shift(1)
    data["lag2"] = data["price"].shift(2)
    data["lag3"] = data["price"].shift(3)

    data["ma5"] = data["price"].rolling(5).mean()
    data["ma10"] = data["price"].rolling(10).mean()

    data["ema12"] = data["price"].ewm(span=12).mean()
    data["ema26"] = data["price"].ewm(span=26).mean()

    data["macd"] = data["ema12"] - data["ema26"]
    data["macd_signal"] = data["macd"].ewm(span=9).mean()

    delta = data["price"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    data["rsi"] = 100 - (100 / (1 + rs))

    data["momentum"] = data["price"] - data["price"].shift(10)

    data["bb_mid"] = data["price"].rolling(20).mean()
    data["bb_std"] = data["price"].rolling(20).std()
    data["bb_upper"] = data["bb_mid"] + 2 * data["bb_std"]
    data["bb_lower"] = data["bb_mid"] - 2 * data["bb_std"]

    data["std5"] = data["price"].rolling(5).std()

    return data.dropna()

features = [
    "lag1","lag2","lag3",
    "ma5","ma10",
    "ema12","ema26",
    "macd","macd_signal",
    "bb_upper","bb_lower",
    "rsi","momentum","std5"
]

# -------------------
# MODELS
# -------------------
models = {
    "rf": RandomForestRegressor(),
    "xgb": XGBRegressor(),
    "lgbm": LGBMRegressor(),
    "cat": CatBoostRegressor(verbose=0)
}

# -------------------
# TRAIN PER COIN
# -------------------
coins = df["coin"].dropna().unique()

for coin in coins:

    print(f"\nTRAINING: {coin}")

    coin_df = df[df["coin"] == coin].copy()
    coin_df = create_features(coin_df)

    if len(coin_df) < 50:
        print("SKIPPED")
        continue

    X = coin_df[features]
    y = coin_df["price"]

    # -------------------
    # TIME SERIES CV
    # -------------------
    tscv = TimeSeriesSplit(n_splits=5)

    best_models = {}
    scores = {}

    # -------------------
    # TRAIN MODELS
    # -------------------
    for name, model in models.items():

        model.fit(X, y)

        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)

        best_models[name] = model
        scores[name] = mae

        joblib.dump(model, f"models/{coin.lower()}_{name}.pkl")

    # -------------------
    # ENSEMBLE WEIGHTS
    # -------------------
    inv_scores = {k: 1/(v+1e-6) for k, v in scores.items()}
    total = sum(inv_scores.values())
    weights = {k: v/total for k, v in inv_scores.items()}

    print("Weights:", weights)

    joblib.dump(weights, f"models/{coin.lower()}_weights.pkl")

print("\nTRAINING COMPLETE")