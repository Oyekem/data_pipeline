import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from config import engine


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
    data["std5"] = data["price"].rolling(5).std()
    return data.dropna()


features = ["lag1", "lag2", "lag3", "ma5", "std5"]


# -------------------
# TRAIN (ALL COINS)
# -------------------
coins = df["coin"].dropna().unique()

for coin in coins:

    print("\n====================")
    print("COIN:", coin)
    print("====================")

    coin_df = df[df["coin"] == coin].copy()

    print("Rows before features:", len(coin_df))

    coin_df = create_features(coin_df)

    print("Rows after features:", len(coin_df))

    if len(coin_df) < 20:
        print("SKIPPED (not enough data)")
        continue

    print("TRAINING RF...")

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(coin_df[features], coin_df["price"])

    joblib.dump(rf, f"models/{coin.lower()}_rf.pkl")

    print("RF SAVED")

    print("TRAINING XGB...")

    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
    xgb.fit(coin_df[features], coin_df["price"])

    joblib.dump(xgb, f"models/{coin.lower()}_xgb.pkl")

    print("XGB SAVED")