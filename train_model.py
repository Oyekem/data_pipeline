```python
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import engine


# -------------------------
# CREATE MODELS FOLDER
# -------------------------
os.makedirs("models", exist_ok=True)


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_sql(
    "SELECT * FROM crypto_prices",
    engine
)

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

df = df.dropna(subset=["price"])


# -------------------------
# FEATURE ENGINEERING
# -------------------------
def create_features(data):

    data = data.copy()

    data["lag1"] = data["price"].shift(1)
    data["lag2"] = data["price"].shift(2)
    data["lag3"] = data["price"].shift(3)

    data["ma5"] = data["price"].rolling(5).mean()
    data["std5"] = data["price"].rolling(5).std()

    data = data.dropna()

    return data


# -------------------------
# TRAIN PER COIN
# -------------------------
coins = sorted(
    df["coin"].dropna().unique()
)

for coin in coins:

    print(f"\nTraining {coin}")

    coin_df = (
        df[df["coin"] == coin]
        .copy()
        .sort_values("created_at")
    )

    coin_df = create_features(
        coin_df
    )

    if len(coin_df) < 50:

        print(
            f"Skipped {coin} (not enough data)"
        )

        continue

    features = [
        "lag1",
        "lag2",
        "lag3",
        "ma5",
        "std5"
    ]

    X = coin_df[features]
    y = coin_df["price"]

    # -------------------------
    # RANDOM FOREST
    # -------------------------
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X, y)

    rf_path = (
        f"models/{coin}_rf.pkl"
    )

    joblib.dump(
        rf_model,
        rf_path
    )

    # -------------------------
    # XGBOOST
    # -------------------------
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    xgb_model.fit(X, y)

    xgb_path = (
        f"models/{coin}_xgb.pkl"
    )

    joblib.dump(
        xgb_model,
        xgb_path
    )

    print(
        f"Saved models for {coin}"
    )


print("\nAll training completed")
```
