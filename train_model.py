import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from config import engine


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_sql("SELECT * FROM crypto_prices", engine)

df = df[df["coin"].str.lower() == "bitcoin"].copy()
df = df.sort_values("created_at")


# -------------------------
# FEATURE ENGINEERING
# -------------------------
df["lag1"] = df["price"].shift(1)
df["lag2"] = df["price"].shift(2)
df["lag3"] = df["price"].shift(3)

df["ma5"] = df["price"].rolling(5).mean()
df["std5"] = df["price"].rolling(5).std()

df = df.dropna()


# -------------------------
# TRAIN MODEL
# -------------------------
features = ["lag1", "lag2", "lag3", "ma5", "std5"]

X = df[features]
y = df["price"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# -------------------------
# SAVE MODEL
# -------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/rf_model.pkl")

print("Model trained and saved successfully!")