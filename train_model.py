import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from config import engine

df = pd.read_sql("SELECT * FROM crypto_prices", engine)
df = df[df["coin"].str.lower() == "bitcoin"].sort_values("created_at")

df["lag1"] = df["price"].shift(1)
df["lag2"] = df["price"].shift(2)
df["lag3"] = df["price"].shift(3)
df["ma5"] = df["price"].rolling(5).mean()
df["std5"] = df["price"].rolling(5).std()

df = df.dropna()

X = df[["lag1","lag2","lag3","ma5","std5"]]
y = df["price"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "models/rf_model.pkl")

print("Model saved!")