from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def backtest(model, df):
    df = create_features(df)

    X = df[["lag1", "lag2", "lag3", "ma5", "std5"]]
    y = df["price"]

    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))

    return mae, rmse