def create_features(df):
    df = df.copy()

    df["lag1"] = df["price"].shift(1)
    df["lag2"] = df["price"].shift(2)
    df["lag3"] = df["price"].shift(3)

    df["ma5"] = df["price"].rolling(5).mean()
    df["std5"] = df["price"].rolling(5).std()

    return df.dropna()