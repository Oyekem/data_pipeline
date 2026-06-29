from ml_model import train_model
from config import engine
import pandas as pd
from utils.logger import logger


def retrain():
    logger.info("Retraining started")

    df = pd.read_sql("SELECT * FROM crypto_prices", engine)

    for coin in df["coin"].unique():
        coin_df = df[df["coin"] == coin]

        try:
            train_model(coin_df)
            logger.info(f"Model retrained: {coin}")
        except Exception as e:
            logger.error(f"Failed retraining {coin}: {e}")

if __name__ == "__main__":
    retrain()