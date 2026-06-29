import time
import requests
import pandas as pd

from api import fetch_crypto
from validation import validate_data
from db import save_to_db, save_pipeline_log
from config import engine

from utils.logger import logger

logger.info("Pipeline started")

try:
    # fetch data
    logger.info("Data fetched successfully")
except Exception as e:
    logger.error(f"Pipeline failed: {e}")



# -------------------------
# METRICS HELPERS
# -------------------------
def compute_success_rate(engine):
    query = "SELECT status FROM pipeline_runs"
    df = pd.read_sql(query, engine)

    if df.empty:
        return 0

    success_rate = (df["status"] == "success").mean()
    return round(success_rate * 100, 2)


def check_api_health():
    try:
        res = requests.get("https://api.coingecko.com/api/v3/ping", timeout=5)
        return res.status_code == 200
    except:
        return False


# -------------------------
# ALERT SYSTEM (PLACEHOLDER)
# -------------------------
def send_alert(message):
    logger.warning(f"ALERT: {message}")
    # TODO: email / telegram integration


# -------------------------
# PIPELINE
# -------------------------
def run_pipeline():
    start = time.time()

    try:
        logger.info("Pipeline started")

        # API health check
        if not check_api_health():
            send_alert("Crypto API is down")

        raw_data = fetch_crypto()
        clean_data = validate_data(raw_data)

        if not clean_data:
            raise ValueError("No valid data returned")

        save_to_db(engine, clean_data)

        runtime = time.time() - start

        save_pipeline_log(engine, "success", runtime, None)

        success_rate = compute_success_rate(engine)

        logger.info(f"Pipeline success rate: {success_rate}%")

    except Exception as e:
        runtime = time.time() - start

        save_pipeline_log(engine, "failed", runtime, str(e))

        send_alert(f"Pipeline failed: {str(e)}")

        logger.error(f"Pipeline crashed: {e}")


if __name__ == "__main__":
    run_pipeline()