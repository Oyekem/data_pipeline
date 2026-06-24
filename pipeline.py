import time
import os
from logger import logger
from api import fetch_crypto
from db import save_to_db
from validation import validate_data
from config import engine

LOCK_FILE = "pipeline.lock"


def run_pipeline():

    start = time.time()
    logger.info("Pipeline started")

    try:
        raw_data = fetch_crypto()
        clean_data = validate_data(raw_data)
        save_to_db(engine, clean_data)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline crashed: {e}")

    finally:
        duration = time.time() - start
        logger.info(f"Pipeline runtime: {duration:.2f}s")


# -------------------------
# LOCK WRAPPER (IMPORTANT)
# -------------------------

if __name__ == "__main__":

    # 1. Check if already running
    if os.path.exists(LOCK_FILE):
        logger.warning("Pipeline already running. Exiting.")
        exit()

    # 2. Create lock file
    open(LOCK_FILE, "w").close()

    try:
        run_pipeline()

    finally:
        # 3. Always remove lock
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)