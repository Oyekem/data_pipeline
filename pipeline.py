import time
from logger import logger
from api import fetch_crypto
from validation import validate_data
from db import save_to_db
from config import engine


def run_pipeline():
    start = time.time()
    logger.info("Pipeline started")

    try:
        # STEP 1: Fetch data
        raw_data = fetch_crypto()

        # STEP 2: Validate data
        clean_data = validate_data(raw_data)

        # STEP 3: Save to DB
        save_to_db(engine, clean_data)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline crashed: {e}")

    finally:
        duration = time.time() - start
        logger.info(f"Pipeline runtime: {duration:.2f}s")


if __name__ == "__main__":
    run_pipeline()