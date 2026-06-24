import time
from api import fetch_crypto
from validation import validate_data
from db import save_to_db, save_pipeline_log
from config import engine
from logger import logger


def run_pipeline():
    start = time.time()

    try:
        logger.info("Pipeline started")

        raw_data = fetch_crypto()
        clean_data = validate_data(raw_data)

        save_to_db(engine, clean_data)

        runtime = time.time() - start
        save_pipeline_log(engine, "success", runtime, None)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        runtime = time.time() - start
        save_pipeline_log(engine, "failed", runtime, str(e))

        logger.error(f"Pipeline crashed: {e}")


if __name__ == "__main__":
    run_pipeline()