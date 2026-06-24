from sqlalchemy import text
from logger import logger

# -------------------------
# SAVE CRYPTO DATA
# -------------------------
def save_to_db(engine, data):
    if not data:
        logger.warning("No data received for DB insert")
        return

    query = text("""
        INSERT INTO crypto_prices (coin, price, market_cap, created_at)
        VALUES (:coin, :price, :market_cap, NOW())
        ON CONFLICT DO NOTHING
    """)

    try:
        with engine.begin() as conn:
            conn.execute(query, data)

        logger.info(f"DB insert successful: {len(data)} records")

    except Exception as e:
        logger.error(f"DB insert failed: {e}")
        raise


# -------------------------
# SAVE PIPELINE LOGS
# -------------------------
def save_pipeline_log(engine, status, runtime, error_message):
    query = text("""
        INSERT INTO pipeline_runs (status, runtime_seconds, error_message)
        VALUES (:status, :runtime, :error)
    """)

    try:
        with engine.begin() as conn:
            conn.execute(query, {
                "status": status,
                "runtime": runtime,
                "error": error_message
            })

    except Exception as e:
        logger.error(f"Pipeline log insert failed: {e}")