from sqlalchemy import text
from logger import logger

def save_to_db(engine, data):
    if not data:
        logger.warning("No data received for DB insert")
        return

    query = text("""
        INSERT INTO crypto_prices (coin, price, market_cap)
        VALUES (:coin, :price, :market_cap)
        ON CONFLICT (coin, created_at) DO NOTHING
    """)

    try:
        with engine.begin() as conn:
            conn.execute(query, data)

        logger.info(f"DB upsert successful: {len(data)} records")

    except Exception as e:
        logger.error(f"DB insert failed: {e}")