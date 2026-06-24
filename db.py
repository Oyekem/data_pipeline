from sqlalchemy import text
from logger import logger

def save_to_db(engine, data):
    if not data:
        logger.warning("No data received for DB insert")
        return

    query = text("""
        INSERT INTO crypto_prices (coin, price, market_cap)
        VALUES (:coin, :price, :market_cap)
        ON CONFLICT DO NOTHING
    """)

    try:
        with engine.begin() as conn:
            conn.execute(query, data)  # list of dicts is OK in SQLAlchemy 2.0+

        logger.info(f"DB insert successful: {len(data)} records")

    except Exception as e:
        logger.error(f"DB insert failed: {e}")
        raise