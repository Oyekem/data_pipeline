import time
import requests
from logger import logger

def fetch_crypto():
    url = "https://api.coingecko.com/api/v3/coins/markets"

    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 10,
        "page": 1,
        "sparkline": "false"
    }

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # validate response inside function
            if not isinstance(data, list):
                logger.error("Invalid API response format")
                return []

            logger.info("API fetch successful")
            return data

        except Exception as e:
            logger.warning(f"API attempt {attempt+1} failed: {e}")
            time.sleep(5 * (attempt + 1))

    logger.error("API failed after 3 attempts")
    return []