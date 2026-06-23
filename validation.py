from logger import logger

def validate_data(data):
    if not data or not isinstance(data, list):
        return []

    cleaned = []

    for item in data:
        try:
            coin = item.get("id")
            price = item.get("current_price")
            market_cap = item.get("market_cap")

            if not coin or price is None or market_cap is None:
                continue

            cleaned.append({
                "coin": coin,
                "price": float(price),
                "market_cap": float(market_cap)
            })

        except Exception:
            continue

    return cleaned