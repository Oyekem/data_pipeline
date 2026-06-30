import logging
import os
from datetime import datetime

os.makedirs("logs", exist_ok=True)

log_file = f"logs/pipeline_{datetime.now().date()}.log"

logger = logging.getLogger("crypto_pipeline")

if not logger.handlers:
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)