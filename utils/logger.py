import logging
from datetime import datetime
import os

os.makedirs("logs", exist_ok=True)

log_file = f"logs/pipeline_{datetime.now().date()}.log"

logger = logging.getLogger("crypto_pipeline")
logger.setLevel(logging.INFO)

if not logger.handlers:   # IMPORTANT FIX
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)