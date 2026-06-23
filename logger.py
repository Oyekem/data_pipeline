import logging
from datetime import datetime
import os

# create logs folder
os.makedirs("logs", exist_ok=True)

log_file = f"logs/pipeline_{datetime.now().date()}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("crypto_pipeline")