import schedule
import time
from pipeline import run_pipeline
from logger import logger

def job():
    try:
        run_pipeline()
    except Exception as e:
        logger.error(f"Scheduled run failed: {e}")

schedule.every(1).minutes.do(job)

print("Scheduler started...")

while True:
    schedule.run_pending()
    time.sleep(1)