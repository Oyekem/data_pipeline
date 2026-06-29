
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

engine = create_engine(os.getenv("DB_URL"))
MODEL_DIR = os.getenv("MODEL_DIR", "models")
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", 30))