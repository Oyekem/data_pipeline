import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus

load_dotenv()

DATABASE_URL = os.getenv("DB_URL")


engine = create_engine(
	DATABASE_URL,
	pool_pre_ping = True)