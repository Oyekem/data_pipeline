import streamlit as st

st.title("Crypto Data Pipeline System")

st.write("""
Welcome to your real-time Crypto Analytics System.

This project demonstrates a full data engineering pipeline:

✔ API ingestion (CoinGecko)
✔ Data validation layer
✔ Cloud PostgreSQL (Neon)
✔ GitHub Actions automation
✔ Live Streamlit dashboard

---
""")

st.subheader("System Architecture")

st.write("""
API → Python ETL Pipeline → Validation → Neon DB → Streamlit Dashboard
""")

st.subheader("Features")

st.write("""
- Live crypto price tracking
- Automated data ingestion every 5 minutes
- Pipeline monitoring
- Failure detection
- Trend visualization
- Prediction engine (next 5 min forecast)
""")