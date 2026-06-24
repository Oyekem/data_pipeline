import streamlit as st

st.autorefresh(interval=30 * 1000, key="datarefresh")

st.set_page_config(
    page_title="Crypto Analytics Platform",
    page_icon="📈",
    layout="wide"
)

st.title("Crypto Analytics Platform")

st.write("""
Use the left sidebar to navigate:

- Introduction
- Analytics Dashboard
- Prediction
""")