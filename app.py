import streamlit as st

st.set_page_config(
    page_title="Crypto Intelligence System",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------
st.title("Crypto Intelligence System")
st.subheader("AI-powered Crypto Prediction & Analytics Platform")

# -----------------------------
# SYSTEM DESCRIPTION
# -----------------------------
st.markdown("""
### What this system does

This platform is a real-time AI system that:

- Collects live crypto price data
- Trains ML models (Random Forest + XGBoost)
- Predicts short-term price movement
- Combines predictions using ensemble logic
- Generates BUY / SELL / HOLD signals
- Tracks prediction accuracy over time
- Shows confidence bands & volatility
- Stores all predictions in a database
- Updates automatically in real-time

---

### Navigate the system below
Click any section to open its module:
""")

# -----------------------------
# NAVIGATION SECTION
# -----------------------------

st.markdown("## 🔗 Modules")

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/1_Introduction.py", label="🏠 Introduction")
    st.page_link("pages/3_Prediction.py", label="📈 Prediction Engine")

with col2:
    st.page_link("pages/2_Analytics_Dashboard.py", label="📊 Analytics Dashboard")


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.info("Tip: Use the Live Dashboard for real-time price tracking")

st.caption("Crypto Intelligence System • Streamlit + ML Pipeline")