import streamlit as st
from time import sleep

def sidebar_menu():
    # 1. Title/Logo area
    st.sidebar.header("Customer Intelligence")
    
    # 2. The Navigation Links
    # Note: We use the actual filenames, but custom Labels and Icons
    st.sidebar.page_link("app.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/1_Churn.py", label="Churn Profiler", icon="🔮")
    st.sidebar.page_link("pages/2_Segmentation.py", label="Segmentation", icon="📊")
    st.sidebar.page_link("pages/3_Sentiment.py", label="Sentiment NLP", icon="💬")
    st.sidebar.page_link("pages/4_Geospatial.py", label="Geospatial", icon="🗺️")

    st.sidebar.markdown("---")