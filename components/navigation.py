import streamlit as st
import pandas as pd
import os
from src.data_loader import load_dataset
from src.config import FILES

def sidebar_menu():
    # --- 1. NAVIGATION LINKS ---
    with st.sidebar:
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/1_Churn.py", label="Churn Prediction", icon="🔮")
        st.page_link("pages/2_Segmentation.py", label="Segmentation", icon="👥")
        st.page_link("pages/3_Sentiment.py", label="Sentiment AI", icon="💬")
        st.page_link("pages/4_Geospatial.py", label="Geospatial", icon="🗺️")
        st.page_link("pages/5_Customer_View.py", label="Single Customer View", icon="👤")
        
        st.divider()
        
        # --- 2. DATA CONTROLS (Persistent) ---
        st.header("📂 Data Settings")
        
        # Initialize Cache
        if 'data_cache' not in st.session_state: st.session_state['data_cache'] = {}
        if 'flags' not in st.session_state: st.session_state['flags'] = {}
        
        mode = st.radio("Source:", ["Demo Data", "Upload File"], label_visibility="collapsed")
        
        # A. LOAD DEMO DATA
        if mode == "Demo Data":
            st.info("Use pre-loaded datasets to test the capabilities.")
            if st.button("🚀 Load Sample Data", type="primary", use_container_width=True):
                with st.spinner("Hydrating Engines..."):
                    
                    # 1. Load Churn
                    if os.path.exists(FILES['churn']):
                        df = load_dataset(FILES['churn'])
                        st.session_state['data_cache']['churn'] = df
                        _check_geo_piggyback(df, 'churn')
                    
                    # 2. Load Segmentation
                    if os.path.exists(FILES['segmentation']):
                        df = load_dataset(FILES['segmentation'])
                        st.session_state['data_cache']['segmentation'] = df
                        _check_geo_piggyback(df, 'segmentation')
                        
                    # 3. Load Sentiment
                    if os.path.exists(FILES['sentiment']):
                        df = load_dataset(FILES['sentiment'])
                        st.session_state['data_cache']['sentiment'] = df
                        _check_geo_piggyback(df, 'sentiment')
                        
                st.success("✅ Demo Data Active!")
                st.rerun()

        # B. UPLOAD USER DATA
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                if st.button("Process File", use_container_width=True):
                    df = load_dataset(uploaded_file)
                    if not df.empty:
                        # AUTO-ROUTING LOGIC
                        cols = [c.lower() for c in df.columns]
                        
                        routed = []
                        # Is it Churn?
                        if 'churn' in cols or 'tenure' in cols:
                            st.session_state['data_cache']['churn'] = df
                            routed.append("Churn")
                        
                        # Is it Sentiment?
                        if any(x in cols for x in ['review', 'comment', 'text', 'feedback']):
                            st.session_state['data_cache']['sentiment'] = df
                            routed.append("Sentiment")
                            
                        # Is it Segmentation? (Fallback if numeric)
                        if len(df.select_dtypes(include=['number']).columns) > 3:
                            st.session_state['data_cache']['segmentation'] = df
                            routed.append("Segmentation")
                        
                        # Check Geo on EVERYTHING
                        if _check_geo_piggyback(df, 'upload'):
                            routed.append("Geospatial")
                        
                        st.success(f"Loaded for: {', '.join(routed)}")
                        st.rerun()

        # --- 3. STATUS INDICATORS ---
        st.markdown("---")
        st.caption("System Status")
        cache = st.session_state['data_cache']
        
        c1, c2 = st.columns(2)
        c1.metric("Churn", "OK" if 'churn' in cache else "OFF")
        c2.metric("Seg", "OK" if 'segmentation' in cache else "OFF")
        
        c3, c4 = st.columns(2)
        c3.metric("NLP", "OK" if 'sentiment' in cache else "OFF")
        c4.metric("Geo", "OK" if 'geo' in cache else "OFF")

def _check_geo_piggyback(df, source_name):
    """
    Helper: Checks any loaded dataframe for geospatial columns.
    If found, registers it as the Geo dataset.
    """
    cols = [c.lower() for c in df.columns]
    # Keywords that imply location
    geo_keywords = ['city', 'country', 'latitude', 'longitude', 'lat', 'lon', 'location', 'region', 'airport']
    
    if any(k in cols for k in geo_keywords):
        st.session_state['data_cache']['geo'] = df
        st.session_state['flags']['geo_source'] = source_name
        return True
    return False