import streamlit as st
import pandas as pd
import time
from src.config import FILES
from src.data_loader import load_dataset
from components.navigation import sidebar_menu

# --- 1. SESSION STATE SETUP ---
def init_session_state():
    if 'data_cache' not in st.session_state: st.session_state['data_cache'] = {}
    if 'capability_map' not in st.session_state: st.session_state['capability_map'] = {}
    if 'flags' not in st.session_state: 
        st.session_state['flags'] = {'churn': False, 'segmentation': False, 'geo': False, 'sentiment': False}

init_session_state()

st.set_page_config(page_title="Customer Intelligence Hub", layout="wide")
sidebar_menu()
st.title("🧠 Customer Intelligence Hub")

# --- 2. THE FIX: LOOSE COLUMN MATCHING ---
def auto_register_data(df, source_name):
    """
    Scans column names using SUBSTRING matching.
    If 'ReviewBody' exists, it sees 'review' and activates Sentiment.
    """
    # Convert all columns to lowercase string for easy matching
    cols = [str(c).lower() for c in df.columns]
    detected = []

    # Helper: Check if ANY keyword exists as a substring in ANY column
    def scan_columns(keywords):
        for col in cols:
            for k in keywords:
                if k in col: return True
        return False

    # A. SENTIMENT DETECTION
    # Matches: 'ReviewBody', 'ReviewHeader', 'VerifiedReview'
    if scan_columns(['review', 'text', 'comment', 'feedback', 'body', 'content', 'rating', 'star']):
        st.session_state['data_cache']['sentiment'] = df
        st.session_state['capability_map']['sentiment'] = 'MEMORY'
        st.session_state['flags']['sentiment'] = True
        detected.append("❤️ Sentiment")

    # B. GEOSPATIAL DETECTION
    # Matches: 'Route', 'Location', 'Airport'
    if scan_columns(['lat', 'lon', 'city', 'country', 'airport', 'location', 'route', 'destination', 'origin']):
        st.session_state['data_cache']['geo'] = df
        st.session_state['capability_map']['geo'] = 'MEMORY'
        st.session_state['flags']['geo'] = True
        detected.append("🌍 Geospatial")

    # C. SEGMENTATION DETECTION
    # Matches: 'ValueForMoney', 'SeatType', 'TotalCharges'
    if scan_columns(['amount', 'sales', 'profit', 'quantity', 'total', 'spend', 'value', 'type', 'class']):
        st.session_state['data_cache']['segmentation'] = df
        st.session_state['capability_map']['segmentation'] = 'MEMORY'
        st.session_state['flags']['segmentation'] = True
        detected.append("📊 Segmentation")
        
    # D. CHURN DETECTION
    if scan_columns(['churn', 'exited', 'status', 'retention']):
        st.session_state['data_cache']['churn'] = df
        st.session_state['capability_map']['churn'] = 'MEMORY'
        st.session_state['flags']['churn'] = True
        detected.append("🔮 Churn")

    return detected

# --- 3. INTERFACE ---
st.markdown("---")
col1, col2 = st.columns([1, 1])

# === SELECT DEMO FILE ===
with col1:
    st.subheader("🚀 Load Demo Data")
    available_files = list(FILES.keys())
    
    if available_files:
        selected_file_key = st.selectbox("Select Dataset:", available_files, format_func=lambda x: x.replace('_', ' ').title())
        
        if st.button("Load Selected Dataset", type="primary"):
            with st.spinner(f"Loading '{selected_file_key}'..."):
                try:
                    df = load_dataset(FILES[selected_file_key])
                    
                    # Run the loose detection
                    modules = auto_register_data(df, selected_file_key)
                    
                    if modules:
                        st.success(f"Loaded! Active Modules: {', '.join(modules)}")
                        time.sleep(1) 
                        st.rerun()
                    else:
                        st.warning("Data loaded, but columns didn't match known patterns.")
                        st.write("Columns:", df.columns.tolist())
                        
                except Exception as e:
                    st.error(f"Failed to load: {e}")

# === UPLOAD FILE ===
with col2:
    st.subheader("📂 Upload Your Own")
    uploaded_file = st.file_uploader("Upload CSV / Excel", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            # Run the loose detection
            modules = auto_register_data(df_upload, "Upload")
            
            if modules:
                st.success(f"Processed! Active Modules: {', '.join(modules)}")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Could not auto-detect module type. Check column names.")
                st.write("Columns found:", df_upload.columns.tolist())
        except Exception as e:
            st.error(f"Error reading file: {e}")

# --- STATUS ---
st.markdown("---")
st.markdown("### 🚦 Active Capabilities")
c1, c2, c3, c4 = st.columns(4)

def badge(key): return "✅ **Online**" if st.session_state['flags'].get(key) else "⚪ *Offline*"

c1.metric("Churn", badge('churn'))
c2.metric("Segmentation", badge('segmentation'))
c3.metric("Geospatial", badge('geo'))
c4.metric("Sentiment", badge('sentiment'))