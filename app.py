import streamlit as st
import pandas as pd
from components.navigation import sidebar_menu
from src.validator import validate_dataset
from src.data_loader import load_dataset
from src.config import FILES

# --- 1. SETUP ---
st.set_page_config(page_title="Customer Intelligence", layout="wide")
sidebar_menu()

if 'data_cache' not in st.session_state: st.session_state['data_cache'] = {}
if 'capability_map' not in st.session_state: st.session_state['capability_map'] = {}
if 'flags' not in st.session_state: st.session_state['flags'] = {}
if 'meta' not in st.session_state: st.session_state['meta'] = {}

st.title("🧠 Customer Intelligence Hub")
st.markdown("### 🛠️ Data Configuration Console")

# --- 2. SELECTION PANEL ---
col1, col2 = st.columns(2)

with col1:
    selected_module_name = st.selectbox(
        "1. Select Module to Test",
        ["Churn Prediction", "Customer Segmentation", "Sentiment Analysis", "Geospatial View"]
    )
    KEY_MAP = {
        "Churn Prediction": "churn",
        "Customer Segmentation": "segmentation",
        "Sentiment Analysis": "sentiment",
        "Geospatial View": "geo"
    }
    target_module = KEY_MAP[selected_module_name]

with col2:
    data_source = st.radio("2. Select Data Source", ["Use Sample Data", "Upload File"], horizontal=True)

# --- 3. DATA LOADING ---
st.markdown("---")
df = pd.DataFrame()
source_ref = None
source_type = None

if data_source == "Use Sample Data":
    st.info(f"📂 Using default sample for **{selected_module_name}**")
    # Logic: Geo defaults to sentiment if selected directly, otherwise use target
    file_key = target_module if target_module != 'geo' else 'sentiment'
    if st.button("🚀 Load & Validate Sample"):
        path = FILES.get(file_key)
        df = load_dataset(path)
        source_type = "FILE"
        source_ref = file_key

elif data_source == "Upload File":
    uploaded_file = st.file_uploader(f"📂 Upload Data for {selected_module_name}", type=["csv", "xlsx"])
    if uploaded_file and st.button("🚀 Process Upload"):
        df = load_dataset(uploaded_file)
        source_type = "MEMORY"
        source_ref = target_module

# --- 4. VALIDATION LOGIC (UPDATED FOR PIGGYBACKING) ---
if not df.empty:
    st.markdown("### 🚦 Validation Results")
    
    # A. Validate the TARGET Module
    report = validate_dataset(df, target_module=target_module)
    module_status = report.get(target_module, {})
    is_ready = module_status.get('ready', False)
    
    if is_ready:
        st.success(f"✅ **{target_module.upper()}** Capabilities Active!")
        
        # Save Primary Data
        st.session_state['flags'][target_module] = True
        if source_type == "MEMORY":
            st.session_state['data_cache'][target_module] = df
            st.session_state['capability_map'][target_module] = 'MEMORY'
        else:
            st.session_state['capability_map'][target_module] = source_ref
            
        # Save Metadata (Flavor/Mapping)
        if 'flavor' in module_status:
            detected_flavor = module_status['flavor']
            st.session_state['meta'][target_module] = {'flavor': detected_flavor}
            st.info(f"🔹 Detected Sub-Type: **{detected_flavor.title()}**")
            if module_status['flavors'][detected_flavor]['column_mapping']:
                st.toast(f"Mapped columns for {target_module}")
        elif 'column_mapping' in module_status:
             if module_status['column_mapping']:
                st.toast(f"Mapped columns for {target_module}")

        # B. AUTO-CHECK FOR GEOSPATIAL (The "Piggyback")
        # We check if this same dataset ALSO supports Geo
        geo_report = validate_dataset(df, target_module='geo')
        geo_status = geo_report.get('geo', {})
        
        if geo_status.get('ready'):
            st.success(f"🌍 **GEOSPATIAL** Capabilities ALSO Detected!")
            
            # Enable Geo Flag
            st.session_state['flags']['geo'] = True
            
            # Link Geo to THIS dataset
            if source_type == "MEMORY":
                st.session_state['data_cache']['geo'] = df # It's the same DF
                st.session_state['capability_map']['geo'] = 'MEMORY'
            else:
                st.session_state['capability_map']['geo'] = source_ref # Same file ref
            
            # Store Context: "This Geo data came from Churn/Sentiment/etc"
            geo_col = geo_status['column_mapping'].get('Location', 'Location') # Default name
            st.session_state['meta']['geo'] = {
                'parent_context': target_module,
                'location_col': geo_col
            }
        else:
            st.caption("No geospatial columns found in this dataset.")

        st.markdown("---")
        st.success(f"👉 Navigation Enabled.")
        
    else:
        st.error(f"❌ **{target_module.upper()}** Failed Validation.")
        if 'flavors' in module_status:
            st.warning("Data did not match any known patterns:")
            for flav, res in module_status['flavors'].items():
                st.write(f"- **{flav.title()}**: Missing `{res['missing']}`")
        else:
            st.write(f"Missing Columns: `{module_status.get('missing')}`")