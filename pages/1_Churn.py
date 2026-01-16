import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import shap
import os

# --- IMPORTS ---
from components.navigation import sidebar_menu
from src.churn_engine import ChurnPredictor
from src.recommendation_engine import generate_business_logic
from src.data_loader import load_dataset
from src.config import FILES

# --- SETUP ---
st.set_page_config(page_title="Churn AI", layout="wide")
sidebar_menu()

# --- 1. SELF-HEALING & RESET LOGIC (The Fix) ---
# This block forces a reload if the engine is "Stale" (missing the new function)
if 'churn_engine' in st.session_state:
    engine = st.session_state['churn_engine']
    # Check if this object is the "Old Version" (missing the fix)
    if not hasattr(engine, 'get_shap_data'):
        del st.session_state['churn_engine']
        st.session_state['churn_engine'] = ChurnPredictor()
        st.rerun() # Refresh page immediately

# Initialize if missing
if 'churn_engine' not in st.session_state:
    st.session_state['churn_engine'] = ChurnPredictor()

churn_bot = st.session_state['churn_engine']

# --- SIDEBAR: MANUAL RETRAIN OPTION ---
with st.sidebar:
    st.markdown("---")
    st.header(" Model Controls")
    if st.button(" Reset & Retrain Model", type="primary"):
        # Clear model from memory
        st.session_state['churn_engine'] = ChurnPredictor()
        # Optional: Delete physical files to force fresh training
        if os.path.exists("models/churn_model.pkl"):
            os.remove("models/churn_model.pkl")
        st.rerun()

# --- 2. DATA LOADING ---
def get_churn_data():
    if 'capability_map' not in st.session_state: return pd.DataFrame()
    source_ref = st.session_state['capability_map'].get('churn')
    
    if source_ref == 'MEMORY':
        return st.session_state['data_cache'].get('churn', pd.DataFrame())
    elif source_ref in FILES:
        return load_dataset(FILES[source_ref])
    elif 'churn' in st.session_state.get('data_cache', {}):
        return st.session_state['data_cache']['churn']
    return pd.DataFrame()

df = get_churn_data()
st.title(" Churn Prediction Engine")

if df.empty:
    st.error(" No Churn Data Found.")
    st.info("Please go to the **Home** page and click ' Load Demo Data' or Upload a file.")
    st.stop()

# --- 3. TRAINING INTERFACE ---
# If model is not trained (or was just reset), show this button
if churn_bot.model is None:
    st.info(f" Dataset loaded: {len(df)} customer records available.")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button(" Train Model Now", type="primary"):
            with st.spinner("Training XGBoost Model..."):
                metrics = churn_bot.train(df)
            
            if "error" in metrics:
                st.error(metrics['error'])
            else:
                st.success(f"Training Complete! Accuracy: {metrics['accuracy']:.1%}")
                st.rerun()
    st.stop() # Stop here until trained

# --- 4. MAIN DASHBOARD ---
tab_dashboard, tab_simulator = st.tabs([" Risk Dashboard", " What-If Simulator"])

# === TAB 1: MACRO DASHBOARD ===
with tab_dashboard:
    if 'churn_scored' not in st.session_state:
        st.session_state['churn_scored'] = churn_bot.predict(df)
    scored_df = st.session_state['churn_scored']

    # A. KPI ROW
    k1, k2, k3 = st.columns(3)
    avg_risk = scored_df['Churn Probability'].mean()
    high_risk_count = len(scored_df[scored_df['Risk Group'] == 'High'])
    high_risk_pct = high_risk_count / len(scored_df)
    
    # Try to find revenue column for "Revenue at Risk"
    mrr_col = next((c for c in scored_df.columns if c.lower() in ['monthlycharges', 'amount', 'totalamount', 'sales']), None)
    
    k1.metric("Avg Churn Probability", f"{avg_risk:.1%}")
    k3.metric("High Risk Customers", high_risk_count, delta=f"{high_risk_pct:.1%} of Base", delta_color="inverse")

    if mrr_col:
        risk_mrr = scored_df[scored_df['Risk Group'] == 'High'][mrr_col].sum()
        k2.metric("Revenue at Risk", f"${risk_mrr:,.0f}")
    else:
        k2.metric("Revenue at Risk", "N/A")

    # B. CHARTS
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk Distribution")
        fig_hist = px.histogram(scored_df, x="Churn Probability", color="Risk Group",
                                color_discrete_map={'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e74c3c'}, nbins=20)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with c2:
        st.subheader("Top Risk Drivers")
        imp_df = churn_bot.get_directional_importance(scored_df)
        if not imp_df.empty:
            fig_bar = px.bar(imp_df, x='Importance', y='Feature', color='Impact',
                             color_discrete_map={"Increases Risk 🔴": '#e74c3c', "Decreases Risk 🟢": '#2ecc71', "Key Risk Driver ⚠️": 'gray'},
                             orientation='h')
            st.plotly_chart(fig_bar, use_container_width=True)

    # C. AUTOMATED STRATEGIES
    st.markdown("---")
    st.subheader(" AI Retention Plan")
    col_rec, _ = st.columns([3, 1])
    
    with col_rec:
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in ['customerID', 'Churn', 'Risk Group', 'churn']]
        
        # Default selections
        default_vars = cat_cols[:4] if len(cat_cols) >= 4 else cat_cols
        selected_vars = st.multiselect("Analyze Retention Drivers:", cat_cols, default=default_vars)
        
        if selected_vars:
            recommendations = churn_bot.recommend_retention_plan(df, target_cols=selected_vars)
            if recommendations:
                rec_cols = st.columns(len(recommendations))
                for i, (feature, details) in enumerate(recommendations.items()):
                    with rec_cols[i]:
                        with st.container(border=True):
                            st.markdown(f"**{feature}**")
                            st.metric(label="Safest Segment", value=str(details['best_option']))
                            st.caption(f"Churn Rate: {details['churn_rate']:.1%}")

# === TAB 2: SIMULATOR ===
with tab_simulator:
    st.subheader(" Customer Inspector & Simulator")
    
    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        st.info("Adjust values to see how risk changes.")
        # Get a baseline customer
        base_row = churn_bot.get_average_customer(df)
        adjustments = {}
        
        # 1. Sliders for Numbers
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        target_nums = [c for c in num_cols if c not in ['Churn', 'churn', 'customerID']]
        
        for col in target_nums[:3]: # Limit to top 3 for UI cleanliness
             val = float(base_row.get(col, 0))
             max_val = float(df[col].max()) * 1.2 if not df.empty else 100.0
             adjustments[col] = st.slider(f"{col}", 0.0, max_val, val)
        
        # 2. Dropdowns for Categories
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        target_cats = [c for c in cat_cols if c not in ['customerID', 'Churn', 'churn']]
        
        for col in target_cats[:4]: # Limit to top 4
            options = df[col].unique().tolist()
            curr = base_row.get(col, options[0])
            # Ensure index is valid
            idx = options.index(curr) if curr in options else 0
            adjustments[col] = st.selectbox(f"{col}", options, index=idx)

    with col_viz:
        # Run Simulation
        sim_result = churn_bot.simulate_churn(base_row, adjustments)
        
        if sim_result:
            new_prob = sim_result['probability']
            s1, s2 = st.columns(2)
            s1.metric("Predicted Churn Probability", f"{new_prob:.1%}")
            
            risk_label = sim_result['risk_group']
            color = "red" if risk_label == "High" else "orange" if risk_label == "Medium" else "green"
            s2.markdown(f"### Risk: :{color}[{risk_label}]")
            
            st.markdown("---")
            st.write("**Why this prediction? (SHAP Waterfall)**")
            
            # Prepare data for SHAP
            # We reconstruct the row as a DataFrame
            sim_row_df = pd.DataFrame([base_row])
            for k, v in adjustments.items():
                sim_row_df[k] = v
            
            try:
                shap_values = churn_bot.get_shap_data(sim_row_df)
                if shap_values:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    # Check if shap_values is the new object type or old list
                    if hasattr(shap_values, "values"):
                        shap.plots.waterfall(shap_values, show=False, max_display=10)
                    else:
                        # Fallback for older SHAP versions if necessary
                        st.warning("SHAP visualization requires newer library version.")
                        
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate SHAP chart: {e}")