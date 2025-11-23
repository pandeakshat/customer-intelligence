import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# --- IMPORTS ---
from components.navigation import sidebar_menu
from src.churn_engine import ChurnPredictor
from src.recommendation_engine import generate_business_logic
from src.data_loader import load_dataset
from src.config import FILES

st.set_page_config(page_title="Churn AI", layout="wide")
sidebar_menu()

# --- AUTO-REPAIR SESSION STATE ---
if 'churn_bot' in st.session_state:
    bot = st.session_state['churn_bot']
    if not hasattr(bot, 'get_average_customer'): # Check for new method
        del st.session_state['churn_bot']
        st.session_state['churn_bot'] = ChurnPredictor()
        st.rerun()
else:
    st.session_state['churn_bot'] = ChurnPredictor()

bot = st.session_state['churn_bot']

# --- HELPER: GET DATA ---
def get_churn_data():
    if 'capability_map' not in st.session_state: return pd.DataFrame()
    source = st.session_state['capability_map'].get('churn')
    if source == 'MEMORY': return st.session_state['data_cache'].get('churn')
    return load_dataset(FILES.get(source)) if source else pd.DataFrame()

# --- MAIN ---
st.title("🔮 Churn Prediction Engine")

# 1. LOAD & TRAIN
flags = st.session_state.get('flags', {})
if not flags.get('churn'):
    st.error("❌ No Churn Data active."); st.stop()

df = get_churn_data()
col1, col2 = st.columns([3, 1])
with col1: st.info(f"Loaded {len(df)} records.")
with col2:
    if st.button("🔄 Train Model", type="primary"):
        with st.spinner("Training XGBoost..."): metrics = bot.train(df)
        st.success(f"Trained! Accuracy: {metrics['accuracy']:.1%}")

if not bot.model: st.warning("⚠️ Please Train the model."); st.stop()

# 3. INTERFACE
tab_dashboard, tab_profiler = st.tabs(["📊 Risk Dashboard", "🕵️ Customer Inspector"])

# --- TAB 1: MACRO VIEW ---
with tab_dashboard:
    scored_df = bot.predict(df)
    
    # METRICS
    k1, k2, k3 = st.columns(3)
    avg_risk = scored_df['Churn Probability'].mean()
    k1.metric("Avg Churn Probability", f"{avg_risk:.1%}")
    
    total_mrr = scored_df['MonthlyCharges'].sum()
    risk_mrr = scored_df[scored_df['Risk Group'] == 'High']['MonthlyCharges'].sum()
    risk_mrr_pct = (risk_mrr / total_mrr) if total_mrr > 0 else 0
    k2.metric("Revenue at Risk (High)", f"${risk_mrr:,.0f}", delta=f"{risk_mrr_pct:.1%} of Total MRR", delta_color="inverse")
    
    total_cust = len(scored_df)
    risk_cust = len(scored_df[scored_df['Risk Group'] == 'High'])
    risk_cust_pct = (risk_cust / total_cust) if total_cust > 0 else 0
    k3.metric("High Risk Customers", f"{risk_cust}", delta=f"{risk_cust_pct:.1%} of Base", delta_color="inverse")

    # CHARTS
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Risk Distribution")
        fig_hist = px.histogram(scored_df, x="Churn Probability", color="Risk Group", 
                                color_discrete_map={'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e74c3c'}, nbins=20)
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        st.subheader("Top Risk Drivers")
        imp_df = bot.get_directional_importance(scored_df)
        fig_bar = px.bar(imp_df, x='Importance', y='Feature', color='Impact',
                         color_discrete_map={"Increases Risk 🔴": '#e74c3c', "Decreases Risk 🟢": '#2ecc71'}, orientation='h')
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- BEST PLAN (CUSTOMIZABLE) ---
    st.markdown("---")
    st.subheader("🛡️ The Ideal Retention Plan")
    
    # Multiselect for Customization
    all_cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Remove metadata columns
    all_cat_cols = [c for c in all_cat_cols if c not in ['customerID', 'Churn', 'Risk Group']]
    
    default_cols = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport']
    # Ensure defaults exist in the data
    default_cols = [c for c in default_cols if c in all_cat_cols]
    
    selected_vars = st.multiselect("Select variables to optimize:", all_cat_cols, default=default_cols)
    
    if selected_vars:
        recommendations = bot.recommend_retention_plan(df, target_cols=selected_vars)
        rec_cols = st.columns(4)
        for i, (feature, details) in enumerate(recommendations.items()):
            col_idx = i % 4
            with rec_cols[col_idx]:
                with st.container(border=True):
                    st.markdown(f"**{feature}**")
                    st.metric(label="Safest Option", value=details['best_option'])
                    st.caption(f"Churn Rate: {details['churn_rate']:.1%}")
    else:
        st.info("Select variables above to see recommendations.")

# --- TAB 2: MICRO VIEW ---
with tab_profiler:
    st.subheader("Individual Customer Analysis")
    
    # 1. SELECT MODE
    mode = st.radio("Simulation Mode:", ["Generic Persona (Average Customer)", "Specific Customer"], horizontal=True)
    
    row = None
    
    if mode == "Specific Customer":
        cust_id = st.selectbox("Search Customer ID:", df['customerID'].unique())
        row = df[df['customerID'] == cust_id].iloc[0]
    else:
        # Create Generic Row
        row = bot.get_average_customer(df)
        st.info("loaded an 'Average Customer' profile based on the dataset median/mode.")
    
    st.markdown("---")
    
    # 2. SIMULATOR CONTROLS
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("### 🎛️ Simulator")
        
        # Smart Defaults based on the loaded row
        # We handle potential errors if column doesn't exist in row
        def safe_get(col, default): return row[col] if col in row else default
        
        # Contract
        curr_contract = safe_get('Contract', 'Month-to-month')
        opts_contract = ["Month-to-month", "One year", "Two year"]
        idx_contract = opts_contract.index(curr_contract) if curr_contract in opts_contract else 0
        sim_contract = st.selectbox("Contract", opts_contract, index=idx_contract)
        
        # Tenure
        curr_tenure = int(safe_get('tenure', 12))
        sim_tenure = st.slider("Tenure (Months)", 0, 72, curr_tenure)
        
        # Charges
        curr_charge = float(safe_get('MonthlyCharges', 50.0))
        sim_charges = st.number_input("Monthly Charges", 0.0, 200.0, curr_charge)
        
        # Tech Support (Extra Variable)
        curr_tech = safe_get('TechSupport', 'No')
        opts_tech = ['No', 'Yes', 'No internet service']
        idx_tech = opts_tech.index(curr_tech) if curr_tech in opts_tech else 0
        sim_tech = st.selectbox("Tech Support", opts_tech, index=idx_tech)
        
        # Prepare Simulation Data
        sim_row = row.copy()
        sim_row['Contract'] = sim_contract
        sim_row['tenure'] = sim_tenure
        sim_row['MonthlyCharges'] = sim_charges
        sim_row['TechSupport'] = sim_tech
        
        # Predict
        new_risk = bot.predict(pd.DataFrame([sim_row])).iloc[0]['Churn Probability']
        
        st.metric("Predicted Risk", f"{new_risk:.1%}")
        
        if new_risk > 0.7: st.error("High Risk")
        elif new_risk < 0.3: st.success("Safe")
        else: st.warning("At Risk")
        
        # Recommendation Logic
        st.markdown("---")
        rec_context = pd.DataFrame([{'probability': new_risk}])
        strategy = generate_business_logic(rec_context, context='churn').iloc[0]
        st.info(f"**Action:** {strategy['Next Best Action']}")

    with col_right:
        st.markdown("### 🧠 Why?")
        st.caption("Waterfall plot explains which features contribute most to the risk.")
        
        shap_data = bot.get_shap_data(pd.DataFrame([sim_row]))
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_data, show=False, max_display=10)
        st.pyplot(fig)