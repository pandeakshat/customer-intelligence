import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- IMPORTS ---
from components.navigation import sidebar_menu
from src.churn_engine import ChurnPredictor
from src.segment_engine import SegmentationEngine
from src.sentiment_engine import SentimentAnalyzer
from src.config import FILES

st.set_page_config(page_title="Customer Inspector", layout="wide")
sidebar_menu()

st.title("Customer Inspector")
st.markdown("Deep dive into specific datasets to analyze individual customer profiles.")

# --- 1. CONTEXT SELECTION ---
def get_available_datasets():
    datasets = {}
    if 'data_cache' in st.session_state:
        for key, df in st.session_state['data_cache'].items():
            if not df.empty:
                datasets[key] = df
    return datasets

available = get_available_datasets()

if not available:
    st.error("No Datasets Loaded.")
    st.info("Please go to **Home** and load Churn, Segmentation, or Sentiment data.")
    st.stop()

# Layout: Context Selection -> ID Selection
c_sel1, c_sel2 = st.columns([1, 2])

with c_sel1:
    selected_context = st.selectbox(
        "1. Select Dataset Source:", 
        list(available.keys()), 
        format_func=lambda x: x.upper()
    )
    df = available[selected_context]

with c_sel2:
    # 1. Try Auto-detect ID
    possible_ids = [c for c in df.columns if any(x in c.lower() for x in ['id', 'customer', 'cust'])]
    
    # 2. Allow Manual Override if missing
    if not possible_ids:
        st.warning("Could not auto-detect ID column.")
        id_col = st.selectbox("Select the Customer ID column:", df.columns)
    else:
        # Default to best guess, but let user change it if wrong
        id_col = st.selectbox("2. Select ID Column:", df.columns, index=df.columns.get_loc(possible_ids[0]))
    
    # 3. Customer Selector
    # Limit to first 5000 for performance
    all_ids = df[id_col].astype(str).unique()
    if len(all_ids) > 5000:
        st.caption(f"Showing first 5,000 of {len(all_ids)} customers.")
        all_ids = all_ids[:5000]
        
    selected_id = st.selectbox(f"3. Select Customer ({len(all_ids)} available):", all_ids)

# Get the Row
customer_row = df[df[id_col].astype(str) == str(selected_id)].iloc[0]

st.markdown("---")

# ==========================================
# MODE A: CHURN VIEW
# ==========================================
if selected_context == 'churn':
    st.header(f"Churn Risk Profile: {selected_id}")
    
    churn_bot = ChurnPredictor()
    risk_data = churn_bot.predict_single(customer_row.to_dict())
    
    if risk_data:
        prob = risk_data['probability']
        risk_group = risk_data['risk_group']
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Probability", f"{prob:.1%}", delta_color="inverse")
        m2.metric("Risk Label", risk_group)
        
        # Try to find a contract/tenure column
        meta_col = next((c for c in df.columns if c.lower() in ['contract', 'tenure', 'subscription']), None)
        val = customer_row[meta_col] if meta_col else "Unknown"
        m3.metric("Contract / Tenure", val)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Risk Score"},
                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#e74c3c" if prob > 0.5 else "#2ecc71"}}
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Key Risk Drivers")
            # Dynamic Table: Show top 5 columns that are NOT ID or Churn
            ignore = [id_col, 'Churn', 'churn', 'customerID']
            cols = [c for c in df.columns if c not in ignore][:5]
            disp_data = {k: customer_row[k] for k in cols}
            st.table(pd.DataFrame(disp_data.items(), columns=['Factor', 'Value']).set_index('Factor'))

# ==========================================
# MODE B: SEGMENTATION VIEW (Fixed Radar)
# ==========================================
elif selected_context == 'segmentation':
    st.header(f"Customer Segment: {selected_id}")
    
    seg_bot = SegmentationEngine()
    
    # 1. AUTO-RUN LOGIC
    if 'Cluster_Label' not in df.columns:
        with st.spinner("Calculating segments..."):
            try:
                res = seg_bot.run_segmentation_model(df, k=4)
                df = res['data']
                customer_row = df[df[id_col].astype(str) == str(selected_id)].iloc[0]
            except Exception as e:
                st.warning(f"Auto-segmentation failed: {e}")

    # 2. Get Segment Info
    cluster_id, label = seg_bot.get_segment_for_customer(customer_row)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Segment Name", label)
    
    # Find Spend Column dynamically
    spend_col = next((c for c in df.columns if any(x in c.lower() for x in ['monetary', 'amount', 'sales', 'spend', 'charge'])), None)
    val = customer_row[spend_col] if spend_col else 0
    m2.metric("Monetary Value", f"${val:,.2f}" if isinstance(val, (int, float)) else val)
    m3.metric("Cluster ID", cluster_id)
    
    # 3. DYNAMIC RADAR CHART & COMPARISON
    st.subheader("Comparative Analysis")
    c_radar, c_table = st.columns([1, 1])
    
    # Identify Numeric Columns for comparison (exclude ID)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != id_col and c != 'Cluster']
    # Pick Top 5 interesting columns (prioritize Recency/Freq/Monetary if exist)
    priority = ['Recency', 'Frequency', 'Monetary', 'Age', 'Spending_Score_Num', 'Tenure', 'MonthlyCharges', 'TotalCharges']
    radar_cols = [c for c in priority if c in num_cols]
    
    # If standard cols missing, just take first 5 numeric
    if len(radar_cols) < 3:
        radar_cols = num_cols[:5]
    
    if len(radar_cols) >= 3:
        # A. RADAR CHART
        with c_radar:
            # Normalize data (Percentile Rank)
            subset = df[radar_cols].dropna()
            cust_vals = []
            
            for col in radar_cols:
                # Percentile rank of this customer
                rank = pd.Series(subset[col]).rank(pct=True)[subset.index.get_loc(customer_row.name)] if customer_row.name in subset.index else 0.5
                cust_vals.append(rank * 100)
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=cust_vals, theta=radar_cols, fill='toself', name='This Customer'))
            fig.add_trace(go.Scatterpolar(r=[50]*len(radar_cols), theta=radar_cols, name='Average', line=dict(dash='dot', color='grey')))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="Percentile Rank vs Average")
            st.plotly_chart(fig, use_container_width=True)
            
        # B. COMPARISON TABLE
        with c_table:
            st.write("**Data Comparison**")
            
            # Calculate Cluster Mean vs Global Mean
            cluster_mean = df[df['Cluster'] == cluster_id][radar_cols].mean()
            global_mean = df[radar_cols].mean()
            
            comp_data = []
            for col in radar_cols:
                comp_data.append({
                    "Metric": col,
                    "This Customer": customer_row[col],
                    "Segment Avg": round(cluster_mean[col], 2),
                    "Global Avg": round(global_mean[col], 2)
                })
            
            st.dataframe(pd.DataFrame(comp_data).set_index("Metric"), use_container_width=True)
            
    else:
        st.info("Not enough numeric data for detailed comparison.")

# ==========================================
# MODE C: SENTIMENT VIEW
# ==========================================
elif selected_context == 'sentiment':
    st.header(f"Voice of Customer: {selected_id}")
    
    sent_bot = SentimentAnalyzer()
    
    # Manual Text Column Selector if auto-detect fails
    text_col = next((c for c in df.columns if any(x in c.lower() for x in ['review', 'comment', 'text', 'body'])), None)
    
    if not text_col:
         text_col = st.selectbox("Select Review Text Column:", [c for c in df.columns if df[c].dtype == 'object'])
    
    if text_col:
        raw_text = str(customer_row[text_col])
        
        # Analyze
        one_row = pd.DataFrame([customer_row])
        analysis = sent_bot.analyze_sentiment(one_row, text_col)
        
        score = analysis['Sentiment_Score'].iloc[0] if 'Sentiment_Score' in analysis else 0
        label = analysis['Sentiment_Label'].iloc[0] if 'Sentiment_Label' in analysis else "Neutral"
        
        m1, m2 = st.columns(2)
        m1.metric("Sentiment Label", label, delta="Positive" if label=="Positive" else "-Negative" if label=="Negative" else "Neutral")
        m2.metric("Confidence Score", f"{score:.2f}")
        
        st.subheader("Review Content")
        st.info(f"\"{raw_text}\"")
        
        if 'Topic_Label' in customer_row:
             st.caption(f"Detected Topic: **{customer_row['Topic_Label']}**")
             
    else:
        st.warning("No text column found.")

# --- 4. PREMIUM HOOK ---
st.markdown("---")
st.markdown("### Enterprise Reports")

with st.container(border=True):
    c_cta1, c_cta2 = st.columns([3, 1])
    
    with c_cta1:
        st.markdown("**Need a Combined 360° Report?**")
        st.caption("Merge Churn Risk + Sentiment History + Lifetime Value into a single PDF dossier.")
        st.warning("This feature is available in the **Enterprise Plan**.")
        
    with c_cta2:
        if st.button("Get Combined Report"):
            st.toast("Feature locked! Contact Sales to upgrade.", icon="🔒")
        
        st.markdown("[Contact Sales](mailto:customerintelligence@pandeakshat.com)")

# Debug Raw Data
with st.expander("View Raw Data Row"):
    st.write(customer_row)