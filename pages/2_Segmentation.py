import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# --- IMPORTS ---
from components.navigation import sidebar_menu
from src.segment_engine import SegmentationEngine # <--- CHANGED: Import Class
from src.recommendation_engine import generate_business_logic
from src.nlg_engine import NarrativeGenerator
from src.data_loader import load_dataset
from src.config import FILES

st.set_page_config(page_title="Segmentation Deep Dive", layout="wide")
sidebar_menu()

# --- INITIALIZE ENGINE ---
if 'seg_engine' not in st.session_state:
    st.session_state['seg_engine'] = SegmentationEngine()
seg_bot = st.session_state['seg_engine']

# --- HELPER: ROBUST DATA LOADER ---
def get_active_data(module_key):
    """
    Robustly retrieves data from Memory (Uploads) or Disk (Sample Paths).
    """
    if 'capability_map' not in st.session_state: 
        return pd.DataFrame()
    
    source = st.session_state['capability_map'].get(module_key)
    
    # 1. Check Memory Cache
    if source == 'MEMORY': 
        return st.session_state.get('data_cache', {}).get(module_key, pd.DataFrame())
    
    # 2. Check Disk
    if source in FILES:
        return load_dataset(FILES[source])

    # 3. Fallback
    return st.session_state.get('data_cache', {}).get(module_key, pd.DataFrame())

st.title("📊 Segmentation Deep Dive")

# 1. LOAD DATA
flags = st.session_state.get('flags', {})
df = pd.DataFrame()
mode_name = "Generic"

if flags.get('segmentation'):
    df = get_active_data('segmentation')
    mode_name = "Demographic"
elif flags.get('rfm'): # Fallback to RFM flag if Segmentation flag missed
    df = get_active_data('rfm')
    mode_name = "RFM"
elif 'segmentation' in st.session_state.get('data_cache', {}):
    df = st.session_state['data_cache']['segmentation']
    mode_name = "Unknown"

if df.empty:
    st.error("❌ No Segmentation Data Found.")
    st.info("Please go to Home and Load/Upload data.")
    st.stop()

# 2. SIDEBAR TUNING
with st.sidebar:
    st.header("⚙️ Hyperparameters")
    
    if st.checkbox("💡 Suggest Optimal K?"):
        with st.spinner("Calculating Silhouette Scores..."):
            # Use a sample for speed if dataset is huge
            sample_df = df.head(1000) if len(df) > 1000 else df
            
            # CALL CLASS METHOD
            scores = seg_bot.suggest_optimal_k(sample_df)
            
            if not scores.empty:
                best_k = scores['score'].idxmax()
                st.bar_chart(scores)
                st.success(f"Mathematical Best: k={best_k}")
                k_default = int(best_k)
            else:
                st.warning("Not enough data to suggest K.")
                k_default = 4
    else:
        k_default = 4
        
    k_clusters = st.slider("Number of Segments (k)", 2, 6, k_default)

# 3. RUN ENGINE
try:
    with st.spinner(f"Clustering customers into {k_clusters} groups..."):
        # CALL CLASS METHOD
        result = seg_bot.run_segmentation_model(df, k=k_clusters)
        
        seg_df = result['data']
        dt_model = result['dt_model']
        dt_features = result['dt_features']
        
        # Determine actual mode used by engine
        engine_mode = result.get('mode', mode_name)
        
        # Save to session state for Customer View to access
        st.session_state['segmentation_results'] = seg_df 

except Exception as e:
    st.error(f"Error running segmentation: {e}")
    st.stop()

# --- 4. EXECUTIVE SUMMARY (NLG) ---
st.markdown("### 📝 Executive Summary")

# Generate narrative if possible
try:
    nlg = NarrativeGenerator()
    narratives = nlg.generate_segmentation_narrative(seg_df, 'Cluster', dt_features)
    
    # Create columns for the top 3 biggest clusters
    top_clusters = seg_df['Cluster'].value_counts().head(3).index.tolist()
    cols = st.columns(len(top_clusters))
    
    for idx, col in enumerate(cols):
        cluster_id = top_clusters[idx]
        # Get the "Smart Label" for this ID
        label_row = seg_df[seg_df['Cluster'] == cluster_id]
        label_name = label_row['Cluster_Label'].iloc[0] if not label_row.empty else f"Cluster {cluster_id}"
        
        with col:
            st.info(f"**{label_name}**")
            st.markdown(narratives.get(cluster_id, "No specific insight available."))
except:
    st.warning("Could not generate narrative summary.")

st.markdown("---")

# 5. VISUALIZATION TABS (Deep Dive)
tab_3d, tab_rules, tab_strategy = st.tabs(["🧊 Visual Explorer", "🌳 Logic Tree", "🚀 Business Strategy"])

# --- TAB 1: 3D EXPLORER ---
with tab_3d:
    st.subheader(f"{engine_mode} Clusters in 3D")
    
    # Dynamic Axis Selection based on available features
    if len(dt_features) >= 3:
        x_ax, y_ax, z_ax = dt_features[0], dt_features[1], dt_features[2]
        fig = px.scatter_3d(seg_df, x=x_ax, y=y_ax, z=z_ax, color='Cluster_Label', opacity=0.8)
        st.plotly_chart(fig, use_container_width=True)
    elif len(dt_features) == 2:
        st.warning("Only 2 dimensions available.")
        fig = px.scatter(seg_df, x=dt_features[0], y=dt_features[1], color='Cluster_Label')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough dimensions for plotting.")

# --- TAB 2: LOGIC TREE ---
with tab_rules:
    st.subheader("🧠 Logic Tree: How the AI defined the groups")
    st.markdown("Follow the path to see the rules used to create each segment.")
    
    # Map Integer Classes to String Labels for the Tree Plot
    cluster_map = seg_df[['Cluster', 'Cluster_Label']].drop_duplicates().set_index('Cluster')['Cluster_Label'].to_dict()
    
    # Ensure classes match model output
    present_classes = sorted(list(cluster_map.keys()))
    class_names = [cluster_map.get(c, str(c)) for c in present_classes]

    if dt_model:
        fig_tree, ax = plt.subplots(figsize=(25, 10))
        plot_tree(dt_model, 
                  feature_names=dt_features, 
                  class_names=class_names, # Use mapped names
                  filled=True, 
                  rounded=True, 
                  fontsize=10, 
                  ax=ax)
        st.pyplot(fig_tree)
    else:
        st.info("Decision Tree model not available.")

# --- TAB 3: BUSINESS STRATEGY ---
with tab_strategy:
    st.subheader("💡 Recommendation Engine")
    
    # Calculate summary stats for the recommendation engine
    summary_stats = seg_df.groupby('Cluster_Label')[dt_features].mean().round(1)
    
    # Use existing recommendation logic
    # We pass the engine_mode (Demographic/RFM) to pick the right strategy
    strategy_df = generate_business_logic(summary_stats, context=engine_mode.lower())
    
    if not strategy_df.empty:
        for label, row in strategy_df.iterrows():
            with st.container(border=True):
                col_a, col_b, col_c = st.columns([1, 2, 2])
                
                with col_a:
                    st.markdown(f"### {label}")
                    # Show key stats
                    stats_txt = ""
                    for feat in dt_features[:3]: 
                        val = row.get(feat, 'N/A')
                        if isinstance(val, float): val = round(val, 1)
                        stats_txt += f"{feat}: {val}\n"
                    st.code(stats_txt)
                        
                with col_b:
                    st.markdown("#### 👤 Persona")
                    st.write(f"**{row.get('Persona', 'Standard User')}**")
                    st.info(f"Strategy: {row.get('Strategy', 'Maintain')}")
                    
                with col_c:
                    st.markdown("#### ⚡ Next Best Action")
                    st.success(f"👉 {row.get('Next Best Action', 'Review Account')}")
    else:
        st.info("No specific strategies generated for this data type.")