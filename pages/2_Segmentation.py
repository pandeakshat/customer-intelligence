import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# --- IMPORTS ---
from components.navigation import sidebar_menu
from src.segment_engine import run_segmentation_model, suggest_optimal_k
from src.recommendation_engine import generate_business_logic
from src.data_loader import load_dataset
from src.config import FILES

st.set_page_config(page_title="Segmentation Deep Dive", layout="wide")
sidebar_menu()

# --- HELPER: GET DATA ---
def get_active_data(module_key):
    if 'capability_map' not in st.session_state: return pd.DataFrame()
    source = st.session_state['capability_map'].get(module_key)
    if source == 'MEMORY': return st.session_state.get('data_cache', {}).get(module_key, pd.DataFrame())
    return load_dataset(FILES.get(source)) if source else pd.DataFrame()

st.title("📊 Segmentation Deep Dive")

# 1. LOAD DATA
flags = st.session_state.get('flags', {})
if flags.get('segmentation'):
    df = get_active_data('segmentation')
    mode_name = "Demographic"
elif flags.get('rfm'):
    df = get_active_data('rfm')
    mode_name = "RFM"
else:
    st.error("❌ No Data Found. Please go to Home and Load/Upload data.")
    st.stop()

# 2. SIDEBAR TUNING
with st.sidebar:
    st.header("⚙️ Hyperparameters")
    
    if st.checkbox("💡 Suggest Optimal K?"):
        st.info("Calculating Silhouette Scores...")
        scores = suggest_optimal_k(df.head(500))
        best_k = scores['score'].idxmax()
        st.bar_chart(scores)
        st.success(f"Mathematical Best: k={best_k}")
        k_default = int(best_k)
    else:
        k_default = 4
        
    k_clusters = st.slider("Number of Segments (k)", 2, 6, k_default)

# 3. RUN ENGINE
try:
    result = run_segmentation_model(df, k=k_clusters)
    seg_df = result['data']
    dt_model = result['dt_model']
    dt_features = result['dt_features']
except Exception as e:
    st.error(f"Error running segmentation: {e}")
    st.stop()

# 4. VISUALIZATION TABS
tab_3d, tab_rules, tab_strategy = st.tabs(["🧊 Visual Explorer", "🌳 Logic Tree", "🚀 Business Strategy"])

# --- TAB 1: 3D EXPLORER ---
with tab_3d:
    st.subheader(f"{mode_name} Clusters in 3D")
    if mode_name == "Demographic":
        fig = px.scatter_3d(seg_df, x='Age', y='Spending_Score_Num', z='Family_Size', color='Cluster_Label', opacity=0.8)
    else:
        fig = px.scatter_3d(seg_df.reset_index(), x='Recency', y='Frequency', z='Monetary', color='Cluster_Label', opacity=0.8)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: LOGIC TREE (CRASH FIXED HERE) ---
with tab_rules:
    st.subheader("🧠 Logic Tree: How the AI defined the groups")
    st.markdown("Follow the path to see the rules used to create each segment.")
    
    # --- THE FIX: Map Integer Classes to String Labels ---
    # 1. Create a dictionary: {0: "Young Savers", 1: "High Spenders", ...}
    cluster_map = seg_df[['Cluster', 'Cluster_Label']].drop_duplicates().set_index('Cluster')['Cluster_Label'].to_dict()
    
    # 2. Convert the tree's integer classes to these strings
    # dt_model.classes_ contains [0, 1, 2, 3]
    class_names = [cluster_map.get(c, str(c)) for c in dt_model.classes_]
    # ---------------------------------------------------

    fig_tree, ax = plt.subplots(figsize=(25, 10))
    
    # We use our new 'class_names' list here
    plot_tree(dt_model, 
              feature_names=dt_features, 
              class_names=class_names, # <--- Fixed argument
              filled=True, 
              rounded=True, 
              fontsize=10, 
              ax=ax)
              
    st.pyplot(fig_tree)

# --- TAB 3: BUSINESS STRATEGY ---
with tab_strategy:
    st.subheader("💡 Recommendation Engine")
    st.markdown("Mapping data clusters to behavioral personas and actionable strategies.")
    
    summary_stats = seg_df.groupby('Cluster_Label')[dt_features].mean().round(1)
    strategy_df = generate_business_logic(summary_stats, context=mode_name.lower())
    
    for label, row in strategy_df.iterrows():
        with st.container(border=True):
            col_a, col_b, col_c = st.columns([1, 2, 2])
            
            with col_a:
                st.markdown(f"### {label}")
                if mode_name == "Demographic":
                    st.code(f"Age: {row['Age']}\nSpend: {row['Spending_Score_Num']}\nFam: {row['Family_Size']}")
                else:
                    st.code(f"Recency: {row['Recency']} days\nSpend: ${row['Monetary']}")
                    
            with col_b:
                st.markdown("#### 👤 Persona")
                st.write(f"**{row['Persona']}**")
                st.info(f"Strategy: {row['Strategy']}")
                
            with col_c:
                st.markdown("#### ⚡ Next Best Action")
                st.success(f"👉 {row['Next Best Action']}")