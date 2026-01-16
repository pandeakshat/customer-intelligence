import streamlit as st
import pandas as pd
import plotly.express as px
from components.navigation import sidebar_menu
from src.sentiment_engine import SentimentAnalyzer
from src.data_loader import load_dataset
from src.config import FILES

st.set_page_config(page_title="Sentiment AI", layout="wide")
sidebar_menu()

# --- 1. RESET LOGIC (Sidebar) ---
# This is crucial: It lets you "Retrain" if you picked the wrong column
with st.sidebar:
    st.markdown("---")
    st.header(" Controls")
    if st.button(" Reset Analysis", type="secondary"):
        if 'sentiment_results' in st.session_state:
            del st.session_state['sentiment_results']
        if 'topics' in st.session_state:
            del st.session_state['topics']
        st.rerun()

# --- 2. LOAD DATA ---
def get_sentiment_data():
    if 'capability_map' not in st.session_state: return pd.DataFrame()
    source = st.session_state['capability_map'].get('sentiment')
    if source == 'MEMORY': return st.session_state['data_cache'].get('sentiment', pd.DataFrame())
    if source in FILES: return load_dataset(FILES[source])
    return st.session_state.get('data_cache', {}).get('sentiment', pd.DataFrame())

df = get_sentiment_data()
st.title(" Sentiment & Topic Engine")

if df.empty:
    st.error(" No Sentiment Data Found.")
    st.info("Please go to Home and load the Airline Reviews dataset.")
    st.stop()

# --- 3. CONFIGURATION (Only show if analysis hasn't run yet) ---
if 'sentiment_results' not in st.session_state:
    st.info(" Configure your analysis below.")
    
    col_text, col_viz = st.columns(2)
    
    with col_text:
        # A. Text Column Selection
        # We allow you to choose ANY string column (Header vs Body)
        text_candidates = df.select_dtypes(include=['object', 'string']).columns
        
        # Smart Default: Try to find 'body' or 'review', else 'header'
        default_idx = 0
        for i, c in enumerate(text_candidates):
            if any(x in c.lower() for x in ['body', 'content', 'text']): # Prioritize Body over Header
                default_idx = i
                break
        
        text_col = st.selectbox("1. Select Review Column (Text):", text_candidates, index=default_idx)
        st.caption("Choose the column containing the actual feedback.")

    with col_viz:
        # B. Classification Selection
        # Filter for Categorical columns (Low cardinality)
        # We pre-select one, but you can change it later in the dashboard
        cat_candidates = [c for c in df.columns if df[c].nunique() < 50 and c != text_col]
        default_cat = cat_candidates[0] if cat_candidates else "None"
        st.write("2. Analysis Mode: Full Corpus + Topic Extraction")
        st.write(f"*(You can group by '{default_cat}' after analysis)*")

    st.markdown("---")
    if st.button(" Run AI Analysis", type="primary"):
        with st.spinner(f"Analyzing '{text_col}'..."):
            try:
                analyzer = SentimentAnalyzer()
                # 1. Sentiment Score
                df_scored = analyzer.analyze_sentiment(df, text_col)
                # 2. Topic Modeling
                topics = analyzer.extract_topics(df_scored, 'Clean_Text', n_topics=5)
                # 3. Topic Assignment
                df_final = analyzer.get_topic_distribution(df_scored, 'Clean_Text')
                
                st.session_state['sentiment_results'] = df_final
                st.session_state['topics'] = topics
                st.rerun()
            except Exception as e:
                st.error(f"Analysis Failed: {str(e)}")
                st.stop()

# --- 4. DASHBOARD (Results Exist) ---
else:
    results = st.session_state['sentiment_results']
    topics = st.session_state.get('topics', {})
    
    # --- TOP METRICS ---
    k1, k2, k3, k4 = st.columns(4)
    pos = (results['Sentiment_Label'] == 'Positive').mean()
    neg = (results['Sentiment_Label'] == 'Negative').mean()
    k1.metric("😊 Positive", f"{pos:.1%}")
    k2.metric("😡 Negative", f"{neg:.1%}")
    k3.metric("Neutral", f"{(1-pos-neg):.1%}")
    k4.metric("Total Reviews", len(results))

    # --- CLASSIFICATION & SLICING ---
    st.markdown("###  Classification Analysis")
    
    # Dynamic Grouping
    slice_cols = [c for c in results.columns if results[c].nunique() < 50 and c not in ['Clean_Text', 'Sentiment_Label', 'Sentiment_Score', 'Topic_Label', 'Topic_ID']]
    
    if slice_cols:
        c_sel, c_chart = st.columns([1, 3])
        with c_sel:
            group_col = st.selectbox("Breakdown By:", slice_cols, index=0)
            st.caption("Change this to view sentiment across different categories.")
            
        with c_chart:
            # Stacked Bar Chart
            df_grouped = results.groupby([group_col, 'Sentiment_Label']).size().reset_index(name='Count')
            fig = px.bar(df_grouped, x=group_col, y='Count', color='Sentiment_Label', 
                         title=f"Sentiment by {group_col}",
                         color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#95a5a6'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No categorical columns found for grouping.")

    # --- TOPIC ANALYSIS ---
    st.markdown("---")
    st.markdown("###  Topic Intelligence")
    
    t1, t2 = st.columns([2, 1])
    with t1:
        if 'Topic_Label' in results.columns:
            # Topic vs Sentiment
            fig_topic = px.histogram(results[results['Topic_Label']!='Unknown'], x='Topic_Label', color='Sentiment_Label', 
                                     barmode='group', title="Sentiment per Topic",
                                     color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#95a5a6'})
            st.plotly_chart(fig_topic, use_container_width=True)
            
    with t2:
        st.write("**Identified Themes:**")
        for topic, keywords in topics.items():
            with st.expander(f" {topic}"):
                st.write(", ".join(keywords))

    # --- RAW DATA TABLE ---
    with st.expander(" View Classified Data"):
        st.dataframe(results)