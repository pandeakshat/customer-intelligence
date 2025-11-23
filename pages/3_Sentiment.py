import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- IMPORTS ---
from components.navigation import sidebar_menu
from src.sentiment_engine import SentimentAnalyzer
from src.data_loader import load_dataset
from src.config import FILES

st.set_page_config(page_title="Voice of Customer", layout="wide")
sidebar_menu()

# --- SETUP ---
if 'sentiment_bot' not in st.session_state:
    st.session_state['sentiment_bot'] = SentimentAnalyzer()
    
bot = st.session_state['sentiment_bot']

# --- HELPER: GET DATA ---
def get_sentiment_data():
    if 'capability_map' not in st.session_state: return pd.DataFrame()
    source = st.session_state['capability_map'].get('sentiment')
    if source == 'MEMORY': return st.session_state['data_cache'].get('sentiment')
    return load_dataset(FILES.get(source)) if source else pd.DataFrame()

# --- MAIN ---
st.title("💬 Voice of Customer (NLP)")

# 1. LOAD
flags = st.session_state.get('flags', {})
if not flags.get('sentiment'):
    st.error("❌ No Sentiment Data active.")
    st.stop()

df = get_sentiment_data()

# Identify Text Column
text_col = None
possible_cols = ['ReviewBody', 'ReviewText', 'text', 'comment', 'content']
for c in possible_cols:
    for col in df.columns:
        if c.lower() in col.lower():
            text_col = col
            break
    if text_col: break

if not text_col:
    st.error("Could not auto-detect a text column.")
    st.stop()

# 2. PROCESS (Run NLP - The 3-Pass Method)
if 'sentiment_processed' not in st.session_state:
    with st.spinner("Analyzing Sentiment & Topics..."):
        # A. Run Sentiment Scoring (VADER)
        processed_df = bot.analyze_sentiment(df, text_col)
        
        # B. Pass 1: Global Topics (Assigns Topic ID to every row)
        topics_global = bot.extract_topics(processed_df, text_col)
        processed_df = bot.get_topic_distribution(processed_df, text_col)
        
        # C. Pass 2: Positive Topics Only
        pos_df = processed_df[processed_df['Sentiment_Label'] == 'Positive']
        if not pos_df.empty:
            topics_pos = bot.extract_topics(pos_df, 'Clean_Text')
        else:
            topics_pos = {}

        # D. Pass 3: Negative Topics Only
        neg_df = processed_df[processed_df['Sentiment_Label'] == 'Negative']
        if not neg_df.empty:
            topics_neg = bot.extract_topics(neg_df, 'Clean_Text')
        else:
            topics_neg = {}
        
        # Store results
        st.session_state['sentiment_processed'] = processed_df
        st.session_state['topics_global'] = topics_global
        st.session_state['topics_pos'] = topics_pos
        st.session_state['topics_neg'] = topics_neg
else:
    processed_df = st.session_state['sentiment_processed']
    topics_global = st.session_state['topics_global']
    topics_pos = st.session_state['topics_pos']
    topics_neg = st.session_state['topics_neg']

# 3. DASHBOARD
tab_overview, tab_topics, tab_factors = st.tabs(["📊 Sentiment Overview", "☁️ Topic Themes", "🔍 Satisfaction Drivers"])

# --- TAB 1: OVERVIEW ---
with tab_overview:
    k1, k2, k3 = st.columns(3)
    avg_score = processed_df['Sentiment_Score'].mean()
    pos_pct = (processed_df['Sentiment_Label'] == 'Positive').mean()
    neg_pct = (processed_df['Sentiment_Label'] == 'Negative').mean()
    
    k1.metric("Average Sentiment", f"{avg_score:.2f}", "-1 to +1 scale")
    k2.metric("Positive Reviews", f"{pos_pct:.1%}")
    k3.metric("Negative Reviews", f"{neg_pct:.1%}", delta="Attention" if neg_pct > 0.2 else "Healthy", delta_color="inverse")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(processed_df, names='Sentiment_Label', color='Sentiment_Label',
                         color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#95a5a6', 'Negative':'#e74c3c'})
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        st.subheader("Sentiment Over Time")
        date_col = next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'timestamp'])), None)
        if date_col:
            processed_df[date_col] = pd.to_datetime(processed_df[date_col], errors='coerce')
            daily = processed_df.dropna(subset=[date_col]).set_index(date_col).resample('ME')['Sentiment_Score'].mean()
            st.line_chart(daily)
        else:
            st.info("No Date column found.")

# --- TAB 2: TOPICS (UPDATED) ---
with tab_topics:
    st.subheader("What are customers talking about?")
    
    # Selection for Visualization
    view_mode = st.radio("View Topics For:", ["Global (All)", "Positive Only", "Negative Only"], horizontal=True)
    
    col_list, col_cloud = st.columns([1, 2])
    
    # DYNAMIC CONTENT BASED ON SELECTION
    if view_mode == "Global (All)":
        active_topics = topics_global
        subset_df = processed_df
    elif view_mode == "Positive Only":
        active_topics = topics_pos
        subset_df = processed_df[processed_df['Sentiment_Label'] == 'Positive']
    else:
        active_topics = topics_neg
        subset_df = processed_df[processed_df['Sentiment_Label'] == 'Negative']

    with col_list:
        st.write(f"### {view_mode} Themes")
        if not active_topics:
            st.warning("No topics found for this category.")
        else:
            for topic, words in active_topics.items():
                with st.expander(f"📌 {topic}", expanded=True):
                    st.caption(", ".join(words[:6]))

    with col_cloud:
        st.write("### Word Cloud")
        if not subset_df.empty:
            # Join all text for the selected sentiment/group
            text_corpus = " ".join(subset_df['Clean_Text'].dropna().astype(str).tolist())
            
            if text_corpus.strip():
                # Generate Cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_corpus)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.warning("Not enough text.")
        else:
            st.warning("No data available for this view.")

# --- TAB 3: FACTORS ---
with tab_factors:
    st.subheader("Correlation Analysis")
    rating_cols = [c for c in df.select_dtypes(include=np.number).columns if 'rating' in c.lower() or 'score' in c.lower() or c in ['SeatComfort', 'CabinStaffService', 'Food&Beverages', 'ValueForMoney']]
    
    if len(rating_cols) > 1:
        corr = df[rating_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("💡 **Tip:** Correlations close to 1.0 (Red) indicate drivers of satisfaction.")
    else:
        st.warning("Not enough numeric rating columns found.")