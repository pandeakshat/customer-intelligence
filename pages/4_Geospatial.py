import streamlit as st
import pandas as pd
import plotly.express as px
from components.navigation import sidebar_menu
from src.geo_engine import GeoAnalyzer
from src.recommendation_engine import generate_business_logic
from src.data_loader import load_dataset
from src.config import FILES

# Clean state
if 'geo_engine' in st.session_state and not hasattr(st.session_state['geo_engine'], 'analyze_location_data'):
    del st.session_state['geo_engine']

st.set_page_config(page_title="Geospatial Intelligence", layout="wide")
sidebar_menu()

def get_geo_data():
    if 'capability_map' not in st.session_state: return pd.DataFrame()
    source = st.session_state['capability_map'].get('geo')
    if source == 'MEMORY': return st.session_state.get('data_cache', {}).get('geo', pd.DataFrame())
    if source in FILES: return load_dataset(FILES[source])
    return st.session_state.get('data_cache', {}).get('geo', pd.DataFrame())

st.title(" Geospatial Intelligence")
df = get_geo_data()

if df.empty or not st.session_state.get('flags', {}).get('geo'):
    st.error(" No Geospatial Data Active.")
    st.stop()

if 'geo_engine' not in st.session_state: st.session_state['geo_engine'] = GeoAnalyzer()
geo_bot = st.session_state['geo_engine']

# Find Location Column
meta = st.session_state.get('meta', {}).get('geo', {})
loc_col = meta.get('location_col', None)
if not loc_col:
    possible = [c for c in df.columns if any(k in c.lower() for k in ['location','route','city','country'])]
    loc_col = possible[0] if possible else df.columns[0]

st.info(f" Analyzing geography from: **{loc_col}**")

# Run Engine (Fast Mode)
status_container = st.empty()
def update_progress(pct, msg):
    if 'deep_scan_active' in st.session_state: status_container.progress(pct, text=msg)

if 'geo_processed_df' not in st.session_state:
    with st.spinner(" Rapid Matching..."):
        geo_df = geo_bot.analyze_location_data(df, loc_col, use_api=False)
        st.session_state['geo_processed_df'] = geo_df
else:
    geo_df = st.session_state['geo_processed_df']

# --- METRICS & UNMATCHED LOGIC ---
raw_col = df[loc_col].astype(str).replace('nan', '').str.strip()
empty_count = (raw_col == '').sum()
unmatched_geo = geo_df[geo_df['lat'].isna()]
unmatched_count = len(unmatched_geo) - empty_count
if unmatched_count < 0: unmatched_count = 0
success_count = len(geo_df.dropna(subset=['lat']))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Rows", len(df))
m2.metric("Mapped Successfully", success_count)
m3.metric("Unmatched (Unknown City)", unmatched_count)
m4.metric("Empty / NaN", empty_count)

if unmatched_count > 0:
    st.warning(f" **{unmatched_count} locations** were not found.")
    with st.expander("🔎 View Unmatched Cities"):
        bad_rows = geo_df[geo_df['lat'].isna() & (geo_df['Mapped_Location'].str.len() > 0)]
        if not bad_rows.empty:
            st.dataframe(bad_rows['Mapped_Location'].value_counts().reset_index(name='Count'), use_container_width=True)

    if st.button(" Deep Scan (Use Online API)"):
        st.session_state['deep_scan_active'] = True
        with st.spinner("Connecting to API..."):
            geo_df = geo_bot.analyze_location_data(df, loc_col, use_api=True, progress_callback=update_progress)
            st.session_state['geo_processed_df'] = geo_df
            st.rerun()

# --- CONTEXT DETECTION (The Fix for "It doesn't know what it is") ---
# We look for other metrics in the dataset to color the map meaningfully
plot_df = geo_df.dropna(subset=['lat']).copy()
valid_cols = plot_df.columns.tolist()

# Priority 1: Sentiment (Red/Green Map)
sentiment_col = next((c for c in valid_cols if 'sentiment_score' in c.lower()), None)
# Priority 2: Churn Risk (Red/Green Map)
churn_col = next((c for c in valid_cols if 'churn probability' in c.lower()), None)
# Priority 3: Money/Spend (Bubble Size Map)
money_col = next((c for c in valid_cols if any(x in c.lower() for x in ['total', 'amount', 'sales', 'charge'])), None)

# Default: Just count frequency if nothing else
if 'Count' not in plot_df.columns:
    plot_df['Count'] = 1

# --- VISUALIZATION ---
tab1, tab2 = st.tabs([" Global Map", " Insights"])

with tab1:
    if not plot_df.empty:
        # Determine Visuals
        color_col = None
        size_col = None
        title_text = "Global Locations"
        
        if sentiment_col:
            color_col = sentiment_col
            title_text = f"Global Sentiment Map (Color = {sentiment_col})"
            st.caption("🟢 Green = Positive, 🔴 Red = Negative")
        elif churn_col:
            color_col = churn_col
            title_text = f"Churn Risk Map (Color = {churn_col})"
            st.caption("🔴 Red = High Risk, 🟢 Green = Safe")
        elif money_col:
            size_col = money_col
            title_text = f"Revenue Map (Size = {money_col})"
        else:
            # Fallback: Size by Frequency
            freq_map = plot_df['Mapped_Location'].value_counts()
            plot_df['Frequency'] = plot_df['Mapped_Location'].map(freq_map)
            size_col = 'Frequency'
            title_text = "Location Frequency Map"

        # Create Map
        if plot_df['Mapped_Location'].str.contains(' to |-', regex=True).any():
            # Route Map
            fig = px.scatter_geo(plot_df, lat='lat', lon='lon', hover_name='Mapped_Location', 
                                 size=size_col, color=color_col,
                                 projection="natural earth", title=title_text)
        else:
            # Scatter Map
            fig = px.scatter_geo(plot_df, lat='lat', lon='lon', hover_name='Mapped_Location',
                                 size=size_col, color=color_col,
                                 projection="natural earth", title=title_text)
            
        fig.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No valid coordinates to plot.")

with tab2:
    if not plot_df.empty:
        c1, c2 = st.columns([2,1])
        
        # Select what to analyze
        # Prefer the detected context columns, else all numbers
        defaults = [c for c in [sentiment_col, churn_col, money_col] if c]
        num_cols = plot_df.select_dtypes(include=['number']).columns.drop(['lat','lon'], errors='ignore')
        
        with c1:
            target = st.selectbox("Analyze Metric:", num_cols, index=list(num_cols).index(defaults[0]) if defaults and defaults[0] in num_cols else 0)
            
            # Show Top 10
            bar_data = plot_df.groupby('Mapped_Location')[target].mean().nlargest(10).reset_index()
            fig_bar = px.bar(bar_data, x='Mapped_Location', y=target, title=f"Top Locations by Avg {target}", color=target)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            st.subheader(" Location Strategy")
            # --- THE FIX FOR THE CRASH ---
            # We explicitly pass a DataFrame (reset_index) so the engine doesn't crash on Series
            top_3 = plot_df.groupby('Mapped_Location')[target].mean().nlargest(3).reset_index()
            
            # Pass context to engine
            context_mode = 'geo'
            if sentiment_col and target == sentiment_col: context_mode = 'geo' # Could refine this
            
            strat = generate_business_logic(top_3, context=context_mode)
            
            for _, row in strat.iterrows():
                with st.container(border=True):
                    loc = row.iloc[0] # First column is location name
                    # Find action column
                    action = row.get('Next Best Action', 'Analyze deeper.')
                    persona = row.get('Persona', 'Key Region')
                    
                    st.markdown(f"** {loc}**")
                    st.caption(f"{persona}")
                    st.info(f"{action}")