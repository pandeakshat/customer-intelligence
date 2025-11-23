import streamlit as st
import pandas as pd
import plotly.express as px

from components.navigation import sidebar_menu
from src.geo_engine import GeoAnalyzer
from src.data_loader import load_dataset
from src.config import FILES

st.set_page_config(page_title="Geospatial Intelligence", layout="wide")
sidebar_menu()

if 'geo_bot' not in st.session_state: st.session_state['geo_bot'] = GeoAnalyzer()
bot = st.session_state['geo_bot']

def get_geo_data():
    if 'capability_map' not in st.session_state: return pd.DataFrame()
    source = st.session_state['capability_map'].get('geo')
    if source == 'MEMORY': return st.session_state['data_cache'].get('geo')
    return load_dataset(FILES.get(source)) if source else pd.DataFrame()

st.title("🗺️ Geospatial Intelligence")

# 1. LOAD
flags = st.session_state.get('flags', {})
if not flags.get('geo'):
    st.error("❌ No Geospatial capabilities detected.")
    st.stop()

df = get_geo_data()
meta = st.session_state.get('meta', {}).get('geo', {})
parent_context = meta.get('parent_context', 'unknown')
# If we didn't save it in meta (legacy), try to guess
if 'location_col' not in meta:
    # Fallback guess
    loc_col = next((c for c in df.columns if any(x in c.lower() for x in ['route', 'country', 'region', 'city'])), 'Route')
else:
    loc_col = meta['location_col']

# 2. PROCESS
with st.spinner(f"Processing Location Data ({loc_col})..."):
    map_df = bot.analyze_location_data(df, loc_col)

if map_df.empty:
    st.warning("Could not parse location data.")
    st.stop()

# 3. CONTEXTUAL DASHBOARD
st.markdown(f"### 🌍 Analysis by {loc_col} (Source: {parent_context.title()})")

tab_map, tab_data = st.tabs(["Map View", "Data View"])

with tab_map:
    # LOGIC: SWITCH PLOT BASED ON CONTEXT
    
    if parent_context == 'churn':
        # Show Churn Rate by Location
        st.caption("Visualizing **Churn Rate** by Location")
        if 'Churn' in map_df.columns:
            # Convert Yes/No
            map_df['Churn_Num'] = map_df['Churn'].map({'Yes': 1, 'No': 0, 1:1, 0:0})
            agg = map_df.groupby('Mapped_Location')['Churn_Num'].mean().reset_index()
            agg.columns = ['Location', 'Churn Rate']
            
            if 'lat' in map_df.columns: # Point Data
                # Need to re-merge lat/lon if we aggregated. 
                pass 
            
            # Choropleth works for countries. Scatter for cities/routes.
            if 'Geo_Type' in map_df.columns and map_df['Geo_Type'].iloc[0] == 'Region':
                 fig = px.choropleth(agg, locations='Location', locationmode='country names', 
                                    color='Churn Rate', color_continuous_scale='RdBu_r',
                                    title="Global Churn Risk")
            else:
                 # Fallback to scatter if we have lat/lon (handled in sentiment section better)
                 # Or just bar chart if no coords
                 fig = px.bar(agg, x='Location', y='Churn Rate')
                 
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Context is Churn, but 'Churn' column missing?")

    elif parent_context == 'sentiment':
        # Show Sentiment Score by Location
        st.caption("Visualizing **Average Sentiment** by Location")
        
        if 'Sentiment_Score' in map_df.columns:
            metric = 'Sentiment_Score'
            title = "Average Sentiment"
        elif 'OverallRating' in map_df.columns:
            metric = 'OverallRating'
            title = "Average Rating"
        else:
            metric = 'count'
            title = "Volume"
            
        if metric != 'count':
            agg = map_df.groupby('Mapped_Location')[metric].mean().reset_index()
        else:
            agg = map_df['Mapped_Location'].value_counts().reset_index()
            agg.columns = ['Mapped_Location', 'count']
            
        # Plot
        if 'lat' in map_df.columns: # Routes/Cities
            # We need to merge lat/lon back to agg
            coords = map_df.groupby('Mapped_Location')[['lat', 'lon']].first().reset_index()
            plot_data = agg.merge(coords, on='Mapped_Location')
            
            fig = px.scatter_geo(plot_data, lat='lat', lon='lon', size=metric, color=metric,
                                 title=f"{title} by Hub", projection="natural earth")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Countries
            fig = px.choropleth(agg, locations='Mapped_Location', locationmode='country names',
                                color=metric, title=f"Global {title}")
            st.plotly_chart(fig, use_container_width=True)

    elif parent_context == 'segmentation':
        st.caption("Visualizing **Dominant Segment** by Location")
        if 'Cluster_Label' in map_df.columns:
            st.info("Segment visualization requires Cluster Labels (Run Segmentation first).")
        else:
            st.info("This dataset hasn't been segmented yet. Go to Segmentation module first.")

    else:
        # Generic / Fallback (Just Count)
        st.caption("Visualizing **Traffic Volume**")
        counts = map_df['Mapped_Location'].value_counts().reset_index()
        counts.columns = ['Location', 'Volume']
        
        if 'lat' in map_df.columns:
             coords = map_df.groupby('Mapped_Location')[['lat', 'lon']].first().reset_index()
             plot_data = counts.merge(coords, left_on='Location', right_on='Mapped_Location')
             fig = px.scatter_geo(plot_data, lat='lat', lon='lon', size='Volume', 
                                  title="Traffic Volume")
             st.plotly_chart(fig, use_container_width=True)
        else:
             fig = px.choropleth(counts, locations='Location', locationmode='country names',
                                 color='Volume', title="Traffic Volume")
             st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.dataframe(map_df.head(50))