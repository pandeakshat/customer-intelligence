# src/segment_engine.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def run_segmentation_model(df: pd.DataFrame, k: int = 4) -> dict:
    """
    Unified Segmentation Engine with Rule Extraction & Smart Labeling.
    """
    model_df = df.copy()
    output = {"mode": "Unknown", "data": None, "features": []}
    
    # --- 1. MODE DETECTION & PREPROCESSING ---
    
    # MODE A: DEMOGRAPHIC
    if {'Age', 'Spending_Score', 'Family_Size'}.issubset(model_df.columns):
        output["mode"] = "Demographic"
        output["features"] = ['Age', 'Spending_Score_Num', 'Family_Size']
        
        # Preprocess Spending Score
        if model_df['Spending_Score'].dtype == 'O':
            mapper = {'Low': 1, 'Average': 2, 'High': 3}
            model_df['Spending_Score_Num'] = model_df['Spending_Score'].map(mapper).fillna(1)
        else:
            model_df['Spending_Score_Num'] = model_df['Spending_Score']

        model_df = model_df.dropna(subset=output["features"])
        X = model_df[output["features"]]

    # MODE B: RFM
    elif {'CustomerID', 'InvoiceDate', 'TotalAmount'}.issubset(model_df.columns):
        output["mode"] = "RFM"
        output["features"] = ['Recency', 'Frequency', 'Monetary']
        
        # RFM Aggregation Logic
        model_df['InvoiceDate'] = pd.to_datetime(model_df['InvoiceDate'])
        snapshot_date = model_df['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        rfm = model_df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'CustomerID': 'count',
            'TotalAmount': 'sum'
        }).rename(columns={'InvoiceDate': 'Recency', 'CustomerID': 'Frequency', 'TotalAmount': 'Monetary'})
        
        model_df = rfm
        X = model_df[output["features"]]
        
    else:
        raise ValueError("Dataset missing required columns for either Demographic or RFM analysis.")

    # --- 2. CLUSTERING (The Math) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    model_df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # --- 3. SMART LABELING (The Meaning) ---
    # Instead of "Segment 1", generate "Young Savers"
    model_df['Cluster_Label'] = generate_smart_labels(model_df, output["mode"], output["features"])
    
    # --- 4. RULE EXTRACTION (The Logic) ---
    # Train a Decision Tree to explain *why* the clusters look like this
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, model_df['Cluster']) # Fit on the Cluster IDs
    
    # Save the explanation models for the UI
    output["dt_model"] = tree
    output["dt_features"] = output["features"]
    output["data"] = model_df
    
    return output

def generate_smart_labels(df, mode, features):
    """
    Analyzes cluster centers and assigns descriptive names.
    """
    labels = []
    # Calculate the average profile for each cluster
    summary = df.groupby('Cluster')[features].mean()
    
    for cluster_id, row in summary.iterrows():
        name_parts = []
        
        if mode == "Demographic":
            # Age Logic
            if row['Age'] < 30: name_parts.append("Young")
            elif row['Age'] < 50: name_parts.append("Mid-Age")
            else: name_parts.append("Senior")
            
            # Spending Logic
            if row['Spending_Score_Num'] < 1.5: name_parts.append("Saver")
            elif row['Spending_Score_Num'] > 2.5: name_parts.append("Spender")
            else: name_parts.append("Standard")
            
            # Family Logic
            if row['Family_Size'] > 3.5: name_parts.append("(Fam)")
            
        elif mode == "RFM":
            # Recency Logic
            if row['Recency'] < 30: name_parts.append("Active")
            elif row['Recency'] > 90: name_parts.append("Lost")
            else: name_parts.append("Regular")
            
            # Monetary Logic
            if row['Monetary'] > df['Monetary'].quantile(0.75): name_parts.append("Whale")
            elif row['Monetary'] < df['Monetary'].quantile(0.25): name_parts.append("LowVal")
            else: name_parts.append("MidVal")

        # Join parts: e.g., "Young Spender (Fam)"
        final_name = f"{' '.join(name_parts)} (C{cluster_id})"
        labels.append(final_name)
        
    # Map the new names back to the dataframe
    return df['Cluster'].map(dict(enumerate(labels)))

def suggest_optimal_k(df):
    """
    Calculates the 'Silhouette Score' for k=2 to 6 to find the mathematical sweet spot.
    """
    # Quick check to ensure we have data
    if df.empty: return pd.DataFrame()

    # Re-select features based on available columns (Mini-logic repetition for standalone utility)
    if {'Age', 'Spending_Score_Num'}.issubset(df.columns):
        X = df[['Age', 'Spending_Score_Num', 'Family_Size']].dropna()
    elif {'Recency', 'Monetary'}.issubset(df.columns):
         X = df[['Recency', 'Frequency', 'Monetary']].dropna()
    else:
        # Fallback: Try to use only numeric columns
        X = df.select_dtypes(include=[np.number]).dropna()

    if X.empty: return pd.DataFrame()

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = []
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append({"k": k, "score": score})
        
    return pd.DataFrame(scores).set_index('k')