import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class SegmentationEngine:
    """
    Unified Segmentation Engine with Rule Extraction & Smart Labeling.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.mode = "Unknown"
        self.features = []

    def run_segmentation_model(self, df: pd.DataFrame, k: int = 4) -> dict:
        """
        Main pipeline: Mode Detection -> Clustering -> Labeling -> Rules.
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
        # We check for standard RFM columns OR raw transaction columns
        elif {'Recency', 'Frequency', 'Monetary'}.issubset(model_df.columns):
             output["mode"] = "RFM"
             output["features"] = ['Recency', 'Frequency', 'Monetary']
             X = model_df[output["features"]]
             
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
            # Fallback for generic data
            nums = model_df.select_dtypes(include=np.number).columns.tolist()
            if len(nums) >= 2:
                 output["mode"] = "Generic"
                 output["features"] = nums[:3] # Take top 3 numeric cols
                 X = model_df[output["features"]].dropna()
            else:
                return output # Cannot run

        # --- 2. CLUSTERING (The Math) ---
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model_df['Cluster'] = self.model.fit_predict(X_scaled)
        
        # --- 3. SMART LABELING (The Meaning) ---
        model_df['Cluster_Label'] = self._generate_smart_labels(model_df, output["mode"], output["features"])
        
        # --- 4. RULE EXTRACTION (The Logic) ---
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree.fit(X, model_df['Cluster']) 
        
        # Save state
        self.mode = output["mode"]
        self.features = output["features"]
        
        output["dt_model"] = tree
        output["dt_features"] = output["features"]
        output["data"] = model_df
        
        return output

    def get_segment_for_customer(self, row: pd.Series):
        """
        Retrieves the segment for a single customer row.
        Priority: 1. Existing Label, 2. Manual Calculation (if impossible, return Unknown)
        """
        # 1. Look for existing calculated labels
        if 'Cluster_Label' in row:
            return 0, row['Cluster_Label']
        if 'Segment' in row:
            return 0, row['Segment']
            
        # 2. Heuristic Fallback (if no model ran, just guess based on rules)
        # This prevents "Unknown" from showing up if the data is obvious
        if 'Monetary' in row and 'Recency' in row:
            mon = pd.to_numeric(row['Monetary'], errors='coerce')
            rec = pd.to_numeric(row['Recency'], errors='coerce')
            if mon > 1000 and rec < 30: return 1, "Champion (Heuristic)"
            if rec > 90: return 2, "Lost (Heuristic)"
            return 3, "Standard (Heuristic)"
            
        return -1, "Unsegmented"

    def _generate_smart_labels(self, df, mode, features):
        """ Analyzes cluster centers and assigns descriptive names. """
        labels = []
        summary = df.groupby('Cluster')[features].mean()
        
        for cluster_id, row in summary.iterrows():
            name_parts = []
            
            if mode == "Demographic":
                if row['Age'] < 30: name_parts.append("Young")
                elif row['Age'] < 50: name_parts.append("Mid-Age")
                else: name_parts.append("Senior")
                
                if row.get('Spending_Score_Num', 0) < 1.5: name_parts.append("Saver")
                elif row.get('Spending_Score_Num', 0) > 2.5: name_parts.append("Spender")
                else: name_parts.append("Standard")
                
                if row.get('Family_Size', 0) > 3.5: name_parts.append("(Fam)")
                
            elif mode == "RFM":
                if row['Recency'] < 30: name_parts.append("Active")
                elif row['Recency'] > 90: name_parts.append("Lost")
                else: name_parts.append("Regular")
                
                # Dynamic Quantiles for labels
                mon_high = df['Monetary'].quantile(0.75)
                mon_low = df['Monetary'].quantile(0.25)
                
                if row['Monetary'] > mon_high: name_parts.append("Whale")
                elif row['Monetary'] < mon_low: name_parts.append("LowVal")
                else: name_parts.append("MidVal")

            else:
                name_parts.append(f"Group {cluster_id}")

            final_name = f"{' '.join(name_parts)}"
            labels.append(final_name)
            
        return df['Cluster'].map(dict(enumerate(labels)))

    def suggest_optimal_k(self, df):
        """ Calculates Silhouette Score for k=2 to 6. """
        if df.empty: return pd.DataFrame()

        # Feature Selection
        if {'Age', 'Spending_Score_Num'}.issubset(df.columns):
            X = df[['Age', 'Spending_Score_Num', 'Family_Size']].dropna()
        elif {'Recency', 'Monetary'}.issubset(df.columns):
             X = df[['Recency', 'Frequency', 'Monetary']].dropna()
        else:
            X = df.select_dtypes(include=[np.number]).dropna()

        if X.empty: return pd.DataFrame()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        scores = []
        for k in range(2, 7):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            scores.append({"k": k, "score": score})
            
        return pd.DataFrame(scores).set_index('k')