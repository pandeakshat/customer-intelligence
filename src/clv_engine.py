import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class CLVPredictor:
    """
    Predicts Future Value (Monetary) using XGBoost Regression.
    Includes MLOps (Auto-Save/Load) for the Single Customer View.
    """
    def __init__(self):
        self.model = None
        self.le = LabelEncoder()
        self.model_path = "models/clv_model.pkl"
        
        # MLOps: Create folder/Load
        os.makedirs("models", exist_ok=True)
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except:
                pass

    def train(self, df: pd.DataFrame):
        # 1. Identify "Value" Columns
        target_col = None
        for c in ['TotalCharges', 'Monetary', 'Spend', 'Sales', 'Amount', 'LTV']:
            if c in df.columns:
                target_col = c
                break
        
        if not target_col:
            return {"error": "No monetary column (TotalCharges, Monetary, etc.) found."}

        # 2. Clean Data
        df_clean = df.copy()
        df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce').fillna(0)
        
        # 3. Prepare Features
        X = df_clean.drop(columns=['customerID', 'id', 'Churn', 'churn', target_col], errors='ignore')
        
        # Encode
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = self.le.fit_transform(X[col].astype(str))
            
        y = df_clean[target_col]
        
        # 4. Train
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
        self.model.fit(X, y)
        
        # MLOps: Save
        joblib.dump(self.model, self.model_path)
        
        return {"status": "success", "target": target_col}

    def predict_single(self, row_series):
        """ Used by Single Customer View """
        if not self.model: return 0.0
        
        try:
            # Create DataFrame from series
            df_single = pd.DataFrame([row_series])
            
            # Align features
            booster = self.model.get_booster()
            expected = booster.feature_names
            
            X = pd.DataFrame(index=[0])
            for feat in expected:
                X[feat] = df_single[feat] if feat in df_single.columns else 0
            
            # Encode
            for col in X.select_dtypes(include=['object']).columns:
                # Simple hash encoding for safety in single prediction
                X[col] = self.le.fit_transform(X[col].astype(str))
                
            pred = self.model.predict(X)[0]
            return float(pred)
        except:
            return 0.0