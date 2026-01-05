import pandas as pd
import xgboost as xgb
import joblib
import os
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class ChurnPredictor:
    """
    Production-ready pipeline for Churn Prediction.
    """
    
    def __init__(self):
        self.model = None
        self.le_dict = {} 
        self.model_path = "models/churn_model.pkl"
        self.encoder_path = "models/churn_encoders.pkl"
        
        os.makedirs("models", exist_ok=True)
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            try:
                self.model = joblib.load(self.model_path)
                self.le_dict = joblib.load(self.encoder_path)
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        for col in ['TotalCharges', 'MonthlyCharges', 'Tenure', 'tenure']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        return df_clean

    def train(self, df: pd.DataFrame):
        df_clean = self._clean_data(df)
        
        target_col = 'Churn'
        if target_col not in df_clean.columns: 
            return {"error": "Target column 'Churn' not found."}
        
        X = df_clean.drop(columns=['Churn', 'customerID', 'customer_id', 'id'], errors='ignore')
        y = df_clean[target_col].map({'Yes': 1, 'No': 0, 'TRUE': 1, 'FALSE': 0, 1: 1, 0: 0}).fillna(0)
        
        X_enc = X.copy()
        self.le_dict = {} 
        
        for col in X_enc.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_enc[col].astype(str))
            self.le_dict[col] = le
            
        X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)
        
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1
        
        self.model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            scale_pos_weight=pos_weight, eval_metric='logloss', use_label_encoder=False
        )
        self.model.fit(X_train, y_train)
        
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.le_dict, self.encoder_path)
        
        return {"accuracy": self.model.score(X_test, y_test)}

    def predict(self, df: pd.DataFrame):
        if not self.model: return pd.DataFrame()
        
        df_clean = self._clean_data(df)
        X = df_clean.drop(columns=['Churn', 'customerID', 'customer_id', 'id'], errors='ignore')
        
        booster = self.model.get_booster()
        expected_features = booster.feature_names
        
        X_aligned = pd.DataFrame(index=X.index)
        for feat in expected_features:
            if feat in X.columns:
                X_aligned[feat] = X[feat]
            else:
                X_aligned[feat] = 0 
                
        for col, le in self.le_dict.items():
            if col in X_aligned.columns:
                X_aligned[col] = X_aligned[col].astype(str).map(
                    lambda x: np.where(le.classes_ == x)[0][0] if x in le.classes_ else -1
                )
                
        preds = self.model.predict_proba(X_aligned)[:, 1]
        
        results = df_clean.copy()
        results['Churn Probability'] = preds
        results['Risk Group'] = pd.cut(preds, bins=[-0.1, 0.4, 0.7, 1.1], labels=['Low', 'Medium', 'High'])
        return results

    # --- THIS WAS MISSING ---
    def predict_single(self, row_dict: dict):
        """ Wrapper to predict for a single dictionary row. """
        if not self.model: return None
        # Convert dict to DataFrame
        df_single = pd.DataFrame([row_dict])
        # Use main predict logic
        res = self.predict(df_single)
        
        if not res.empty:
            return {
                'probability': res.iloc[0]['Churn Probability'],
                'risk_group': res.iloc[0]['Risk Group']
            }
        return None

    def get_average_customer(self, df: pd.DataFrame):
        df_clean = self._clean_data(df)
        avg_row = {}
        for col in df_clean.columns:
            if col in ['customerID', 'Churn']: continue
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                avg_row[col] = df_clean[col].median()
            else:
                avg_row[col] = df_clean[col].mode()[0]
        return pd.Series(avg_row)

    def simulate_churn(self, base_row: pd.Series, adjustments: dict) -> dict:
        if not self.model: return {}
        sim_data = base_row.copy()
        for k, v in adjustments.items(): sim_data[k] = v
        # Use the new helper
        return self.predict_single(sim_data.to_dict()) or {}

    def get_directional_importance(self, df: pd.DataFrame):
        if not self.model: return pd.DataFrame()
        booster = self.model.get_booster()
        imp_map = booster.get_score(importance_type='gain')
        
        imp_df = pd.DataFrame({'Feature': list(imp_map.keys()), 'Importance': list(imp_map.values())})
        imp_df = imp_df.sort_values('Importance', ascending=False).head(5)
        
        df_scored = self.predict(df)
        directions = []
        for feat in imp_df['Feature']:
            if feat in df_scored.columns and pd.api.types.is_numeric_dtype(df_scored[feat]):
                corr = df_scored[feat].corr(df_scored['Churn Probability'])
                direction = "Increases Risk 🔴" if corr > 0 else "Decreases Risk 🟢"
                directions.append(direction)
            else:
                directions.append("Key Risk Driver ⚠️")
        imp_df['Impact'] = directions
        return imp_df
        
    def recommend_retention_plan(self, df: pd.DataFrame, target_cols: list = None):
        df_clean = self._clean_data(df)
        recommendations = {}
        cols_to_check = target_cols if target_cols else df_clean.select_dtypes(include=['object']).columns.tolist()
        for col in cols_to_check:
            if col in ['customerID', 'Churn', 'Risk Group', 'Churn Probability']: continue
            if col not in df_clean.columns: continue
            churn_num = df_clean['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0)
            stats = churn_num.groupby(df_clean[col]).mean().sort_values()
            if len(stats) > 1:
                recommendations[col] = {"best_option": stats.index[0], "churn_rate": stats.iloc[0]}
        return recommendations

    def get_shap_data(self, row_df):
        if not self.model: return None
        df_clean = self._clean_data(row_df)
        X = df_clean.drop(columns=['Churn', 'customerID', 'customer_id', 'id'], errors='ignore')
        
        booster = self.model.get_booster()
        X_aligned = pd.DataFrame(index=X.index)
        for feat in booster.feature_names:
            X_aligned[feat] = X[feat] if feat in X.columns else 0
        
        for col, le in self.le_dict.items():
            if col in X_aligned.columns:
                X_aligned[col] = X_aligned[col].astype(str).map(lambda x: np.where(le.classes_==x)[0][0] if x in le.classes_ else -1)
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(X_aligned)
        return shap_values[0]