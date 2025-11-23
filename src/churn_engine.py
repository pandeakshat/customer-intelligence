# src/churn_engine.py
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class ChurnPredictor:
    """
    Production-ready pipeline for Churn Prediction.
    Includes robust 'Self-Healing' data cleaning.
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.explainer = None
        
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ INTERNAL HELPER: Forces all numeric columns to be actual numbers. """
        df_clean = df.copy()
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)
        if 'MonthlyCharges' in df_clean.columns:
            df_clean['MonthlyCharges'] = pd.to_numeric(df_clean['MonthlyCharges'], errors='coerce').fillna(0)
        return df_clean

    def train(self, df: pd.DataFrame):
        """ Trains XGBoost and saves the pipeline. """
        df_clean = self._clean_data(df)
        X = df_clean.drop(columns=['Churn', 'customerID', 'customer_id', 'id'], errors='ignore')
        y = df_clean['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        num_transformer = SimpleImputer(strategy='mean')
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, numeric_features),
                ('cat', cat_transformer, categorical_features)
            ],
            verbose_feature_names_out=False
        )
        
        X_processed = self.preprocessor.fit_transform(X)
        
        try:
            self.feature_names = self.preprocessor.get_feature_names_out()
        except:
            self.feature_names = numeric_features + categorical_features
            
        pos_weight = (y == 0).sum() / (y == 1).sum()
        self.model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            scale_pos_weight=pos_weight, eval_metric='logloss'
        )
        self.model.fit(X_processed, y)
        self.explainer = shap.TreeExplainer(self.model)
        
        return {"accuracy": self.model.score(X_processed, y)}

    def predict(self, df: pd.DataFrame):
        """ Returns DataFrame with 'Probability' and 'Risk_Label'. """
        if not self.model: return pd.DataFrame()
        df_clean = self._clean_data(df)
        X = df_clean.drop(columns=['Churn', 'customerID', 'customer_id', 'id'], errors='ignore')
        X_proc = self.preprocessor.transform(X)
        probs = self.model.predict_proba(X_proc)[:, 1]
        
        results = df_clean.copy()
        results['Churn Probability'] = probs
        results['Risk Group'] = pd.cut(probs, bins=[-0.1, 0.4, 0.7, 1.1], labels=['Low', 'Medium', 'High'])
        return results

    def get_directional_importance(self, scored_df: pd.DataFrame):
        """ Determines if a feature Increases or Decreases risk. """
        if not self.model: return pd.DataFrame()
        df_clean = self._clean_data(scored_df)

        imp_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        X = df_clean.drop(columns=['Churn', 'customerID', 'Churn Probability', 'Risk Group'], errors='ignore')
        X_proc = self.preprocessor.transform(X)
        X_proc_df = pd.DataFrame(X_proc, columns=self.feature_names)
        X_proc_df['target_prob'] = df_clean['Churn Probability'].values
        
        directions = []
        for feat in imp_df['Feature']:
            corr = X_proc_df[feat].corr(X_proc_df['target_prob'])
            direction = "Increases Risk 🔴" if corr > 0 else "Decreases Risk 🟢"
            directions.append(direction)
        imp_df['Impact'] = directions
        return imp_df

    def recommend_retention_plan(self, df: pd.DataFrame, target_cols: list = None):
        """
        Identifies the 'Safest' options for categorical features.
        Args:
            target_cols: List of column names to analyze (optional).
        """
        df_clean = self._clean_data(df)
        recommendations = {}
        
        # If no specific columns asked, analyze all categorical ones
        if not target_cols:
            cols_to_check = df_clean.select_dtypes(include=['object']).columns.tolist()
        else:
            cols_to_check = target_cols
            
        for col in cols_to_check:
            if col in ['customerID', 'Churn', 'Risk Group', 'Churn Probability']: continue
            if col not in df_clean.columns: continue
            
            # Convert Yes/No to 1/0 for mean calculation
            churn_numeric = df_clean['Churn'].map({'Yes': 1, 'No': 0})
            stats = churn_numeric.groupby(df_clean[col]).mean().sort_values()
            
            if len(stats) > 1: # Only if variation exists
                best_option = stats.index[0]
                churn_rate = stats.iloc[0]
                recommendations[col] = {
                    "best_option": best_option,
                    "churn_rate": churn_rate
                }
        return recommendations
    
    def get_shap_data(self, row_df: pd.DataFrame):
        """ Used for the Waterfall plot. """
        row_clean = self._clean_data(row_df)
        X = row_clean.drop(columns=['Churn', 'customerID', 'customer_id', 'id'], errors='ignore')
        X_proc = self.preprocessor.transform(X)
        shap_values = self.explainer(X_proc)
        shap_values.feature_names = self.feature_names
        return shap_values[0]
    
    def get_average_customer(self, df: pd.DataFrame):
        """ Creates a 'Generic' customer row (Mode/Median). """
        df_clean = self._clean_data(df)
        avg_row = {}
        
        for col in df_clean.columns:
            if col in ['customerID', 'Churn']: continue
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                avg_row[col] = df_clean[col].median()
            else:
                avg_row[col] = df_clean[col].mode()[0]
        
        return pd.Series(avg_row)