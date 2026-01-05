import pandas as pd
import numpy as np

class NarrativeGenerator:
    """
    Turns Dataframes into Executive Summaries.
    Uses logic-based templates to simulate an 'AI Analyst'.
    """
    
    def generate_segmentation_narrative(self, df: pd.DataFrame, cluster_col: str, feature_cols: list) -> dict:
        """
        Analyzes clusters and writes a story for the top segments.
        """
        if df.empty: return {}
        
        # 1. Calculate Core Stats
        total_users = len(df)
        stats = df.groupby(cluster_col)[feature_cols].mean()
        counts = df[cluster_col].value_counts()
        
        # 2. Identify Key Clusters
        largest_cluster = counts.idxmax()
        smallest_cluster = counts.idxmin()
        
        narratives = {}
        
        # 3. Generate Text for Each Cluster
        for cluster_id in stats.index:
            size = counts[cluster_id]
            pct = (size / total_users) * 100
            avg_profile = stats.loc[cluster_id]
            
            # Dynamic Adjectives
            desc = []
            
            # Age Logic (if available)
            if 'Age' in avg_profile:
                age = avg_profile['Age']
                if age < 30: desc.append("younger")
                elif age > 50: desc.append("senior")
                else: desc.append("middle-aged")
                
            # Spend Logic (if available)
            # Check for various spend column names
            spend_col = next((c for c in avg_profile.index if 'spend' in c.lower() or 'monetary' in c.lower()), None)
            if spend_col:
                val = avg_profile[spend_col]
                global_mean = df[spend_col].mean()
                if val > global_mean * 1.2: desc.append("high-value")
                elif val < global_mean * 0.8: desc.append("budget-conscious")
                else: desc.append("average-spending")
            
            # Construct the Sentence
            adj_str = ", ".join(desc) if desc else "standard"
            
            summary = (
                f"**Segment Overview:** This group represents **{pct:.1f}%** of your customer base ({size} users). "
                f"They are predominantly **{adj_str}** customers."
            )
            
            # Actionable Recommendation (Simple Logic)
            if "high-value" in desc and "younger" in desc:
                action = "Strategy: Promote trending, premium items on social channels (Instagram/TikTok)."
            elif "high-value" in desc and "senior" in desc:
                action = "Strategy: Focus on loyalty rewards and high-touch customer service."
            elif "budget-conscious" in desc:
                action = "Strategy: Target with bulk discounts, bundles, or seasonal sales."
            else:
                action = "Strategy: Maintain engagement with regular newsletters."
                
            narratives[cluster_id] = f"{summary}  \n\n💡 *{action}*"
            
        return narratives

    def generate_churn_summary(self, risk_counts: dict) -> str:
        """ Writes a headline for the Churn Dashboard. """
        high_risk = risk_counts.get('High', 0)
        total = sum(risk_counts.values())
        rate = (high_risk / total) * 100 if total > 0 else 0
        
        if rate > 30:
            return f"⚠️ **Critical Alert:** {rate:.1f}% of your customers are High Risk. Immediate retention campaigns required."
        elif rate > 15:
            return f"⚠️ **Warning:** Churn risk is elevated ({rate:.1f}%). Investigate 'Contract' and 'Support' issues."
        else:
            return f"✅ **Healthy:** Only {rate:.1f}% of customers are High Risk. Focus on acquisition."