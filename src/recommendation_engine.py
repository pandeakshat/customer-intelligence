import pandas as pd
import numpy as np

def generate_business_logic(df_summary: pd.DataFrame, context: str = "demographic") -> pd.DataFrame:
    """
    Main Entry Point. Enriches the summary data with Business Logic.
    
    Args:
        df_summary: The dataframe containing cluster centers (means) or top metrics.
        context: The 'flavor' of analysis ('demographic', 'rfm', 'churn', 'geo').
    """
    
    # --- FIX: ROBUSTNESS CHECK ---
    # If input is a Series (e.g. from a simple .mean() operation), convert to DataFrame
    # This prevents the "axis" error in apply()
    if isinstance(df_summary, pd.Series):
        df_summary = df_summary.to_frame(name='Value').reset_index()
        # Rename common index columns if needed
        if 'index' in df_summary.columns: 
            df_summary = df_summary.rename(columns={'index': 'Feature'})

    # If input is DataFrame but the relevant info is in the Index
    if isinstance(df_summary, pd.DataFrame):
        if df_summary.index.name is not None and df_summary.index.name != list(df_summary.index.names):
            df_summary = df_summary.reset_index()

    # Normalize context string
    ctx = context.lower()
    
    # --- DISPATCHER ---
    # Routes the data to the specific logic handler
    if ctx == 'demographic':
        return _apply_logic(df_summary, _get_demographic_persona)
    elif ctx == 'rfm':
        return _apply_logic(df_summary, _get_rfm_persona)
    elif ctx == 'churn':
        return _apply_logic(df_summary, _get_churn_persona) 
    elif ctx == 'geo':
        return _apply_logic(df_summary, _get_geo_persona)
    else:
        # Fallback: Return empty dataframe or as-is
        return pd.DataFrame()

# --- CORE PROCESSING HELPER ---
def _apply_logic(df, logic_function):
    """Applies a specific logic function row-by-row."""
    if df.empty: return pd.DataFrame()
    
    try:
        results = df.apply(logic_function, axis=1)
        
        # Expand the results (which are dicts) into columns
        # We assume the logic function returns a dict like {'Persona': ..., 'Action': ...}
        logic_df = pd.DataFrame(results.tolist(), index=df.index)
        
        # Return just the logic columns (the calling app usually displays them alongside the original data)
        return pd.concat([df, logic_df], axis=1)
    except Exception as e:
        print(f"Recommendation Engine Error: {e}")
        # Return safe fallback
        return pd.DataFrame({'Persona': ['Unknown'], 'Next Best Action': ['Analyze Manually']}, index=df.index)

# ==========================================
# MODULE 1: DEMOGRAPHIC LOGIC
# ==========================================
def _get_demographic_persona(row):
    traits = []
    
    # Safely get values with defaults
    fam_size = row.get('Family_Size', 0)
    age = row.get('Age', 0)
    spend = row.get('Spending_Score_Num', 0)
    
    # 1. Family Logic
    if fam_size <= 1.5: traits.append("Solo")
    elif fam_size <= 3.5: traits.append("Small Fam")
    else: traits.append("Large Fam")
        
    # 2. Age Logic
    if age < 30: traits.append("Gen Z")
    elif age < 50: traits.append("Pro")
    else: traits.append("Senior")
        
    # 3. Spend Logic
    if spend > 2.5: traits.append("Spender")
    elif spend < 1.5: traits.append("Saver")
    
    persona_name = " / ".join(traits)
    
    # 4. Strategy Mapping
    tactic = "Standard"
    action = "Newsletter"
    
    if "Solo" in persona_name and "Gen Z" in persona_name:
        tactic = "Trend & FOMO"
        action = "Push: 'Trending Now'"
    elif "Fam" in persona_name and "Saver" in persona_name:
        tactic = "Value Bundles"
        action = "Email: 'Buy 2 Get 1'"
    elif "Senior" in persona_name:
        tactic = "Trust & Loyalty"
        action = "Mail: Loyalty Club"
    elif "Spender" in persona_name:
        tactic = "Exclusivity"
        action = "Invite: VIP Early Access"
        
    return {
        "Persona": persona_name,
        "Strategy": tactic,
        "Next Best Action": action
    }

# ==========================================
# MODULE 2: RFM LOGIC
# ==========================================
def _get_rfm_persona(row):
    persona = "Standard"
    tactic = "Maintain"
    action = "Weekly Update"
    
    # Get values (assuming standard RFM columns exist)
    recency = row.get('Recency', 0)
    monetary = row.get('Monetary', 0)
    
    is_recent = recency < 30
    is_lost = recency > 90
    is_big_spender = monetary > 1000 
    
    if is_recent and is_big_spender:
        persona = "Champion"
        tactic = "Reward"
        action = "VIP Tier Upgrade"
    elif is_lost and is_big_spender:
        persona = "At-Risk Whale"
        tactic = "Win-Back"
        action = "Personal Call + Gift"
    elif is_recent:
        persona = "New/Active"
        tactic = "Nurture"
        action = "Cross-sell Recommendation"
    elif is_lost:
        persona = "Hibernating"
        tactic = "Re-engage"
        action = "Automated Drip Campaign"
        
    return {
        "Persona": persona,
        "Strategy": tactic,
        "Next Best Action": action
    }

# ==========================================
# MODULE 3: CHURN LOGIC
# ==========================================
def _get_churn_persona(row):
    """
    Logic for Churn Recommendations.
    """
    # Robustly try to find a probability/risk value
    risk = row.get('probability', 0)
    if 'Value' in row and isinstance(row['Value'], (int, float)):
        # If generic 'Value' column is small (0-1), assume it's probability
        if row['Value'] <= 1.0:
            risk = row['Value']
    
    if risk > 0.8:
        persona = "Flight Risk (Critical)"
        tactic = "Immediate Intervention"
        action = "Call: Offer 20% Retention Discount"
    elif risk > 0.5:
        persona = "At Risk"
        tactic = "Value Reinforcement"
        action = "Email: Highlight Premium Features"
    elif risk < 0.2:
        persona = "Loyalist"
        tactic = "Advocacy"
        action = "Ask for Referral"
    else:
        persona = "Neutral"
        tactic = "Monitor"
        action = "No Action"
        
    return {
        "Persona": persona,
        "Strategy": tactic,
        "Next Best Action": action
    }

# ==========================================
# MODULE 4: GEOSPATIAL LOGIC
# ==========================================
def _get_geo_persona(row):
    """
    Logic for Geospatial Routes.
    Handles both 'Traffic/Rating' (Airline data) and Generic 'Value' (Maps).
    """
    # 1. Try Specific Airline Columns
    traffic = row.get('Traffic', None)
    rating = row.get('OverallRating', None)
    
    # 2. Try Generic Columns (from simple Map aggregations)
    val = row.get('Value', 0)
    
    persona = "Key Location"
    tactic = "Maintain"
    action = "Monitor Metrics"

    # LOGIC A: Detailed Airline Route Logic
    if traffic is not None and rating is not None:
        if traffic > 50:
            if rating < 4:
                persona = "High Vol / Low Sat"
                tactic = "Operational Fix"
                action = "Audit Ground Staff & Delays"
            else:
                persona = "Flagship Route"
                tactic = "Maximize Yield"
                action = "Increase Business Class Prices"
        elif traffic < 10:
            persona = "Niche / Low Volume"
            tactic = "Efficiency Check"
            action = "Evaluate Route Profitability"
        else:
            persona = "Standard Route"
            tactic = "Growth"
            action = "Seasonal Promotion"

    # LOGIC B: Generic Metric Logic (e.g. Spend or Sentiment Score)
    elif val != 0:
        if val > 1000: # Assuming High Spend
            persona = "Power Region"
            tactic = "Logistics Priority"
            action = "Scale support operations"
        elif val < 0: # Assuming Negative Sentiment
            persona = "Detractor Zone"
            tactic = "Quality Control"
            action = "Investigate local reviews"
            
    return {
        "Persona": persona,
        "Strategy": tactic,
        "Next Best Action": action
    }