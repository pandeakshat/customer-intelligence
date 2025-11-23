import pandas as pd

def generate_business_logic(df_summary: pd.DataFrame, context: str = "demographic") -> pd.DataFrame:
    """
    Main Entry Point. Enriches the summary data with Business Logic.
    
    Args:
        df_summary: The dataframe containing cluster centers (means).
        context: The 'flavor' of analysis ('demographic', 'rfm', 'churn', etc.)
    """
    
    # Normalize context string
    ctx = context.lower()
    
    # --- DISPATCHER ---
    # Routes the data to the specific logic handler
    if ctx == 'demographic':
        return _apply_logic(df_summary, _get_demographic_persona)
    elif ctx == 'rfm':
        return _apply_logic(df_summary, _get_rfm_persona)
    elif ctx == 'churn':
        # Placeholder for your future Churn Recommendations
        return _apply_logic(df_summary, _get_churn_persona) 
    else:
        # Fallback: Return as-is if no logic exists
        return df_summary

# --- CORE PROCESSING HELPER ---
def _apply_logic(df, logic_function):
    """Applies a specific logic function row-by-row."""
    results = df.apply(logic_function, axis=1)
    
    # Expand the results (which are dicts) into columns
    logic_df = pd.DataFrame(results.tolist(), index=df.index)
    
    # Combine original stats with new logic
    return pd.concat([df, logic_df], axis=1)

# --- MODULE 1: DEMOGRAPHIC LOGIC ---
def _get_demographic_persona(row):
    traits = []
    
    # 1. Family Logic
    if row['Family_Size'] <= 1.5: traits.append("Solo")
    elif row['Family_Size'] <= 3.5: traits.append("Small Fam")
    else: traits.append("Large Fam")
        
    # 2. Age Logic
    if row['Age'] < 30: traits.append("Gen Z")
    elif row['Age'] < 50: traits.append("Pro")
    else: traits.append("Senior")
        
    # 3. Spend Logic
    if row['Spending_Score_Num'] > 2.5: traits.append("Spender")
    elif row['Spending_Score_Num'] < 1.5: traits.append("Saver")
    
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

# --- MODULE 2: RFM LOGIC ---
def _get_rfm_persona(row):
    # Logic based on relative strength
    # Note: In a real app, use quantiles here, not hardcoded numbers
    
    persona = "Standard"
    tactic = "Maintain"
    action = "Weekly Update"
    
    is_recent = row['Recency'] < 30
    is_lost = row['Recency'] > 90
    is_big_spender = row['Monetary'] > 1000 # Placeholder threshold
    
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

# --- MODULE 3: CHURN LOGIC (Future Placeholder) ---
def _get_churn_persona(row):
    # Example for when you build Churn Recommendations
    risk = row.get('probability', 0)
    return {
        "Persona": "High Risk" if risk > 0.7 else "Safe",
        "Strategy": "Intervention" if risk > 0.7 else "Upsell",
        "Next Best Action": "Call" if risk > 0.7 else "None"
    }


# ... [Previous imports and dispatcher remain the same] ...

# --- MODULE 3: CHURN LOGIC ---
def _get_churn_persona(row):
    """
    Logic for Churn Recommendations.
    Input row expects: {'probability': 0.85}
    """
    risk = row.get('probability', 0)
    
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



# [Inside src/recommendation_engine.py]

# 1. UPDATE DISPATCHER
def generate_business_logic(df_summary: pd.DataFrame, context: str = "demographic") -> pd.DataFrame:
    ctx = context.lower()
    if ctx == 'demographic':
        return _apply_logic(df_summary, _get_demographic_persona)
    elif ctx == 'rfm':
        return _apply_logic(df_summary, _get_rfm_persona)
    elif ctx == 'churn':
        return _apply_logic(df_summary, _get_churn_persona)
    elif ctx == 'geo':  # <--- NEW
        return _apply_logic(df_summary, _get_geo_persona)
    else:
        return df_summary

# 2. ADD GEO LOGIC MODULE
def _get_geo_persona(row):
    """
    Logic for Geospatial Routes.
    Input row expects: {'Traffic': int, 'OverallRating': float}
    """
    traffic = row.get('Traffic', 0)
    rating = row.get('OverallRating', 0)
    
    # Route Classification Logic
    if traffic > 50:
        if rating < 4:
            persona = "High Volume / Low Satisfaction"
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
        
    return {
        "Persona": persona,
        "Strategy": tactic,
        "Next Best Action": action
    }