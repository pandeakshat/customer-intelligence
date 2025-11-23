import pandas as pd
import numpy as np
import re

# --- 1. VALIDATION HELPERS ---
def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)

def is_string_or_object(series):
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)

def is_datetime(series):
    return pd.api.types.is_datetime64_any_dtype(series)

# --- 2. CONFIGURATION: RULES ENGINE ---
VALIDATION_RULES = {
    "churn": {
        "type": "simple",
        "columns": {
            "Churn": {"pattern": r"(?i)^churn$|^target$|^exited$", "type": is_string_or_object},
            "Tenure": {"pattern": r"(?i)tenure|months?|duration", "type": is_numeric},
            "MonthlyCharges": {"pattern": r"(?i)monthly\s*[_.]?(charge|fee|bill|amt)", "type": is_numeric}
        }
    },
    "segmentation": {
        "type": "flavored",
        "flavors": {
            "demographic": {
                "Age": {"pattern": r"(?i)\bage\b", "type": is_numeric},
                "Spending_Score": {"pattern": r"(?i)spending\s*[_.]?score", "type": None},
                "Profession": {"pattern": r"(?i)profession|job|occupation", "type": is_string_or_object}
            },
            "rfm": {
                "CustomerID": {"pattern": r"(?i)(customer|client|user|account)\s*[_.]?(id|no|code|key)", "type": None},
                "InvoiceDate": {"pattern": r"(?i)(invoice|txn|transaction|purchase)\s*[_.]?date", "type": None},
                "TotalAmount": {"pattern": r"(?i)(total)?\s*[_.]?(amount|amt|price|value|spend|cost)", "type": is_numeric}
            }
        }
    },
    "sentiment": {
        "type": "simple",
        "columns": {
            "ReviewText": {"pattern": r"(?i)review|comment|feedback|body|text|content", "type": is_string_or_object}
        }
    },
    "geo": {
        "type": "simple",
        "columns": {
            "Location": {
                # Matches: Route, Country, Region, State, City, Zip, etc.
                "pattern": r"(?i)route|destination|flight\s*path|country|region|state|city|province|zip|postal",
                "type": is_string_or_object,
                "desc": "Geographic Column (Route, Country, City, etc.)"
            }
        }
    }
}

# --- 3. CORE VALIDATION LOGIC ---
def _check_columns(df, rules):
    df_cols = df.columns.tolist()
    missing = []
    type_errors = []
    mapping = {}

    for std_col, criteria in rules.items():
        match_found = False
        for actual_col in df_cols:
            if re.search(criteria["pattern"], actual_col):
                mapping[std_col] = actual_col
                match_found = True
                if criteria["type"]:
                    if not criteria["type"](df[actual_col]):
                        actual_type = df[actual_col].dtype
                        type_errors.append(f"{std_col} (found as '{actual_col}') expects {criteria['type'].__name__}, got {actual_type}")
                break 
        if not match_found:
            missing.append(std_col)

    return {
        "ready": (len(missing) == 0) and (len(type_errors) == 0),
        "missing": missing,
        "type_errors": type_errors,
        "column_mapping": mapping
    }

def validate_dataset(df: pd.DataFrame, target_module: str = None) -> dict:
    report = {}
    if target_module:
        scope = {target_module: VALIDATION_RULES[target_module]}
    else:
        scope = VALIDATION_RULES

    for module, config in scope.items():
        if config["type"] == "simple":
            result = _check_columns(df, config["columns"])
            report[module] = result
        elif config["type"] == "flavored":
            module_report = {"ready": False, "flavor": None, "flavors": {}}
            for flavor_name, rules in config["flavors"].items():
                flavor_result = _check_columns(df, rules)
                module_report["flavors"][flavor_name] = flavor_result
                if flavor_result["ready"]:
                    module_report["ready"] = True
                    module_report["flavor"] = flavor_name
            report[module] = module_report
    return report