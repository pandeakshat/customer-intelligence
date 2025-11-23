# src/data_loader.py
import pandas as pd
import os

# src/data_loader.py
import pandas as pd
import os
import io

def load_dataset(file_source, mapping: dict = None) -> pd.DataFrame:
    """
    Universal Loader: Handles paths (str) AND file buffers (UploadedFile).
    """
    # 1. Determine File Type & Source
    if isinstance(file_source, str):
        # It's a file path (Sample Data)
        if not os.path.exists(file_source):
            return pd.DataFrame()
        _, ext = os.path.splitext(file_source)
        ext = ext.lower()
        source = file_source # Pass path to pandas
    else:
        # It's a Streamlit UploadedFile object (Memory Buffer)
        # We assume CSV by default for uploads, or check name
        filename = getattr(file_source, "name", "").lower()
        ext = os.path.splitext(filename)[1] if filename else ".csv"
        source = file_source # Pass buffer to pandas

    try:
        # 2. Load based on extension
        if ext == '.csv':
            df = pd.read_csv(source)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(source, engine='openpyxl')
        elif ext == '.parquet':
            df = pd.read_parquet(source, engine='pyarrow')
        elif ext == '.json':
            df = pd.read_json(source, orient='records') 
        else:
            return pd.DataFrame()

        # 3. Apply Column Mapping (Smart Rename)
        if mapping:
            # {Standard: Actual} -> {Actual: Standard}
            rename_dict = {v: k for k, v in mapping.items()}
            df = df.rename(columns=rename_dict)

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
def normalize_columns(df, mapping):
    """
    Renames the columns based on the Validator's discovery.
    Input: df, mapping={'TotalAmount': 'Amt'}
    Output: df with 'TotalAmount' column
    """
    # Invert the mapping to fit pandas rename format: {OldName: NewName}
    rename_dict = {v: k for k, v in mapping.items()}
    return df.rename(columns=rename_dict)