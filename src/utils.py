# Utility functions, e.g., preprocessing input JSON to dataframe can go here

import pandas as pd

def preprocess_input(data: dict) -> pd.DataFrame:
    # Example conversion of input dict to DataFrame with proper features
    df = pd.DataFrame([data])
    # Example: add feature if needed
    # df['days_since_last_claim'] = ...
    return df
