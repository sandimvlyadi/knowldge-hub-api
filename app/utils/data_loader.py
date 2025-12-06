"""
Utility functions untuk data loading dan preprocessing
"""

import json
import pandas as pd
from pathlib import Path


def load_json_data(filename):
    """Load JSON data dengan berbagai format"""
    p = Path(filename)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    
    try:
        return pd.read_json(p)
    except ValueError:
        try:
            return pd.read_json(p, lines=True)
        except ValueError:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.json_normalize(data)


def create_text_column(df):
    """Bersihkan dan gabungkan summary + description"""
    df['summary'] = df['summary'].fillna('').astype(str)
    df['description'] = df['description'].fillna('').astype(str)
    
    df['summary'] = df['summary'].apply(lambda x: x + '.' if x and not x.strip().endswith('.') else x)
    df['description'] = df['description'].apply(lambda x: x + '.' if x and not x.strip().endswith('.') else x)
    
    df['text'] = (df['summary'] + ' ' + df['description']).str.strip()
    return df
