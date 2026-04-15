import pandas as pd

def get_signal(row, prev_row) -> int:
    if pd.isna(row['ma20']) or pd.isna(row['ma50']):
        return 0
    
    # Buy: ma20 crosses above ma50 with volume confirmation
    if row['ma20'] > row['ma50'] and prev_row['ma20'] <= prev_row['ma50']:
        if row['vol_ratio'] > 1.2:
            return 1
    
    # Sell: ma20 crosses below ma50
    if row['ma20'] < row['ma50'] and prev_row['ma20'] >= prev_row['ma50']:
        return -1
    
    return 0