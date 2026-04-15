import pandas as pd
import numpy as np

WARMUP_PERIOD = 50


def get_required_warmup_period() -> int:
    return WARMUP_PERIOD


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators on a stock DataFrame."""
    result = df.copy()
    
    result['ma20'] = result['close'].rolling(window=20, min_periods=20).mean()
    result['ma50'] = result['close'].rolling(window=50, min_periods=50).mean()
    
    bb_mid = result['close'].rolling(window=20, min_periods=20).mean()
    bb_std = result['close'].rolling(window=20, min_periods=20).std()
    result['bb_upper'] = bb_mid + (bb_std * 2)
    result['bb_mid'] = bb_mid
    result['bb_lower'] = bb_mid - (bb_std * 2)
    
    prev_close = result['close'].shift(1)
    tr1 = result['high'] - result['low']
    tr2 = abs(result['high'] - prev_close)
    tr3 = abs(result['low'] - prev_close)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    result['atr14'] = true_range.rolling(window=14, min_periods=14).mean()
    
    result['roc10'] = (result['close'] - result['close'].shift(10)) / result['close'].shift(10)
    
    result['vol_ma20'] = result['volume'].rolling(window=20, min_periods=20).mean()
    result['vol_ratio'] = result['volume'] / result['vol_ma20']
    
    return result
