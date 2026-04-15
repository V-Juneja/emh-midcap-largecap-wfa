import pandas as pd
import logging
import importlib.util

logger = logging.getLogger(__name__)


def generate_signals(df: pd.DataFrame, strategy_module) -> pd.Series:
    """Apply the strategy's get_signal function row by row to a DataFrame."""
    if len(df) == 0:
        return pd.Series(dtype=int, index=df.index)
    
    signals = pd.Series(0, index=df.index, dtype=int)
    
    if len(df) == 1:
        return signals
    
    # Skip first 50 rows for warmup period (unstable indicators)
    warmup_period = 50
    
    for i in range(max(1, warmup_period), len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # Check if key indicators are available and not NaN
        if (pd.isna(current_row.get('ma20')) or pd.isna(current_row.get('ma50')) or 
            pd.isna(current_row.get('atr14')) or pd.isna(current_row.get('vol_ratio'))):
            signals.iloc[i] = 0
            continue
        
        try:
            signal = strategy_module.get_signal(current_row, prev_row)
            signals.iloc[i] = signal
        except Exception as e:
            logger.error(f"Error generating signal for row {df.index[i]}: {str(e)}")
            signals.iloc[i] = 0
    
    return signals


def load_strategy(strategy_path: str):
    """Dynamically load a strategy file from an absolute path."""
    spec = importlib.util.spec_from_file_location("strategy_module", strategy_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load strategy from {strategy_path}")
    
    strategy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_module)
    
    if not hasattr(strategy_module, 'get_signal'):
        raise ValueError(f"Strategy module must have a 'get_signal' function")
    
    if not callable(strategy_module.get_signal):
        raise ValueError(f"'get_signal' must be a callable function")
    
    return strategy_module
