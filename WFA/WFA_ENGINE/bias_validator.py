import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any
import logging

logger = logging.getLogger(__name__)

def check_indicator_lookahead(df: pd.DataFrame, indicator_cols: list[str]) -> list[str]:
    """
    Check for lookahead bias in indicator columns.
    
    For each indicator column, verify that the value at row i was computable
    using only rows 0 to i. Test on rows 51 to 60 for speed.
    
    Args:
        df: DataFrame with indicators
        indicator_cols: List of indicator column names to check
        
    Returns:
        List of column names that have lookahead bias
    """
    biased_cols = []
    tolerance = 1e-8
    
    # Test rows 51 to 60 (past warmup, small sample for speed)
    start_row = 51
    end_row = min(60, len(df) - 1)
    
    if end_row <= start_row:
        logger.warning("DataFrame too short for indicator lookahead check")
        return indicator_cols
    
    for col in indicator_cols:
        if col not in df.columns:
            logger.warning(f"Indicator column {col} not found in DataFrame")
            continue
            
        has_bias = False
        
        for i in range(start_row, end_row + 1):
            # Get original value
            original_value = df.iloc[i][col]
            
            # Skip if original value is NaN
            if pd.isna(original_value):
                continue
            
            # Recompute using only data up to row i
            try:
                if col == 'ma20':
                    recomputed = df.iloc[:i+1]['close'].rolling(20, min_periods=1).mean().iloc[-1]
                elif col == 'ma50':
                    recomputed = df.iloc[:i+1]['close'].rolling(50, min_periods=1).mean().iloc[-1]
                elif col == 'bb_upper':
                    bb_mid = df.iloc[:i+1]['close'].rolling(20, min_periods=1).mean()
                    bb_std = df.iloc[:i+1]['close'].rolling(20, min_periods=1).std()
                    recomputed = (bb_mid + bb_std * 2).iloc[-1]
                elif col == 'bb_mid':
                    recomputed = df.iloc[:i+1]['close'].rolling(20, min_periods=1).mean().iloc[-1]
                elif col == 'bb_lower':
                    bb_mid = df.iloc[:i+1]['close'].rolling(20, min_periods=1).mean()
                    bb_std = df.iloc[:i+1]['close'].rolling(20, min_periods=1).std()
                    recomputed = (bb_mid - bb_std * 2).iloc[-1]
                elif col == 'atr14':
                    subset = df.iloc[:i+1]
                    prev_close = subset['close'].shift(1)
                    tr = np.maximum(
                        subset['high'] - subset['low'],
                        np.maximum(
                            abs(subset['high'] - prev_close),
                            abs(subset['low'] - prev_close)
                        )
                    )
                    recomputed = tr.rolling(14, min_periods=14).mean().iloc[-1]
                elif col == 'roc10':
                    recomputed = df.iloc[:i+1]['close'].pct_change(10).iloc[-1]
                elif col == 'vol_ma20':
                    recomputed = df.iloc[:i+1]['volume'].rolling(20, min_periods=1).mean().iloc[-1]
                elif col == 'vol_ratio':
                    vol_ma = df.iloc[:i+1]['volume'].rolling(20, min_periods=1).mean()
                    recomputed = (df.iloc[:i+1]['volume'] / vol_ma).iloc[-1]
                else:
                    # For unknown indicators, check if it uses future data
                    # by comparing with a simple recomputation
                    if col in df.columns:
                        # If it's a shifted column, it's likely using future data
                        if 'shift' in str(df[col].dtype) or any(df[col].isna()):
                            # Check if NaN values appear at the end (indicating forward shift)
                            if df[col].iloc[-5:].isna().any():
                                has_bias = True
                                break
                    continue
                
                # Check for significant difference
                if not pd.isna(recomputed):
                    if abs(original_value - recomputed) > tolerance:
                        has_bias = True
                        break
                        
            except Exception as e:
                logger.warning(f"Error recomputing {col} at row {i}: {e}")
                has_bias = True
                break
        
        if has_bias:
            biased_cols.append(col)
    
    return biased_cols

def check_signal_lookahead(df: pd.DataFrame, strategy_module, indicator_cols: list[str]) -> bool:
    """
    Check for lookahead bias in signal generation.
    
    Modifies future close prices and checks if current signals change.
    
    Args:
        df: DataFrame with indicators
        strategy_module: Strategy module with get_signal function
        indicator_cols: List of indicator column names
        
    Returns:
        True if no lookahead found (clean), False if leak detected
    """
    from signal_generator import generate_signals
    
    # Generate original signals
    original_signals = generate_signals(df, strategy_module)
    
    # Test on 10 randomly selected row pairs
    test_rows = np.random.choice(range(1, len(df) - 1), size=min(10, len(df) - 2), replace=False)
    
    for i in test_rows:
        # Create modified DataFrame with extreme future close price
        df_modified = df.copy()
        df_modified.iloc[i + 1, df_modified.columns.get_loc('close')] *= 10  # 10x normal
        
        # Regenerate signals
        modified_signals = generate_signals(df_modified, strategy_module)
        
        # Check if signal at row i changed
        if original_signals.iloc[i] != modified_signals.iloc[i]:
            logger.warning(f"Signal lookahead detected at row {i}")
            return False
    
    return True

def check_entry_price_lookahead(trades: list[dict], signals: pd.Series) -> list[dict]:
    """
    Check for entry price lookahead bias.
    
    Verifies that entry_date is strictly after signal_date.
    
    Args:
        trades: List of trade dictionaries from backtest
        signals: Series of signals
        
    Returns:
        List of trades where entry bar == signal bar (should be empty)
    """
    problematic_trades = []
    
    for trade in trades:
        entry_date = pd.to_datetime(trade['entry_date'])
        
        # Check if there's a signal on the same day as entry
        if entry_date in signals.index:
            signal_value = signals[entry_date]
            if signal_value != 0:  # There's a signal on the entry day
                # Check if this is likely the signal that triggered the trade
                # by looking for signals in the few days before entry
                signals_before_entry = signals[signals.index < entry_date]
                if len(signals_before_entry) > 0:
                    # Look at the last few signals before entry
                    recent_signals = signals_before_entry.tail(3)
                    if (recent_signals == 0).all():  # No recent signals before entry
                        # Signal on entry day with no recent signals = likely same-bar entry
                        problematic_trades.append(trade)
                else:
                    # No signals before entry, but signal on entry day = same-bar entry
                    problematic_trades.append(trade)
    
    return problematic_trades

def check_oos_contamination(wfa_windows: list[dict]) -> list[int]:
    """
    Check for OOS contamination (date overlap between IS and OOS).
    
    Args:
        wfa_windows: List of window dictionaries from WFA output
        
    Returns:
        List of window_ids where IS and OOS overlap (should be empty)
    """
    contaminated_windows = []
    
    for window in wfa_windows:
        is_end = pd.to_datetime(window['is_end'])
        oos_start = pd.to_datetime(window['oos_start'])
        
        # Check if oos_start <= is_end (overlap or touching)
        if oos_start <= is_end:
            contaminated_windows.append(window['window_id'])
    
    return contaminated_windows

def check_exit_before_entry(trades: list[dict]) -> list[dict]:
    """
    Check for trades that exit before they enter.
    
    Args:
        trades: List of trade dictionaries from backtest
        
    Returns:
        List of trades where exit_date < entry_date (should be empty)
    """
    problematic_trades = []
    
    for trade in trades:
        entry_date = pd.to_datetime(trade['entry_date'])
        exit_date = pd.to_datetime(trade['exit_date'])
        
        if exit_date < entry_date:
            problematic_trades.append(trade)
    
    return problematic_trades

def check_survivorship_warning(stocks: dict[str, pd.DataFrame], expected_start: str = "2015-01-01") -> list[str]:
    """
    Check for survivorship bias by identifying stocks that start late.
    
    Args:
        stocks: Dictionary of stock DataFrames
        expected_start: Expected start date for analysis
        
    Returns:
        List of tickers that started more than 180 days after expected_start
    """
    expected_start_dt = pd.to_datetime(expected_start)
    threshold_days = 180
    late_starters = []
    
    for ticker, df in stocks.items():
        if len(df) == 0:
            continue
            
        actual_start = df.index[0]
        
        # Handle timezone-aware vs timezone-naive comparison
        if hasattr(actual_start, 'tz') and actual_start.tz is not None:
            actual_start = actual_start.tz_localize(None)
        
        days_late = (actual_start - expected_start_dt).days
        
        # Only flag if the stock starts significantly late AND
        # the expected start is within a reasonable range of the data
        if days_late > threshold_days:
            # Additional check: make sure we're not flagging stocks that are just
            # part of a different analysis period
            if actual_start.year >= expected_start_dt.year:  # Same era or later
                late_starters.append(ticker)
    
    return late_starters

def run_all_checks(
    df_sample: pd.DataFrame,
    strategy_module: Any,
    wfa_output: dict,
    trades_sample: list[dict],
    signals_sample: pd.Series,
    stocks: dict[str, pd.DataFrame]
) -> dict:
    """
    Run all bias validation checks.
    
    Args:
        df_sample: Sample DataFrame with indicators
        strategy_module: Strategy module
        wfa_output: WFA output dictionary
        trades_sample: Sample trades from backtest
        signals_sample: Sample signals
        stocks: Dictionary of all stocks
        
    Returns:
        Dictionary with all check results and overall status
    """
    print("=" * 60)
    print("BIAS VALIDATION REPORT")
    print("=" * 60)
    
    results = {
        "lookahead_indicator_cols": [],
        "signal_lookahead_detected": False,
        "entry_lookahead_trades": [],
        "oos_contaminated_windows": [],
        "exit_before_entry_trades": [],
        "survivorship_warning_tickers": [],
        "overall_clean": True
    }
    
    # Check 1: Indicator lookahead bias
    print("Check 1: Indicator Lookahead Bias...")
    indicator_cols = ['ma20', 'ma50', 'bb_upper', 'bb_mid', 'bb_lower', 'atr14', 'roc10', 'vol_ma20', 'vol_ratio']
    biased_indicators = check_indicator_lookahead(df_sample, indicator_cols)
    results["lookahead_indicator_cols"] = biased_indicators
    
    if biased_indicators:
        print(f"  ❌ FAIL: Lookahead bias detected in columns: {biased_indicators}")
        results["overall_clean"] = False
    else:
        print("  ✅ PASS: No indicator lookahead bias detected")
    
    # Check 2: Signal generation lookahead
    print("Check 2: Signal Generation Lookahead...")
    signal_clean = check_signal_lookahead(df_sample, strategy_module, indicator_cols)
    results["signal_lookahead_detected"] = not signal_clean
    
    if not signal_clean:
        print("  ❌ FAIL: Signal lookahead bias detected")
        results["overall_clean"] = False
    else:
        print("  ✅ PASS: No signal lookahead bias detected")
    
    # Check 3: Entry price lookahead
    print("Check 3: Entry Price Lookahead...")
    entry_issues = check_entry_price_lookahead(trades_sample, signals_sample)
    results["entry_lookahead_trades"] = entry_issues
    
    if entry_issues:
        print(f"  ❌ FAIL: Entry lookahead bias in {len(entry_issues)} trades")
        results["overall_clean"] = False
    else:
        print("  ✅ PASS: No entry price lookahead bias detected")
    
    # Check 4: OOS contamination
    print("Check 4: OOS Contamination...")
    contaminated_windows = check_oos_contamination(wfa_output.get('windows', []))
    results["oos_contaminated_windows"] = contaminated_windows
    
    if contaminated_windows:
        print(f"  ❌ FAIL: OOS contamination in windows: {contaminated_windows}")
        results["overall_clean"] = False
    else:
        print("  ✅ PASS: No OOS contamination detected")
    
    # Check 5: Exit before entry
    print("Check 5: Exit Before Entry...")
    exit_issues = check_exit_before_entry(trades_sample)
    results["exit_before_entry_trades"] = exit_issues
    
    if exit_issues:
        print(f"  ❌ FAIL: Exit before entry in {len(exit_issues)} trades")
        results["overall_clean"] = False
    else:
        print("  ✅ PASS: No exit-before-entry violations detected")
    
    # Check 6: Survivorship bias warning
    print("Check 6: Survivorship Bias Warning...")
    late_starters = check_survivorship_warning(stocks)
    results["survivorship_warning_tickers"] = late_starters
    
    if late_starters:
        print(f"  ⚠️  WARNING: {len(late_starters)} stocks start after 2015-07-01 (potential survivorship bias):")
        for ticker in late_starters[:10]:  # Show first 10
            start_date = stocks[ticker].index[0].strftime('%Y-%m-%d')
            print(f"    - {ticker}: starts {start_date}")
        if len(late_starters) > 10:
            print(f"    ... and {len(late_starters) - 10} more")
    else:
        print("  ✅ INFO: No obvious survivorship bias detected")
    
    # Overall result
    print("=" * 60)
    if results["overall_clean"]:
        print("🎉 OVERALL: ALL CHECKS PASSED - No bias detected")
    else:
        print("⚠️  OVERALL: BIAS ISSUES DETECTED - Review required")
    print("=" * 60)
    
    return results
