#!/usr/bin/env python3
"""
Property-based test for WFA system.
Runs a quick integration test with controlled data to verify system behavior.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add WFA_ENGINE to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all_stocks
from indicators import compute_indicators
from signal_generator import load_strategy, generate_signals
from wfa_runner import run_wfa

def test_compounded_equity_curve():
    """Verify equity curve uses compound returns not simple sum"""
    # Create synthetic data
    dates = pd.date_range('2015-01-01', periods=2500, freq='D')
    
    stocks = {}
    signals_dict = {}
    
    for i in range(3):
        ticker = f'STOCK{i}'
        
        # Create steadily rising prices
        prices = np.linspace(100.0, 130.0, 2500)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + 2.0,
            'low': prices - 2.0,
            'close': prices + np.random.normal(0, 0.1, 2500),
            'volume': np.full(2500, 1000000),
            'ma20': pd.Series(prices).rolling(20, min_periods=1).mean(),
            'ma50': pd.Series(prices).rolling(50, min_periods=1).mean(),
            'bb_upper': pd.Series(prices).rolling(20, min_periods=1).mean() + 4.0,
            'bb_mid': pd.Series(prices).rolling(20, min_periods=1).mean(),
            'bb_lower': pd.Series(prices).rolling(20, min_periods=1).mean() - 4.0,
            'atr14': np.full(2500, 2.0),
            'roc10': np.zeros(2500),
            'vol_ma20': np.full(2500, 1000000),
            'vol_ratio': np.full(2500, 1.0)
        }, index=dates)
        
        stocks[ticker] = df
        
        # Create signals
        signals = pd.Series(0, index=dates)
        for idx in range(100, 2400, 50):
            signals.iloc[idx] = 1
        signals_dict[ticker] = signals
    
    # Run WFA
    result = run_wfa(
        stocks, signals_dict,
        start_date='2015-01-01',
        end_date='2024-12-31',
        is_window_months=24,
        oos_window_months=6
    )
    
    # Get equity curve and raw returns
    equity_curve = result['summary']['oos_equity_curve']
    windows = result['windows']
    
    # Verify compound calculation
    raw_returns = [w['oos_metrics'].get('total_return_pct', 0) / 100 for w in windows]
    
    # Calculate expected compounded values
    cumulative = 1.0
    expected = []
    for r in raw_returns:
        cumulative *= (1 + r)
        expected.append((cumulative - 1) * 100)
    
    # Verify within tolerance
    for i, (actual, exp) in enumerate(zip(equity_curve, expected)):
        if abs(actual - exp) > 0.5:
            print(f"FAIL: Equity curve mismatch at window {i}: got {actual}, expected {exp}")
            return False
    
    print(f"PASS: Equity curve correctly compounded across {len(equity_curve)} windows")
    return True

def test_mean_not_sum_for_returns():
    """Verify total_return_pct is averaged not summed across stocks"""
    dates = pd.date_range('2015-01-01', periods=1000, freq='D')
    
    stocks = {}
    signals_dict = {}
    
    # Create 3 stocks with identical rising prices
    for i in range(3):
        ticker = f'STOCK{i}'
        prices = np.linspace(100.0, 110.0, 1000)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + 2.0,
            'low': prices - 2.0,
            'close': prices,
            'volume': np.full(1000, 1000000),
            'ma20': pd.Series(prices).rolling(20, min_periods=1).mean(),
            'ma50': pd.Series(prices).rolling(50, min_periods=1).mean(),
            'bb_upper': pd.Series(prices).rolling(20, min_periods=1).mean() + 4.0,
            'bb_mid': pd.Series(prices).rolling(20, min_periods=1).mean(),
            'bb_lower': pd.Series(prices).rolling(20, min_periods=1).mean() - 4.0,
            'atr14': np.full(1000, 2.0),
            'roc10': np.zeros(1000),
            'vol_ma20': np.full(1000, 1000000),
            'vol_ratio': np.full(1000, 1.0)
        }, index=dates)
        
        stocks[ticker] = df
        signals = pd.Series(0, index=dates)
        signals.iloc[100] = 1
        signals_dict[ticker] = signals
    
    result = run_wfa(
        stocks, signals_dict,
        start_date='2015-01-01',
        end_date='2017-12-31',
        is_window_months=12,
        oos_window_months=6
    )
    
    # If all stocks have same returns, mean should be similar to individual
    # Sum would be 3x larger
    for window in result['windows']:
        oos_return = window['oos_metrics'].get('total_return_pct', 0)
        # With identical stocks, mean should be reasonable (not 3x inflated)
        if oos_return > 30:  # Unrealistically high if summed
            print(f"FAIL: OOS return {oos_return}% looks like sum not mean")
            return False
    
    print("PASS: Returns appear to be averaged not summed")
    return True

def main():
    """Run all property tests"""
    print("Running property-based tests...")
    print("=" * 50)
    
    tests = [
        ("Compounded Equity Curve", test_compounded_equity_curve),
        ("Mean Not Sum for Returns", test_mean_not_sum_for_returns)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\nTest: {name}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"FAIL: Exception - {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
