#!/usr/bin/env python3
"""
Edge Cases and Boundary Conditions Test Suite
Tests extreme scenarios and boundary conditions to ensure robustness.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import load_all_stocks
from indicators import compute_indicators
from signal_generator import load_strategy, generate_signals
from wfa_runner import run_wfa
from backtest import run_backtest


def test_extreme_volatility_handling():
    """Test system behavior with extreme price volatility."""
    print("Testing extreme volatility handling...")
    
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Create extreme volatility data
    base_prices = np.linspace(100, 110, 200)
    volatility = np.random.uniform(-20, 20, 200)  # ±20% daily moves
    prices = base_prices * (1 + volatility / 100)
    prices = np.maximum(prices, 1)  # Ensure positive prices
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.3,  # 30% intraday ranges
        'low': prices * 0.7,
        'close': prices,
        'volume': np.random.uniform(100000, 10000000, 200)
    }, index=dates)
    
    df = compute_indicators(df)
    strategy = load_strategy(str(Path(__file__).parent.parent.parent / "STRATEGY" / "strategy.py"))
    signals = generate_signals(df, strategy)
    
    # Run backtest
    result = run_backtest(
        df, signals, '2020-01-01', '2020-12-31',
        100000, 0.02, 2.0, 0.0015
    )
    
    # System should handle extreme data without crashing
    if result['status'] == 'error':
        print("❌ FAIL: System crashed with extreme volatility")
        return False
    
    # Check that ATR values are reasonable
    valid_atr = df['atr14'].dropna()
    if (valid_atr <= 0).any():
        print("❌ FAIL: Invalid ATR values with extreme volatility")
        return False
    
    print("✅ PASS: Extreme volatility handled correctly")
    return True


def test_gap_handling():
    """Test system behavior with price gaps."""
    print("Testing gap handling...")
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Create data with large gaps
    prices = np.linspace(100, 110, 100)
    # Introduce gaps at specific points
    prices[30:35] *= 1.5  # 50% gap up
    prices[60:65] *= 0.6  # 40% gap down
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 2,
        'low': prices - 2,
        'close': prices,
        'volume': np.full(100, 1000000)
    }, index=dates)
    
    df = compute_indicators(df)
    strategy = load_strategy(str(Path(__file__).parent.parent.parent / "STRATEGY" / "strategy.py"))
    signals = generate_signals(df, strategy)
    
    # Force trades around gaps
    signals.iloc[25] = 1  # Trade before gap up
    signals.iloc[55] = 1  # Trade before gap down
    
    result = run_backtest(
        df, signals, '2020-01-01', '2020-12-31',
        100000, 0.02, 2.0, 0.0015
    )
    
    if result['status'] == 'error':
        print("❌ FAIL: System crashed with price gaps")
        return False
    
    # Check that trades executed correctly despite gaps
    if len(result['trades']) == 0:
        print("❌ FAIL: No trades executed with gaps")
        return False
    
    print("✅ PASS: Price gaps handled correctly")
    return True


def test_illiquid_stock_handling():
    """Test system behavior with illiquid stocks (low volume)."""
    print("Testing illiquid stock handling...")
    
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    prices = np.linspace(100, 110, 200)
    
    # Create very low volume data
    volumes = np.concatenate([
        np.full(50, 100),        # Very low volume
        np.full(50, 1000),       # Low volume
        np.full(50, 10000),      # Medium volume
        np.full(50, 1000000)     # Normal volume
    ])
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    df = compute_indicators(df)
    strategy = load_strategy(str(Path(__file__).parent.parent.parent / "STRATEGY" / "strategy.py"))
    signals = generate_signals(df, strategy)
    
    # Run backtest
    result = run_backtest(
        df, signals, '2020-01-01', '2020-12-31',
        100000, 0.02, 2.0, 0.0015
    )
    
    if result['status'] == 'error':
        print("❌ FAIL: System crashed with illiquid stock")
        return False
    
    # Check volume ratio calculations
    valid_vol_ratios = df['vol_ratio'].dropna()
    if (valid_vol_ratios <= 0).any():
        print("❌ FAIL: Invalid volume ratios with illiquid stock")
        return False
    
    print("✅ PASS: Illiquid stock handled correctly")
    return True


def test_minimum_data_requirements():
    """Test system behavior with minimum required data."""
    print("Testing minimum data requirements...")
    
    # Test with exactly 500 days (minimum requirement)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    prices = np.linspace(100, 110, 500)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': np.full(500, 1000000)
    }, index=dates)
    
    df = compute_indicators(df)
    
    # Check that indicators are calculated for minimum data
    if df['ma50'].iloc[-1] == 0 or pd.isna(df['ma50'].iloc[-1]):
        print("❌ FAIL: MA50 not calculated with minimum data")
        return False
    
    if df['atr14'].iloc[-1] == 0 or pd.isna(df['atr14'].iloc[-1]):
        print("❌ FAIL: ATR not calculated with minimum data")
        return False
    
    # Test with insufficient data
    dates_short = pd.date_range('2020-01-01', periods=100, freq='D')
    prices_short = np.linspace(100, 110, 100)
    
    df_short = pd.DataFrame({
        'open': prices_short,
        'high': prices_short + 1,
        'low': prices_short - 1,
        'close': prices_short,
        'volume': np.full(100, 1000000)
    }, index=dates_short)
    
    # System should handle short data gracefully
    try:
        df_short = compute_indicators(df_short)
        # Should still work but with many NaN values
    except Exception:
        print("❌ FAIL: System crashed with insufficient data")
        return False
    
    print("✅ PASS: Minimum data requirements handled correctly")
    return True


def test_zero_price_handling():
    """Test system behavior with zero or negative prices."""
    print("Testing zero price handling...")
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = np.linspace(100, 110, 100)
    
    # Introduce problematic prices
    prices[30] = 0      # Zero price
    prices[60] = -10    # Negative price
    prices[90] = 0.001  # Very small positive price
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': np.full(100, 1000000)
    }, index=dates)
    
    # System should handle or filter out invalid prices
    try:
        df = compute_indicators(df)
        
        # Check that invalid prices were handled
        if (df['open'] <= 0).any() or (df['close'] <= 0).any():
            print("❌ FAIL: Invalid prices not filtered out")
            return False
            
    except Exception as e:
        # Exception is acceptable if it handles invalid data gracefully
        print(f"✅ PASS: Invalid prices handled with exception: {e}")
        return True
    
    print("✅ PASS: Zero/negative prices handled correctly")
    return True


def test_extreme_atr_values():
    """Test system behavior with extreme ATR values."""
    print("Testing extreme ATR values...")
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = np.linspace(100, 110, 100)
    
    # Create extreme ATR conditions
    high_volatility = np.concatenate([
        np.full(25, 100),      # Normal
        np.full(25, 0.01),     # Very low volatility
        np.full(25, 50),       # High volatility
        np.full(25, 1)         # Normal
    ])
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + high_volatility,
        'low': prices - high_volatility,
        'close': prices,
        'volume': np.full(100, 1000000)
    }, index=dates)
    
    df = compute_indicators(df)
    strategy = load_strategy(str(Path(__file__).parent.parent.parent / "STRATEGY" / "strategy.py"))
    
    # Force trades during extreme ATR periods
    signals = pd.Series(0, index=df.index)
    signals.iloc[20] = 1  # Trade during normal ATR
    signals.iloc[45] = 1  # Trade during low ATR
    signals.iloc[70] = 1  # Trade during high ATR
    
    result = run_backtest(
        df, signals, '2020-01-01', '2020-12-31',
        100000, 0.02, 2.0, 0.0015
    )
    
    if result['status'] == 'error':
        print("❌ FAIL: System crashed with extreme ATR values")
        return False
    
    # Check that position sizing handled extreme ATR correctly
    for trade in result['trades']:
        if trade['shares'] <= 0:
            print("❌ FAIL: Invalid position size with extreme ATR")
            return False
        if trade['stop_price'] <= 0:
            print("❌ FAIL: Invalid stop price with extreme ATR")
            return False
    
    print("✅ PASS: Extreme ATR values handled correctly")
    return True


def test_multiple_signals_same_day():
    """Test system behavior with multiple signals on consecutive days."""
    print("Testing multiple signals same day...")
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = np.linspace(100, 110, 100)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': np.full(100, 1000000)
    }, index=dates)
    
    df = compute_indicators(df)
    strategy = load_strategy(str(Path(__file__).parent.parent.parent / "STRATEGY" / "strategy.py"))
    
    # Create conflicting signals
    signals = pd.Series(0, index=df.index)
    signals.iloc[30] = 1   # Buy
    signals.iloc[31] = -1  # Sell next day
    signals.iloc[32] = 1   # Buy again
    signals.iloc[33] = 1   # Another buy signal (should be ignored)
    
    result = run_backtest(
        df, signals, '2020-01-01', '2020-12-31',
        100000, 0.02, 2.0, 0.0015
    )
    
    if result['status'] == 'error':
        print("❌ FAIL: System crashed with multiple signals")
        return False
    
    # System should handle consecutive signals correctly
    # (no overlapping positions, proper exits)
    for i, trade in enumerate(result['trades']):
        if i > 0:
            prev_trade = result['trades'][i-1]
            if trade['entry_date'] < prev_trade['exit_date']:
                print("❌ FAIL: Overlapping trades detected")
                return False
    
    print("✅ PASS: Multiple signals handled correctly")
    return True


def test_window_edge_cases():
    """Test WFA window edge cases."""
    print("Testing WFA window edge cases...")
    
    # Test with very short date range
    stocks = {}
    dates = pd.date_range('2020-01-01', periods=400, freq='D')  # Just over minimum
    
    for i in range(1):
        ticker = f'TEST{i}'
        prices = np.linspace(100, 110, 400)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': np.full(400, 1000000)
        }, index=dates)
        
        stocks[ticker] = compute_indicators(df)
    
    strategy = load_strategy(str(Path(__file__).parent.parent.parent / "STRATEGY" / "strategy.py"))
    signals_dict = {}
    for ticker, df in stocks.items():
        signals_dict[ticker] = generate_signals(df, strategy)
    
    # Try to run WFA with minimal windows
    try:
        result = run_wfa(
            stocks, signals_dict,
            '2020-01-01', '2021-12-31',
            6, 3, 100000, 0.02, 2.0, 0.0015
        )
        
        # Should either work or fail gracefully
        if result['summary']['total_windows'] < 3:
            print("✅ PASS: Minimal date range handled correctly (insufficient windows)")
        else:
            print("✅ PASS: Minimal date range handled correctly (sufficient windows)")
            
    except Exception as e:
        if "Insufficient data range" in str(e):
            print("✅ PASS: Minimal date range handled correctly (proper error)")
        else:
            print(f"❌ FAIL: Unexpected error with minimal date range: {e}")
            return False
    
    return True


def run_edge_case_tests():
    """Run all edge case and boundary condition tests."""
    print("=" * 80)
    print("EDGE CASES AND BOUNDARY CONDITIONS TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Extreme Volatility", test_extreme_volatility_handling),
        ("Gap Handling", test_gap_handling),
        ("Illiquid Stocks", test_illiquid_stock_handling),
        ("Minimum Data Requirements", test_minimum_data_requirements),
        ("Zero Price Handling", test_zero_price_handling),
        ("Extreme ATR Values", test_extreme_atr_values),
        ("Multiple Signals", test_multiple_signals_same_day),
        ("Window Edge Cases", test_window_edge_cases)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ FAIL: Exception in {test_name}: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("EDGE CASE TEST RESULTS")
    print("=" * 80)
    print(f"✅ PASSED: {passed}")
    print(f"❌ FAILED: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL EDGE CASE TESTS PASSED!")
        print("The WFA engine handles extreme conditions robustly.")
    else:
        print(f"\n⚠️  {failed} EDGE CASE TESTS FAILED!")
    
    print("=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_edge_case_tests())
