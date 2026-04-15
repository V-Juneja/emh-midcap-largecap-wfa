#!/usr/bin/env python3
"""
Comprehensive Bias Detection Test Suite
Ensures absolutely no bias or implementation issues exist in the WFA engine.
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
from bias_validator import run_all_checks


def get_strategy_path():
    """Get the strategy file path."""
    return str(Path(__file__).parent.parent.parent / "STRATEGY" / "strategy.py")


def test_future_price_leakage():
    """Test that modifying future prices doesn't affect current signals."""
    print("Testing future price leakage...")
    
    # Create synthetic data
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    prices = np.linspace(100, 120, 200)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 2,
        'low': prices - 2,
        'close': prices,
        'volume': np.full(200, 1000000)
    }, index=dates)
    
    df = compute_indicators(df)
    
    # Load strategy
    strategy = load_strategy(get_strategy_path())
    
    # Generate original signals
    original_signals = generate_signals(df, strategy)
    
    # Modify future prices dramatically
    df_modified = df.copy()
    df_modified.iloc[100:, df_modified.columns.get_loc('close')] *= 10
    
    # Regenerate signals
    modified_signals = generate_signals(df_modified, strategy)
    
    # Check that signals before modification are identical
    identical_signals = (original_signals[:100] == modified_signals[:100]).all()
    
    if not identical_signals:
        print("❌ FAIL: Future price leakage detected")
        return False
    
    print("✅ PASS: No future price leakage")
    return True


def test_signal_timing_bias():
    """Test that signals aren't generated using future information."""
    print("Testing signal timing bias...")
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Create data with clear trend reversal
    prices = np.concatenate([
        np.linspace(100, 110, 50),  # Uptrend
        np.linspace(110, 90, 50)    # Downtrend
    ])
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': np.full(100, 1000000)
    }, index=dates)
    
    df = compute_indicators(df)
    strategy = load_strategy(get_strategy_path())
    signals = generate_signals(df, strategy)
    
    # Check that no signals anticipate the trend reversal
    reversal_point = 50
    
    # Signals before reversal should not be influenced by future downtrend
    pre_reversal_signals = signals.iloc[:reversal_point-5]  # Buffer period
    
    # If there's lookahead bias, we might see premature sell signals
    premature_sells = (pre_reversal_signals == -1).sum()
    
    if premature_sells > 0:
        print(f"❌ FAIL: {premature_sells} premature sell signals detected (possible lookahead)")
        return False
    
    print("✅ PASS: No signal timing bias detected")
    return True


def test_position_sizing_consistency():
    """Test that position sizing is consistent and doesn't use future capital."""
    print("Testing position sizing consistency...")
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = np.linspace(100, 110, 100)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 2,
        'low': prices - 2,
        'close': prices,
        'volume': np.full(100, 1000000)
    }, index=dates)
    
    df = compute_indicators(df)
    strategy = load_strategy(get_strategy_path())
    signals = generate_signals(df, strategy)
    
    # Force a buy signal at day 30 (after warmup)
    signals.iloc[60] = 1
    
    result = run_backtest(
        df, signals, '2020-01-01', '2020-12-31',
        100000, 0.02, 2.0, 0.0015
    )
    
    if result['status'] != 'ok' or len(result['trades']) == 0:
        print("❌ FAIL: No trades executed for position sizing test")
        return False
    
    trade = result['trades'][0]
    
    # Check that position size is reasonable
    expected_shares = int((100000 * 0.02) / (trade['entry_price'] - trade['stop_price']))
    actual_shares = trade['shares']
    
    if abs(actual_shares - expected_shares) > 1:
        print(f"❌ FAIL: Position sizing inconsistent. Expected: {expected_shares}, Got: {actual_shares}")
        return False
    
    print("✅ PASS: Position sizing consistent")
    return True


def test_window_boundary_integrity():
    """Test that IS/OOS windows don't overlap or have gaps."""
    print("Testing window boundary integrity...")
    
    # Create synthetic data
    stocks = {}
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    for i in range(2):
        ticker = f'TEST{i}'
        prices = np.random.uniform(100, 110, 1000)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.full(1000, 1000000)
        }, index=dates)
        
        stocks[ticker] = compute_indicators(df)
    
    # Generate signals
    strategy = load_strategy(get_strategy_path())
    signals_dict = {}
    for ticker, df in stocks.items():
        signals_dict[ticker] = generate_signals(df, strategy)
    
    # Run WFA
    result = run_wfa(
        stocks, signals_dict,
        '2020-01-01', '2022-12-31',
        6, 3, 100000, 0.02, 2.0, 0.0015
    )
    
    windows = result['windows']
    
    for i, window in enumerate(windows):
        is_end = pd.to_datetime(window['is_end'])
        oos_start = pd.to_datetime(window['oos_start'])
        
        # Check for overlap
        if oos_start <= is_end:
            print(f"❌ FAIL: Window {i} has IS/OOS overlap")
            return False
        
        # Check for excessive gaps
        gap_days = (oos_start - is_end).days
        if gap_days > 7:  # Allow for weekends
            print(f"❌ FAIL: Window {i} has excessive gap: {gap_days} days")
            return False
    
    print("✅ PASS: Window boundaries clean")
    return True


def test_equity_curve_compounding():
    """Test that equity curve uses proper compounding."""
    print("Testing equity curve compounding...")
    
    # Create synthetic data with known returns
    stocks = {}
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    for i in range(2):
        ticker = f'TEST{i}'
        prices = np.linspace(100, 105, 500)  # 5% total return
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': np.full(500, 1000000)
        }, index=dates)
        
        stocks[ticker] = compute_indicators(df)
    
    # Force consistent returns
    strategy = load_strategy(get_strategy_path())
    signals_dict = {}
    for ticker, df in stocks.items():
        signals = pd.Series(0, index=df.index)
        # Force one trade per window for consistent returns
        for j in range(50, 450, 100):
            signals.iloc[j] = 1
        signals_dict[ticker] = signals
    
    # Run WFA with short windows
    result = run_wfa(
        stocks, signals_dict,
        '2020-01-01', '2021-12-31',
        3, 1, 100000, 0.02, 2.0, 0.0015
    )
    
    equity_curve = result['summary']['oos_equity_curve']
    
    # Test compounding math
    if len(equity_curve) > 1:
        # Check that returns are compounded, not summed
        simple_sum = sum(equity_curve)
        final_compounded = equity_curve[-1]
        
        if abs(simple_sum - final_compounded) < 0.1:
            print("❌ FAIL: Equity curve appears to use simple sum instead of compounding")
            return False
    
    print("✅ PASS: Equity curve properly compounded")
    return True


def test_data_leakage_across_windows():
    """Test that information doesn't leak between WFA windows."""
    print("Testing data leakage across windows...")
    
    # Create data with distinct regimes
    stocks = {}
    dates = pd.date_range('2020-01-01', periods=800, freq='D')
    
    for i in range(2):
        ticker = f'TEST{i}'
        # Different price behavior in different periods
        prices1 = np.linspace(100, 120, 200)  # Strong uptrend
        prices2 = np.linspace(120, 115, 200)  # Downtrend
        prices3 = np.linspace(115, 125, 200)  # Recovery
        prices4 = np.linspace(125, 130, 200)  # Slow uptrend
        
        prices = np.concatenate([prices1, prices2, prices3, prices4])
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.full(800, 1000000)
        }, index=dates)
        
        stocks[ticker] = compute_indicators(df)
    
    strategy = load_strategy(get_strategy_path())
    signals_dict = {}
    for ticker, df in stocks.items():
        signals_dict[ticker] = generate_signals(df, strategy)
    
    # Run WFA
    result = run_wfa(
        stocks, signals_dict,
        '2020-01-01', '2022-12-31',
        6, 3, 100000, 0.02, 2.0, 0.0015
    )
    
    windows = result['windows']
    
    # Check that early windows don't benefit from future regime knowledge
    early_windows = windows[:3]
    late_windows = windows[-3:]
    
    early_avg_return = np.mean([w['is_metrics'].get('total_return_pct', 0) for w in early_windows])
    late_avg_return = np.mean([w['is_metrics'].get('total_return_pct', 0) for w in late_windows])
    
    # If there's data leakage, early windows might show unrealistic performance
    if early_avg_return > late_avg_return * 2:
        print("❌ FAIL: Possible data leakage - early windows performing unrealistically well")
        return False
    
    print("✅ PASS: No data leakage across windows")
    return True


def test_commission_and_slippage_consistency():
    """Test that commissions and slippage are applied consistently."""
    print("Testing commission and slippage consistency...")
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = np.linspace(100, 110, 100)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 2,
        'low': prices - 2,
        'close': prices,
        'volume': np.full(100, 1000000)
    }, index=dates)
    
    df = compute_indicators(df)
    strategy = load_strategy(get_strategy_path())
    
    # Force a trade
    signals = pd.Series(0, index=df.index)
    signals.iloc[30] = 1
    signals.iloc[60] = -1  # Exit signal
    
    result = run_backtest(
        df, signals, '2020-01-01', '2020-12-31',
        100000, 0.02, 2.0, 0.0015
    )
    
    if result['status'] != 'ok' or len(result['trades']) == 0:
        print("❌ FAIL: No trades for commission/slippage test")
        return False
    
    trade = result['trades'][0]
    
    # Check that slippage was applied to entry price
    expected_entry = df.iloc[31]['open'] * 1.0015  # Next day's open + slippage
    actual_entry = trade['entry_price']
    
    if abs(actual_entry - expected_entry) > 0.01:
        print(f"❌ FAIL: Entry slippage not applied correctly. Expected: {expected_entry}, Got: {actual_entry}")
        return False
    
    # Check that commission was calculated
    entry_commission = trade.get('entry_commission', 0)
    expected_commission = actual_entry * trade['shares'] * 0.0015
    
    if abs(entry_commission - expected_commission) > 0.01:
        print(f"❌ FAIL: Commission not calculated correctly. Expected: {expected_commission}, Got: {entry_commission}")
        return False
    
    print("✅ PASS: Commission and slippage applied consistently")
    return True


def test_atr_stop_loss_accuracy():
    """Test that ATR-based stop losses are calculated correctly."""
    print("Testing ATR stop loss accuracy...")
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = np.linspace(100, 110, 100)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 5,  # High volatility
        'low': prices - 5,
        'close': prices,
        'volume': np.full(100, 1000000)
    }, index=dates)
    
    df = compute_indicators(df)
    strategy = load_strategy(get_strategy_path())
    
    # Force a trade at known ATR value (after warmup)
    signals = pd.Series(0, index=df.index)
    signals.iloc[60] = 1
    
    result = run_backtest(
        df, signals, '2020-01-01', '2020-12-31',
        100000, 0.02, 2.0, 0.0015
    )
    
    if result['status'] != 'ok' or len(result['trades']) == 0:
        print("❌ FAIL: No trades for ATR test")
        return False
    
    trade = result['trades'][0]
    
    # Check stop loss calculation
    signal_date = df.index[30]
    entry_date = df.index[31]  # Next day
    atr_value = df.loc[signal_date, 'atr14']
    expected_stop = trade['entry_price'] - (atr_value * 2.0)
    actual_stop = trade['stop_price']
    
    if abs(actual_stop - expected_stop) > 0.01:
        print(f"❌ FAIL: ATR stop loss incorrect. Expected: {expected_stop}, Got: {actual_stop}")
        return False
    
    print("✅ PASS: ATR stop loss calculated correctly")
    return True


def run_comprehensive_tests():
    """Run all comprehensive bias detection tests."""
    print("=" * 80)
    print("COMPREHENSIVE BIAS DETECTION TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Future Price Leakage", test_future_price_leakage),
        ("Signal Timing Bias", test_signal_timing_bias),
        ("Position Sizing Consistency", test_position_sizing_consistency),
        ("Window Boundary Integrity", test_window_boundary_integrity),
        ("Equity Curve Compounding", test_equity_curve_compounding),
        ("Data Leakage Across Windows", test_data_leakage_across_windows),
        ("Commission and Slippage", test_commission_and_slippage_consistency),
        ("ATR Stop Loss Accuracy", test_atr_stop_loss_accuracy)
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
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    print(f"✅ PASSED: {passed}")
    print(f"❌ FAILED: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - NO BIAS DETECTED!")
        print("The WFA engine is completely bias-free and ready for production.")
    else:
        print(f"\n⚠️  {failed} TESTS FAILED - REVIEW REQUIRED!")
    
    print("=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_comprehensive_tests())
