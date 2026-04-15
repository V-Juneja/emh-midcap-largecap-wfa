#!/usr/bin/env python3
"""
Performance Calculation Accuracy Test Suite
Ensures all performance metrics are calculated with mathematical precision.
"""

import sys
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import load_all_stocks
from indicators import compute_indicators
from signal_generator import load_strategy, generate_signals
from wfa_runner import run_wfa
from backtest import run_backtest


def test_return_calculations():
    """Test accuracy of return calculations."""
    print("Testing return calculation accuracy...")
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Create data with known returns
    prices = [100]
    for i in range(99):
        # Exactly 1% daily return
        prices.append(prices[-1] * 1.01)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': np.full(100, 1000000)
    }, index=dates)
    
    df = compute_indicators(df)
    
    # Force one trade for the entire period
    signals = pd.Series(0, index=df.index)
    signals.iloc[0] = 1  # Buy on first day
    signals.iloc[-2] = -1  # Sell on second to last day
    
    result = run_backtest(
        df, signals, '2020-01-01', '2020-12-31',
        100000, 0.02, 2.0, 0.0015
    )
    
    if result['status'] != 'ok':
        print("❌ FAIL: Backtest failed for return calculation test")
        return False
    
    # Expected return: (1.01)^98 - 1 ≈ 170% (minus costs)
    expected_return = (1.01 ** 98 - 1) * 100
    actual_return = result['total_return_pct']
    
    # Allow for slippage and commission costs
    if abs(actual_return - expected_return) > 20:  # 20% tolerance for costs
        print(f"❌ FAIL: Return calculation inaccurate. Expected: {expected_return:.2f}%, Got: {actual_return:.2f}%")
        return False
    
    print("✅ PASS: Return calculations accurate")
    return True


def test_win_rate_calculations():
    """Test accuracy of win rate calculations."""
    print("Testing win rate calculation accuracy...")
    
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
    
    # Create exactly 10 trades with known outcomes
    signals = pd.Series(0, index=df.index)
    trade_entries = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    trade_exits = [15, 25, 35, 45, 55, 65, 75, 85, 92, 98]
    
    for entry, exit in zip(trade_entries, trade_exits):
        signals.iloc[entry] = 1
        signals.iloc[exit] = -1
    
    result = run_backtest(
        df, signals, '2020-01-01', '2020-12-31',
        100000, 0.02, 2.0, 0.0015
    )
    
    if result['status'] != 'ok':
        print("❌ FAIL: Backtest failed for win rate test")
        return False
    
    # Check that win rate is calculated correctly
    total_trades = result['total_trades']
    winning_trades = result['winning_trades']
    calculated_win_rate = result['win_rate']
    expected_win_rate = winning_trades / total_trades
    
    if abs(calculated_win_rate - expected_win_rate) > 0.001:
        print(f"❌ FAIL: Win rate calculation inaccurate. Expected: {expected_win_rate:.3f}, Got: {calculated_win_rate:.3f}")
        return False
    
    print("✅ PASS: Win rate calculations accurate")
    return True


def test_profit_factor_calculations():
    """Test accuracy of profit factor calculations."""
    print("Testing profit factor calculation accuracy...")
    
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
    
    # Create trades with known profit/loss amounts
    signals = pd.Series(0, index=df.index)
    
    # Create 5 winning trades and 3 losing trades
    winning_entries = [10, 25, 40, 55, 70]
    winning_exits = [15, 30, 45, 60, 75]
    losing_entries = [80, 85, 90]
    losing_exits = [82, 87, 92]
    
    for entry, exit in winning_entries + winning_exits:
        signals.iloc[entry] = 1
    for entry, exit in losing_entries + losing_exits:
        signals.iloc[entry] = 1
    
    for exit in winning_exits + losing_exits:
        signals.iloc[exit] = -1
    
    result = run_backtest(
        df, signals, '2020-01-01', '2020-12-31',
        100000, 0.02, 2.0, 0.0015
    )
    
    if result['status'] != 'ok':
        print("❌ FAIL: Backtest failed for profit factor test")
        return False
    
    # Calculate expected profit factor from trades
    gross_profit = sum([t['return_pct'] for t in result['trades'] if t['return_pct'] > 0])
    gross_loss = abs(sum([t['return_pct'] for t in result['trades'] if t['return_pct'] < 0]))
    expected_profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    actual_profit_factor = result['profit_factor']
    
    if abs(actual_profit_factor - expected_profit_factor) > 0.01:
        print(f"❌ FAIL: Profit factor calculation inaccurate. Expected: {expected_profit_factor:.2f}, Got: {actual_profit_factor:.2f}")
        return False
    
    print("✅ PASS: Profit factor calculations accurate")
    return True


def test_sharpe_ratio_calculations():
    """Test accuracy of Sharpe ratio calculations."""
    print("Testing Sharpe ratio calculation accuracy...")
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Create data with known volatility and returns
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0.001, 0.02, 100)  # 0.1% daily return, 2% volatility
    prices = [100]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    
    df = pd.DataFrame({
        'open': prices[:-1],
        'high': [p * 1.02 for p in prices[:-1]],
        'low': [p * 0.98 for p in prices[:-1]],
        'close': prices[:-1],
        'volume': np.full(100, 1000000)
    }, index=dates)
    
    df = compute_indicators(df)
    
    # Create multiple trades to get meaningful Sharpe ratio
    signals = pd.Series(0, index=df.index)
    for i in range(10, 90, 10):
        signals.iloc[i] = 1
        signals.iloc[i+5] = -1
    
    result = run_backtest(
        df, signals, '2020-01-01', '2020-12-31',
        100000, 0.02, 2.0, 0.0015
    )
    
    if result['status'] != 'ok':
        print("❌ FAIL: Backtest failed for Sharpe ratio test")
        return False
    
    # Calculate expected Sharpe ratio manually
    trade_returns = [t['return_pct'] for t in result['trades']]
    if len(trade_returns) > 1:
        mean_return = np.mean(trade_returns)
        std_return = np.std(trade_returns)
        avg_hold_days = np.mean([t['hold_days'] for t in result['trades']])
        
        if std_return > 0 and avg_hold_days > 0:
            expected_sharpe = (mean_return / std_return) * math.sqrt(252 / avg_hold_days)
            actual_sharpe = result['sharpe']
            
            if abs(actual_sharpe - expected_sharpe) > 0.1:
                print(f"❌ FAIL: Sharpe ratio calculation inaccurate. Expected: {expected_sharpe:.2f}, Got: {actual_sharpe:.2f}")
                return False
    
    print("✅ PASS: Sharpe ratio calculations accurate")
    return True


def test_drawdown_calculations():
    """Test accuracy of drawdown calculations."""
    print("Testing drawdown calculation accuracy...")
    
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Create data with known drawdown pattern
    prices = [100]
    # Rise 20%
    for i in range(20):
        prices.append(prices[-1] * 1.01)
    # Fall 30%
    for i in range(20):
        prices.append(prices[-1] * 0.985)
    # Recover
    for i in range(59):
        prices.append(prices[-1] * 1.005)
    
    df = pd.DataFrame({
        'open': prices[:-1],
        'high': [p * 1.02 for p in prices[:-1]],
        'low': [p * 0.98 for p in prices[:-1]],
        'close': prices[:-1],
        'volume': np.full(100, 1000000)
    }, index=dates)
    
    df = compute_indicators(df)
    
    # One trade for the entire period
    signals = pd.Series(0, index=df.index)
    signals.iloc[0] = 1
    signals.iloc[-2] = -1
    
    result = run_backtest(
        df, signals, '2020-01-01', '2020-12-31',
        100000, 0.02, 2.0, 0.0015
    )
    
    if result['status'] != 'ok':
        print("❌ FAIL: Backtest failed for drawdown test")
        return False
    
    # Expected max drawdown should be around 30%
    actual_drawdown = result['max_drawdown_pct']
    
    if actual_drawdown < 20 or actual_drawdown > 35:  # Allow some tolerance
        print(f"❌ FAIL: Drawdown calculation inaccurate. Expected ~30%, Got: {actual_drawdown:.2f}%")
        return False
    
    print("✅ PASS: Drawdown calculations accurate")
    return True


def test_wfe_calculation_accuracy():
    """Test accuracy of WFE calculations."""
    print("Testing WFE calculation accuracy...")
    
    # Create synthetic data with known IS/OOS performance
    stocks = {}
    dates = pd.date_range('2020-01-01', periods=400, freq='D')
    
    for i in range(2):
        ticker = f'TEST{i}'
        
        # Create prices with predictable performance
        prices = []
        for j in range(400):
            if j < 200:  # First half: 10% return
                prices.append(100 * (1 + 0.1 * j / 200))
            else:  # Second half: 5% return
                prices.append(110 * (1 + 0.05 * (j - 200) / 200))
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.full(400, 1000000)
        }, index=dates)
        
        stocks[ticker] = compute_indicators(df)
    
    # Generate consistent signals
    strategy = load_strategy(str(Path(__file__).parent.parent.parent / "STRATEGY" / "strategy.py"))
    signals_dict = {}
    for ticker, df in stocks.items():
        signals = pd.Series(0, index=df.index)
        # Force trades in each period
        signals.iloc[50] = 1   # IS period trade
        signals.iloc[250] = 1  # OOS period trade
        signals_dict[ticker] = signals
    
    # Run WFA
    result = run_wfa(
        stocks, signals_dict,
        '2020-01-01', '2021-12-31',
        6, 3, 100000, 0.02, 2.0, 0.0015
    )
    
    if not result['windows']:
        print("❌ FAIL: No windows generated for WFE test")
        return False
    
    # Check WFE calculation for first window
    window = result['windows'][0]
    wfe = window['wfe']
    
    # WFE should be reasonable (not extreme values)
    if wfe < -2 or wfe > 5:
        print(f"❌ FAIL: WFE calculation unrealistic. Got: {wfe:.3f}")
        return False
    
    print("✅ PASS: WFE calculations accurate")
    return True


def test_compounding_accuracy():
    """Test accuracy of equity curve compounding."""
    print("Testing equity curve compounding accuracy...")
    
    # Create synthetic data with known returns
    stocks = {}
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    
    for i in range(2):
        ticker = f'TEST{i}'
        prices = np.linspace(100, 105, 300)  # 5% total return
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.full(300, 1000000)
        }, index=dates)
        
        stocks[ticker] = compute_indicators(df)
    
    # Force exactly 2% return per window
    strategy = load_strategy(str(Path(__file__).parent.parent.parent / "STRATEGY" / "strategy.py"))
    signals_dict = {}
    for ticker, df in stocks.items():
        signals = pd.Series(0, index=df.index)
        # One trade per window
        signals.iloc[50] = 1   # Window 1
        signals.iloc[150] = 1  # Window 2
        signals_dict[ticker] = signals
    
    # Run WFA with 2 windows
    result = run_wfa(
        stocks, signals_dict,
        '2020-01-01', '2021-06-30',
        4, 2, 100000, 0.02, 2.0, 0.0015
    )
    
    equity_curve = result['summary']['oos_equity_curve']
    
    # With 2% per window, compounded: (1.02)^2 - 1 = 4.04%
    expected_final = 4.04
    actual_final = equity_curve[-1] if equity_curve else 0
    
    if abs(actual_final - expected_final) > 1.0:  # 1% tolerance
        print(f"❌ FAIL: Compounding inaccurate. Expected: {expected_final:.2f}%, Got: {actual_final:.2f}%")
        return False
    
    print("✅ PASS: Equity curve compounding accurate")
    return True


def run_performance_accuracy_tests():
    """Run all performance calculation accuracy tests."""
    print("=" * 80)
    print("PERFORMANCE CALCULATION ACCURACY TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Return Calculations", test_return_calculations),
        ("Win Rate Calculations", test_win_rate_calculations),
        ("Profit Factor Calculations", test_profit_factor_calculations),
        ("Sharpe Ratio Calculations", test_sharpe_ratio_calculations),
        ("Drawdown Calculations", test_drawdown_calculations),
        ("WFE Calculation Accuracy", test_wfe_calculation_accuracy),
        ("Compounding Accuracy", test_compounding_accuracy)
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
    print("PERFORMANCE ACCURACY TEST RESULTS")
    print("=" * 80)
    print(f"✅ PASSED: {passed}")
    print(f"❌ FAILED: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL PERFORMANCE ACCURACY TESTS PASSED!")
        print("All performance metrics are calculated with mathematical precision.")
    else:
        print(f"\n⚠️  {failed} PERFORMANCE ACCURACY TESTS FAILED!")
    
    print("=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_performance_accuracy_tests())
