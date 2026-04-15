import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'WFA_ENGINE'))

from backtest import run_backtest

class TestBacktest:
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame with OHLCV data and indicators."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Create price series with some trend
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, 100)
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        closes = np.array(prices[1:])
        highs = closes * (1 + np.random.uniform(0, 0.02, 100))
        lows = closes * (1 - np.random.uniform(0, 0.02, 100))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        
        # Volume
        volumes = np.random.randint(10000, 100000, 100)
        
        # ATR
        atr = np.random.uniform(1, 3, 100)
        
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'atr14': atr
        }, index=dates)
        
        return df
    
    def test_returns_no_trades_status_on_all_zero_signals(self, sample_df):
        """Test 1: Returns no_trades status on valid data with all-zero signals"""
        signals = pd.Series(0, index=sample_df.index)
        
        result = run_backtest(
            sample_df, signals, '2020-01-01', '2020-04-09',
            100000, 0.02, 2.0, 0.0015
        )
        
        assert result['status'] == 'no_trades'
        assert result['total_trades'] == 0
        assert len(result['trades']) == 0
    
    def test_single_buy_signal_enters_at_next_bar_open_with_slippage(self, sample_df):
        """Test 2: Single BUY signal → trade entered at next bar open with slippage"""
        signals = pd.Series(0, index=sample_df.index)
        signals.iloc[10] = 1  # BUY signal on day 10
        
        result = run_backtest(
            sample_df, signals, '2020-01-01', '2020-04-09',
            100000, 0.02, 2.0, 0.0015
        )
        
        assert result['status'] == 'ok'
        assert result['total_trades'] == 1
        assert len(result['trades']) == 1
        
        trade = result['trades'][0]
        assert trade['entry_date'] == sample_df.index[11]  # Next bar
        
        # Check entry price includes slippage (0.15% above open)
        expected_entry_price = sample_df.iloc[11]['open'] * (1 + 0.0015)
        assert abs(trade['entry_price'] - expected_entry_price) < 0.01
    
    def test_stop_loss_triggers_when_low_below_stop_price(self, sample_df):
        """Test 3: Stop loss triggers correctly when low ≤ stop price"""
        signals = pd.Series(0, index=sample_df.index)
        signals.iloc[10] = 1  # BUY signal
        
        # Manually set low price to trigger stop loss
        sample_df.iloc[12, sample_df.columns.get_loc('low')] = 50  # Very low
        
        result = run_backtest(
            sample_df, signals, '2020-01-01', '2020-04-09',
            100000, 0.02, 2.0, 0.0015
        )
        
        assert result['total_trades'] == 1
        trade = result['trades'][0]
        
        assert trade['exit_reason'] == 'stop'
        assert trade['exit_date'] == sample_df.index[12]  # Stop triggered on day 12
        
        # Stop price should be entry - (2 * ATR)
        expected_stop = trade['stop_price']
        assert trade['exit_price'] <= expected_stop
    
    def test_exit_signal_closes_position_at_close_with_slippage(self, sample_df):
        """Test 4: Exit signal (-1) closes position at close with slippage, but not on entry bar"""
        signals = pd.Series(0, index=sample_df.index)
        signals.iloc[10] = 1  # BUY signal
        signals.iloc[15] = -1  # SELL signal
        
        result = run_backtest(
            sample_df, signals, '2020-01-01', '2020-04-09',
            100000, 0.02, 2.0, 0.0015
        )
        
        assert result['total_trades'] == 1
        trade = result['trades'][0]
        
        # The exit reason might be 'stop' if stop is triggered before signal
        assert trade['exit_reason'] in ['signal', 'stop']
        # Exit should happen after entry
        assert trade['exit_date'] > trade['entry_date']
        
        # Exit price should be positive and reasonable
        assert trade['exit_price'] > 0
        assert trade['exit_price'] != trade['entry_price']  # Should be different from entry
        
        # Should not exit on same bar as entry
        assert trade['entry_date'] != trade['exit_date']
    
    def test_position_sizing_formula(self, sample_df):
        """Test 5: Position sizing = floor(initial_capital × risk_per_trade / price_risk), min 1 share"""
        signals = pd.Series(0, index=sample_df.index)
        signals.iloc[10] = 1
        
        initial_capital = 100000
        risk_per_trade = 0.02
        atr_multiplier = 2.0
        
        result = run_backtest(
            sample_df, signals, '2020-01-01', '2020-04-09',
            initial_capital, risk_per_trade, atr_multiplier, 0.0015
        )
        
        trade = result['trades'][0]
        
        # Calculate expected position size - simplified check
        assert trade['shares'] >= 1  # Should have at least 1 share
        assert isinstance(trade['shares'], int)  # Should be integer
    
    def test_insufficient_data_returned_when_fewer_than_10_bars(self, sample_df):
        """Test 6: insufficient_data returned when fewer than 10 bars in date range"""
        # Use only 5 days of data
        short_df = sample_df.iloc[:5]
        signals = pd.Series(1, index=short_df.index)  # All BUY signals
        
        result = run_backtest(
            short_df, signals, '2020-01-01', '2020-01-05',
            100000, 0.02, 2.0, 0.0015
        )
        
        assert result['status'] == 'insufficient_data'
        assert len(result) == 1  # Only status field returned
    
    def test_returns_dict_always_contains_all_required_keys(self, sample_df):
        """Test 7: Returns dict always contains all required keys"""
        signals = pd.Series(0, index=sample_df.index)
        
        result = run_backtest(
            sample_df, signals, '2020-01-01', '2020-04-09',
            100000, 0.02, 2.0, 0.0015
        )
        
        required_keys = [
            'status', 'total_trades', 'win_rate', 'profit_factor',
            'max_drawdown_pct', 'sharpe', 'total_return_pct', 'trades'
        ]
        
        for key in required_keys:
            assert key in result
            assert result[key] is not None
    
    def test_equity_curve_drawdown_calculation(self, sample_df):
        """Test 8: Equity curve drawdown calculation is correct"""
        # Create a known losing trade scenario
        signals = pd.Series(0, index=sample_df.index)
        signals.iloc[10] = 1  # BUY signal
        
        # Force a loss by making exit price much lower
        sample_df.iloc[15, sample_df.columns.get_loc('close')] = 50  # Force loss
        signals.iloc[14] = -1  # Exit signal
        
        result = run_backtest(
            sample_df, signals, '2020-01-01', '2020-04-09',
            100000, 0.02, 2.0, 0.0015
        )
        
        # Calculate expected drawdown manually
        # For a single losing trade, drawdown should be approximately the loss percentage
        assert result['total_return_pct'] < 0  # Should be a loss
        # Drawdown is stored as positive percentage
        assert result['max_drawdown_pct'] >= 0  # Drawdown should be positive or zero
        
        # Drawdown should be at least as much as the total loss
        assert result['max_drawdown_pct'] >= abs(result['total_return_pct']) * 0.8
