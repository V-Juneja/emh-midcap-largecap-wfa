import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'WFA_ENGINE'))

from bias_validator import (
    check_indicator_lookahead,
    check_signal_lookahead,
    check_entry_price_lookahead,
    check_oos_contamination,
    check_exit_before_entry,
    check_survivorship_warning,
    run_all_checks
)

class TestBiasValidator:
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame with OHLCV data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        closes = 100 + np.cumsum(np.random.normal(0, 1, 100))
        opens = closes + np.random.normal(0, 0.5, 100)
        highs = np.maximum(opens, closes) + np.random.uniform(0, 2, 100)
        lows = np.minimum(opens, closes) - np.random.uniform(0, 2, 100)
        volumes = np.random.randint(100000, 1000000, 100)
        
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def strategy_module(self):
        """Create a simple strategy module for testing."""
        class MockStrategy:
            @staticmethod
            def get_signal(current_row, prev_row):
                # Simple moving average crossover
                if (current_row['ma20'] > current_row['ma50'] and 
                    prev_row['ma20'] <= prev_row['ma50']):
                    return 1
                elif (current_row['ma20'] < current_row['ma50'] and 
                      prev_row['ma20'] >= prev_row['ma50']):
                    return -1
                return 0
        
        return MockStrategy()
    
    def test_indicator_lookahead_bias_detected(self, sample_df):
        """Test 1: Detect lookahead bias in fake indicator using future data."""
        # Add a fake indicator that uses future data
        sample_df['fake_indicator'] = sample_df['close'].shift(-1)  # Future leak!
        
        biased_cols = check_indicator_lookahead(sample_df, ['fake_indicator'])
        
        assert 'fake_indicator' in biased_cols, "Should detect lookahead bias in fake_indicator"
    
    def test_indicator_lookahead_clean_sma(self, sample_df):
        """Test 2: Clean SMA indicator should not be flagged."""
        # Add clean SMA indicators
        sample_df['ma20'] = sample_df['close'].rolling(20, min_periods=1).mean()
        sample_df['ma50'] = sample_df['close'].rolling(50, min_periods=1).mean()
        
        biased_cols = check_indicator_lookahead(sample_df, ['ma20', 'ma50'])
        
        assert len(biased_cols) == 0, f"Clean SMA indicators should not be flagged, got: {biased_cols}"
    
    def test_entry_lookahead_same_bar(self, sample_df):
        """Test 3: Detect entry on same bar as signal."""
        # Create trade where entry_date == signal_date
        signal_date = sample_df.index[10]
        entry_date = signal_date  # Same bar - this is bad!
        
        trades = [{
            'entry_date': entry_date,
            'exit_date': sample_df.index[15],
            'entry_price': 100.0,
            'exit_price': 105.0
        }]
        
        signals = pd.Series(0, index=sample_df.index)
        signals.iloc[10] = 1  # Signal on same day as entry
        
        problematic_trades = check_entry_price_lookahead(trades, signals)
        
        assert len(problematic_trades) == 1, "Should flag trade where entry == signal date"
        assert problematic_trades[0]['entry_date'] == entry_date
    
    def test_entry_lookahead_valid_next_bar(self, sample_df):
        """Test 4: Valid entry on next bar should not be flagged."""
        # Create trade where entry_date is one bar after signal_date
        signal_date = sample_df.index[10]
        entry_date = sample_df.index[11]  # Next bar - this is correct!
        
        trades = [{
            'entry_date': entry_date,
            'exit_date': sample_df.index[15],
            'entry_price': 100.0,
            'exit_price': 105.0
        }]
        
        signals = pd.Series(0, index=sample_df.index)
        signals.iloc[10] = 1  # Signal one day before entry
        
        problematic_trades = check_entry_price_lookahead(trades, signals)
        
        assert len(problematic_trades) == 0, "Valid next-bar entry should not be flagged"
    
    def test_oos_contamination_overlap(self, sample_df):
        """Test 5: Detect OOS contamination when windows overlap."""
        # Create window with overlap: oos_start == is_end
        windows = [{
            'window_id': 0,
            'is_start': '2020-01-01',
            'is_end': '2020-06-30',
            'oos_start': '2020-06-30',  # Same day as IS end - overlap!
            'oos_end': '2020-09-30'
        }]
        
        contaminated = check_oos_contamination(windows)
        
        assert 0 in contaminated, "Should detect OOS contamination in window 0"
    
    def test_exit_before_entry(self, sample_df):
        """Test 6: Detect trades that exit before they enter."""
        # Create trade with exit before entry (impossible!)
        entry_date = sample_df.index[20]
        exit_date = sample_df.index[15]  # Before entry - this is bad!
        
        trades = [{
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': 100.0,
            'exit_price': 95.0
        }]
        
        problematic_trades = check_exit_before_entry(trades)
        
        assert len(problematic_trades) == 1, "Should detect exit before entry"
        assert problematic_trades[0]['exit_date'] == exit_date
    
    def test_survivorship_warning_late_starter(self, sample_df):
        """Test 7: Detect survivorship bias for late-starting stocks."""
        # Create stocks dict where one stock starts 200 days after expected start
        early_stock = sample_df.copy()
        late_stock = sample_df.copy()
        
        # Shift early stock to start at expected start
        early_start = pd.Timestamp('2015-01-01')
        early_stock.index = early_start + (early_stock.index - early_stock.index[0])
        
        # Shift late stock to start 200 days later
        late_start = pd.Timestamp('2015-07-20')  # 200 days after 2015-01-01
        late_stock.index = late_start + (late_stock.index - late_stock.index[0])
        
        stocks = {
            'EARLY_STOCK': early_stock,
            'LATE_STOCK': late_stock
        }
        
        late_starters = check_survivorship_warning(stocks, "2015-01-01")
        
        assert 'LATE_STOCK' in late_starters, "Should detect late-starting stock"
        assert 'EARLY_STOCK' not in late_starters, "Should not flag early-starting stock"
    
    def test_run_all_checks_clean_inputs(self, sample_df, strategy_module):
        """Test 8: All checks should pass with clean inputs."""
        # Add clean indicators - using proper calculations matching indicators.py
        sample_df['ma20'] = sample_df['close'].rolling(20, min_periods=20).mean()
        sample_df['ma50'] = sample_df['close'].rolling(50, min_periods=50).mean()
        bb_std = sample_df['close'].rolling(20, min_periods=20).std()
        sample_df['bb_upper'] = sample_df['ma20'] + (bb_std * 2)
        sample_df['bb_mid'] = sample_df['ma20']
        sample_df['bb_lower'] = sample_df['ma20'] - (bb_std * 2)
        
        # Proper ATR using True Range (matching indicators.py)
        prev_close = sample_df['close'].shift(1)
        tr1 = sample_df['high'] - sample_df['low']
        tr2 = abs(sample_df['high'] - prev_close)
        tr3 = abs(sample_df['low'] - prev_close)
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        sample_df['atr14'] = true_range.rolling(14, min_periods=14).mean()
        
        sample_df['roc10'] = (sample_df['close'] - sample_df['close'].shift(10)) / sample_df['close'].shift(10)
        sample_df['vol_ma20'] = sample_df['volume'].rolling(20, min_periods=20).mean()
        sample_df['vol_ratio'] = sample_df['volume'] / sample_df['vol_ma20']
        
        # Create clean trades
        trades = [{
            'entry_date': sample_df.index[11],  # One bar after signal
            'exit_date': sample_df.index[20],
            'entry_price': 100.0,
            'exit_price': 105.0
        }]
        
        # Create clean signals
        signals = pd.Series(0, index=sample_df.index)
        signals.iloc[10] = 1
        
        # Create clean WFA output
        wfa_output = {
            'windows': [{
                'window_id': 0,
                'is_start': '2020-01-01',
                'is_end': '2020-06-29',  # One day before OOS starts
                'oos_start': '2020-06-30',
                'oos_end': '2020-09-30'
            }]
        }
        
        # Create clean stocks dict
        stocks = {'SAMPLE_STOCK': sample_df}
        
        # Run all checks
        results = run_all_checks(
            df_sample=sample_df,
            strategy_module=strategy_module,
            wfa_output=wfa_output,
            trades_sample=trades,
            signals_sample=signals,
            stocks=stocks
        )
        
        # Verify all checks pass
        assert results["overall_clean"] is True, "All checks should pass with clean inputs"
        assert len(results["lookahead_indicator_cols"]) == 0, "No indicator lookahead bias"
        assert results["signal_lookahead_detected"] is False, "No signal lookahead bias"
        assert len(results["entry_lookahead_trades"]) == 0, "No entry lookahead bias"
        assert len(results["oos_contaminated_windows"]) == 0, "No OOS contamination"
        assert len(results["exit_before_entry_trades"]) == 0, "No exit-before-entry violations"
        # Survivorship warning is informational, so we don't check it
