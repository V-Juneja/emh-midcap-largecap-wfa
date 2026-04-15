import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import signal_generator
sys.path.insert(0, str(Path(__file__).parent.parent / 'WFA_ENGINE'))

from signal_generator import generate_signals, load_strategy

class TestSignalGenerator:
    
    def test_buy_signals_on_specific_rows(self):
        """Test 1: BUY signals on rows 5, 10, 15 where conditions are explicitly met"""
        # Create synthetic DataFrame with 20 rows
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        
        # Base values
        data = {
            'open': np.full(20, 100.0),
            'high': np.full(20, 105.0),
            'low': np.full(20, 95.0),
            'close': np.full(20, 100.0),
            'volume': np.full(20, 1000000),
            'ma20': np.full(20, 98.0),
            'ma50': np.full(20, 95.0),
            'bb_upper': np.full(20, 99.0),
            'bb_mid': np.full(20, 95.0),
            'bb_lower': np.full(20, 91.0),
            'atr14': np.full(20, 2.0),
            'roc10': np.full(20, 0.0),
            'vol_ma20': np.full(20, 1000000),
            'vol_ratio': np.full(20, 1.0)
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # Set BUY conditions on rows 5, 10, 15
        buy_rows = [5, 10, 15]
        for row in buy_rows:
            df.loc[df.index[row], 'ma20'] = 100.0  # ma20 > ma50
            df.loc[df.index[row], 'close'] = 102.0  # close > bb_upper
            df.loc[df.index[row], 'roc10'] = 0.05   # roc10 > 0.03
        
        # Create a simple strategy that returns 1 for these specific rows
        class TestStrategy:
            def get_signal(self, row, prev_row):
                if row.name in [df.index[i] for i in buy_rows]:
                    return 1
                return 0
        
        strategy = TestStrategy()
        signals = generate_signals(df, strategy)
        
        # Check signals
        assert len(signals) == 20
        assert signals.iloc[0] == 0  # Row 0 always 0
        
        for i in range(20):
            if i in buy_rows:
                assert signals.iloc[i] == 1, f"Row {i} should have BUY signal"
            else:
                assert signals.iloc[i] == 0, f"Row {i} should have no signal"
    
    def test_sell_signals_conditions(self):
        """Test 2: SELL signals when conditions are met"""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        
        data = {
            'open': np.full(10, 100.0),
            'high': np.full(10, 105.0),
            'low': np.full(10, 95.0),
            'close': np.full(10, 100.0),
            'volume': np.full(10, 1000000),
            'ma20': np.full(10, 98.0),
            'ma50': np.full(10, 95.0),
            'bb_upper': np.full(10, 105.0),
            'bb_mid': np.full(10, 100.0),
            'bb_lower': np.full(10, 95.0),
            'atr14': np.full(10, 2.0),
            'roc10': np.full(10, 0.0),
            'vol_ma20': np.full(10, 1000000),
            'vol_ratio': np.full(10, 1.0)
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # Set SELL conditions on rows 3 and 7
        df.loc[df.index[3], 'close'] = 95.0   # close < ma20
        df.loc[df.index[7], 'close'] = 94.0   # close < bb_lower
        
        # Create a simple strategy that returns -1 for these specific rows
        class TestStrategy:
            def get_signal(self, row, prev_row):
                if row.name == df.index[3] or row.name == df.index[7]:
                    return -1
                return 0
        
        strategy = TestStrategy()
        signals = generate_signals(df, strategy)
        
        # Check signals
        assert signals.iloc[3] == -1, "Row 3 should have SELL signal"
        assert signals.iloc[7] == -1, "Row 7 should have SELL signal"
        
        # Other rows should be 0
        for i in range(10):
            if i not in [3, 7]:
                assert signals.iloc[i] == 0, f"Row {i} should have no signal"
    
    def test_signals_same_index_as_input(self):
        """Test 3: Returned Series has same index as input DataFrame"""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        
        data = {
            'open': np.full(5, 100.0),
            'high': np.full(5, 105.0),
            'low': np.full(5, 95.0),
            'close': np.full(5, 100.0),
            'volume': np.full(5, 1000000),
            'ma20': np.full(5, 98.0),
            'ma50': np.full(5, 95.0),
            'bb_upper': np.full(5, 105.0),
            'bb_mid': np.full(5, 100.0),
            'bb_lower': np.full(5, 95.0),
            'atr14': np.full(5, 2.0),
            'roc10': np.full(5, 0.0),
            'vol_ma20': np.full(5, 1000000),
            'vol_ratio': np.full(5, 1.0)
        }
        
        df = pd.DataFrame(data, index=dates)
        
        class TestStrategy:
            def get_signal(self, row, prev_row):
                return 1 if row.name == df.index[2] else 0
        
        strategy = TestStrategy()
        signals = generate_signals(df, strategy)
        
        # Check index is identical
        assert signals.index.equals(df.index), "Signals index should match DataFrame index"
        assert list(signals.index) == list(df.index), "Signals index should be identical"
    
    def test_row_zero_always_zero(self):
        """Test 4: Row 0 always has signal 0 regardless of indicator values"""
        dates = pd.date_range('2020-01-01', periods=3, freq='D')
        
        data = {
            'open': np.full(3, 100.0),
            'high': np.full(3, 105.0),
            'low': np.full(3, 95.0),
            'close': np.full(3, 100.0),
            'volume': np.full(3, 1000000),
            'ma20': np.full(3, 100.0),  # Set to trigger BUY condition
            'ma50': np.full(3, 95.0),
            'bb_upper': np.full(3, 99.0),
            'bb_mid': np.full(3, 95.0),
            'bb_lower': np.full(3, 91.0),
            'atr14': np.full(3, 2.0),
            'roc10': np.full(3, 0.05),   # Set to trigger BUY condition
            'vol_ma20': np.full(3, 1000000),
            'vol_ratio': np.full(3, 1.0)
        }
        
        df = pd.DataFrame(data, index=dates)
        
        class TestStrategy:
            def get_signal(self, row, prev_row):
                # Always return 1 (BUY) for any row
                return 1
        
        strategy = TestStrategy()
        signals = generate_signals(df, strategy)
        
        # Row 0 should still be 0 even though strategy returns 1
        assert signals.iloc[0] == 0, "Row 0 should always be 0"
    
    def test_strategy_exception_handling(self):
        """Test 5: Strategy exceptions are handled without crashing"""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        
        data = {
            'open': np.full(5, 100.0),
            'high': np.full(5, 105.0),
            'low': np.full(5, 95.0),
            'close': np.full(5, 100.0),
            'volume': np.full(5, 1000000),
            'ma20': np.full(5, 98.0),
            'ma50': np.full(5, 95.0),
            'bb_upper': np.full(5, 105.0),
            'bb_mid': np.full(5, 100.0),
            'bb_lower': np.full(5, 95.0),
            'atr14': np.full(5, 2.0),
            'roc10': np.full(5, 0.0),
            'vol_ma20': np.full(5, 1000000),
            'vol_ratio': np.full(5, 1.0)
        }
        
        df = pd.DataFrame(data, index=dates)
        
        class TestStrategy:
            def get_signal(self, row, prev_row):
                if row.name == df.index[2]:
                    raise ValueError("Test exception")
                return 1
        
        strategy = TestStrategy()
        signals = generate_signals(df, strategy)
        
        # Row with exception should have signal 0
        assert signals.iloc[2] == 0, "Row with exception should have signal 0"
        
        # Other rows should have signal 1
        for i in range(5):
            if i != 2 and i != 0:  # Row 0 is always 0
                assert signals.iloc[i] == 1, f"Row {i} should have signal 1"
    
    def test_load_strategy_missing_function(self):
        """Test 6: load_strategy raises ValueError when get_signal is missing"""
        # Create a temporary strategy file without get_signal function
        strategy_content = '''
def some_other_function():
    pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(strategy_content)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Strategy module must have a 'get_signal' function"):
                load_strategy(temp_path)
        finally:
            os.unlink(temp_path)
        
        # Test with non-callable get_signal
        strategy_content_bad = '''
get_signal = "not a function"
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(strategy_content_bad)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="'get_signal' must be a callable function"):
                load_strategy(temp_path)
        finally:
            os.unlink(temp_path)
