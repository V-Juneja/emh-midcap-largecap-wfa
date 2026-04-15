import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to the path to import indicators
sys.path.insert(0, str(Path(__file__).parent.parent / 'WFA_ENGINE'))

from indicators import compute_indicators, get_required_warmup_period

class TestIndicators:
    
    def test_compute_indicators_all_columns_present(self):
        """Test 1: All 8 indicator columns are present after compute_indicators"""
        # Create synthetic DataFrame with 200 rows
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        
        # Realistic OHLCV data with upward trend
        base_price = 100
        trend = np.linspace(0, 50, 200)  # Upward trend
        noise = np.random.normal(0, 2, 200)
        
        close = base_price + trend + noise
        high = close + np.random.uniform(0.5, 2, 200)
        low = close - np.random.uniform(0.5, 2, 200)
        open_price = close + np.random.uniform(-1, 1, 200)
        volume = np.random.randint(1000000, 5000000, 200)
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        result = compute_indicators(df)
        
        # Check all indicator columns are present
        expected_columns = ['ma20', 'ma50', 'bb_upper', 'bb_mid', 'bb_lower', 'atr14', 'roc10', 'vol_ma20', 'vol_ratio']
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
        
        # Check original columns are still present
        original_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in original_columns:
            assert col in result.columns, f"Missing original column: {col}"
    
    def test_ma50_nan_early_rows(self):
        """Test 2: Rows 0-49 have NaN in ma50, rows 50+ have no NaN in ma50"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        close = 100 + np.random.normal(0, 5, 100)
        df = pd.DataFrame({
            'open': close + np.random.uniform(-1, 1, 100),
            'high': close + np.random.uniform(0.5, 2, 100),
            'low': close - np.random.uniform(0.5, 2, 100),
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        result = compute_indicators(df)
        
        # Rows 0-49 should have NaN in ma50
        for i in range(49):  # 0-indexed, so rows 0-48 should be NaN
            assert pd.isna(result['ma50'].iloc[i]), f"Row {i} should have NaN in ma50"
        
        # Row 49+ should have no NaN in ma50 (50th row in 1-based indexing)
        for i in range(49, 100):
            assert not pd.isna(result['ma50'].iloc[i]), f"Row {i} should not have NaN in ma50"
    
    def test_bollinger_bands_ordering(self):
        """Test 3: bb_upper >= bb_mid >= bb_lower for all non-NaN rows"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        close = 100 + np.random.normal(0, 5, 100)
        df = pd.DataFrame({
            'open': close + np.random.uniform(-1, 1, 100),
            'high': close + np.random.uniform(0.5, 2, 100),
            'low': close - np.random.uniform(0.5, 2, 100),
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        result = compute_indicators(df)
        
        # Check Bollinger Bands ordering for non-NaN rows
        for i in range(len(result)):
            if not (pd.isna(result['bb_upper'].iloc[i]) or pd.isna(result['bb_mid'].iloc[i]) or pd.isna(result['bb_lower'].iloc[i])):
                assert result['bb_upper'].iloc[i] >= result['bb_mid'].iloc[i], f"Row {i}: bb_upper < bb_mid"
                assert result['bb_mid'].iloc[i] >= result['bb_lower'].iloc[i], f"Row {i}: bb_mid < bb_lower"
    
    def test_atr14_positive(self):
        """Test 4: atr14 is always positive for all non-NaN rows"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        close = 100 + np.random.normal(0, 5, 100)
        df = pd.DataFrame({
            'open': close + np.random.uniform(-1, 1, 100),
            'high': close + np.random.uniform(0.5, 2, 100),
            'low': close - np.random.uniform(0.5, 2, 100),
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        result = compute_indicators(df)
        
        # Check ATR14 is positive for non-NaN rows
        for i in range(len(result)):
            if not pd.isna(result['atr14'].iloc[i]):
                assert result['atr14'].iloc[i] > 0, f"Row {i}: atr14 should be positive"
    
    def test_vol_ratio_identical_volumes(self):
        """Test 5: vol_ratio equals exactly 1.0 when all 20 volume values are identical"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        
        close = 100 + np.random.normal(0, 5, 30)
        # Use identical volume values
        volume = np.full(30, 2000000)
        
        df = pd.DataFrame({
            'open': close + np.random.uniform(-1, 1, 30),
            'high': close + np.random.uniform(0.5, 2, 30),
            'low': close - np.random.uniform(0.5, 2, 30),
            'close': close,
            'volume': volume
        }, index=dates)
        
        result = compute_indicators(df)
        
        # For rows 19+ (after 20-period window), vol_ratio should be exactly 1.0
        for i in range(19, len(result)):
            if not pd.isna(result['vol_ratio'].iloc[i]):
                assert abs(result['vol_ratio'].iloc[i] - 1.0) < 1e-10, f"Row {i}: vol_ratio should be 1.0"
    
    def test_input_dataframe_not_modified(self):
        """Test 6: Input DataFrame is not modified"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        close = 100 + np.random.normal(0, 5, 100)
        original_df = pd.DataFrame({
            'open': close + np.random.uniform(-1, 1, 100),
            'high': close + np.random.uniform(0.5, 2, 100),
            'low': close - np.random.uniform(0.5, 2, 100),
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Store original shape and values
        original_shape = original_df.shape
        original_columns = original_df.columns.tolist()
        original_values = original_df.copy()
        
        # Compute indicators
        result = compute_indicators(original_df)
        
        # Check original DataFrame is unchanged
        assert original_df.shape == original_shape, "Original DataFrame shape changed"
        assert original_df.columns.tolist() == original_columns, "Original DataFrame columns changed"
        pd.testing.assert_frame_equal(original_df, original_values, "Original DataFrame values changed")
        
        # Check result has additional columns
        assert result.shape[1] > original_df.shape[1], "Result should have more columns"
    
    def test_get_required_warmup_period(self):
        """Test 7: get_required_warmup_period() returns 50"""
        warmup = get_required_warmup_period()
        assert warmup == 50, f"Expected warmup period 50, got {warmup}"
