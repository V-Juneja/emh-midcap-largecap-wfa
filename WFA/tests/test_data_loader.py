import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add the parent directory to the path to import data_loader
sys.path.insert(0, str(Path(__file__).parent.parent / 'WFA_ENGINE'))

from data_loader import load_single_stock, load_all_stocks

class TestDataLoader:
    
    def test_load_single_stock_valid_csv(self):
        """Test 1: Load a valid CSV with correct columns and 600 rows"""
        # Create synthetic valid CSV
        dates = pd.date_range('2020-01-01', periods=600, freq='D')
        data = {
            'Date': dates,
            'Open': [100.0 + i * 0.1 for i in range(600)],
            'High': [101.0 + i * 0.1 for i in range(600)],
            'Low': [99.0 + i * 0.1 for i in range(600)],
            'Close': [100.5 + i * 0.1 for i in range(600)],
            'Volume': [1000000 + i * 1000 for i in range(600)]
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            result = load_single_stock(temp_path)
            
            # Assertions
            assert result is not None
            assert len(result) == 600
            assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
            assert isinstance(result.index, pd.DatetimeIndex)
            assert result.index.is_monotonic_increasing
            
            # Check data types
            assert result['open'].dtype == 'float64'
            assert result['high'].dtype == 'float64'
            assert result['low'].dtype == 'float64'
            assert result['close'].dtype == 'float64'
            assert result['volume'].dtype == 'int64'
            
            # Check no nulls
            assert not result.isnull().any().any()
            
        finally:
            os.unlink(temp_path)
    
    def test_load_single_stock_missing_column(self):
        """Test 2: CSV with missing volume column should return None"""
        dates = pd.date_range('2020-01-01', periods=600, freq='D')
        data = {
            'Date': dates,
            'Open': [100.0 + i * 0.1 for i in range(600)],
            'High': [101.0 + i * 0.1 for i in range(600)],
            'Low': [99.0 + i * 0.1 for i in range(600)],
            'Close': [100.5 + i * 0.1 for i in range(600)]
            # Missing Volume column
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            result = load_single_stock(temp_path)
            assert result is None
            
        finally:
            os.unlink(temp_path)
    
    def test_load_single_stock_insufficient_rows(self):
        """Test 3: CSV with only 200 rows should return None"""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        data = {
            'Date': dates,
            'Open': [100.0 + i * 0.1 for i in range(200)],
            'High': [101.0 + i * 0.1 for i in range(200)],
            'Low': [99.0 + i * 0.1 for i in range(200)],
            'Close': [100.5 + i * 0.1 for i in range(200)],
            'Volume': [1000000 + i * 1000 for i in range(200)]
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            result = load_single_stock(temp_path)
            assert result is None
            
        finally:
            os.unlink(temp_path)
    
    def test_load_single_stock_duplicate_dates(self):
        """Test 4: CSV with 5 duplicate dates should have no duplicates in output"""
        # Create data with duplicate dates
        dates = list(pd.date_range('2020-01-01', periods=595, freq='D'))
        # Add 5 duplicate dates
        dates.extend([pd.Timestamp('2020-01-01')] * 5)
        
        data = {
            'Date': dates,
            'Open': [100.0 + i * 0.1 for i in range(600)],
            'High': [101.0 + i * 0.1 for i in range(600)],
            'Low': [99.0 + i * 0.1 for i in range(600)],
            'Close': [100.5 + i * 0.1 for i in range(600)],
            'Volume': [1000000 + i * 1000 for i in range(600)]
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            result = load_single_stock(temp_path)
            
            assert result is not None
            assert len(result) == 595  # Should have 595 unique dates
            assert not result.index.duplicated().any()
            
        finally:
            os.unlink(temp_path)
    
    def test_load_single_stock_zero_close_values(self):
        """Test 5: CSV with 3 rows where close = 0 should drop those rows"""
        dates = pd.date_range('2020-01-01', periods=503, freq='D')
        close_values = [100.5 + i * 0.1 for i in range(503)]
        # Set 3 rows to have close = 0
        close_values[100] = 0
        close_values[200] = 0
        close_values[300] = 0
        
        data = {
            'Date': dates,
            'Open': [100.0 + i * 0.1 for i in range(503)],
            'High': [101.0 + i * 0.1 for i in range(503)],
            'Low': [99.0 + i * 0.1 for i in range(503)],
            'Close': close_values,
            'Volume': [1000000 + i * 1000 for i in range(503)]
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            result = load_single_stock(temp_path)
            
            assert result is not None
            assert len(result) == 500  # Should have 500 rows after dropping 3
            assert (result['close'] > 0).all()
            assert (result['open'] > 0).all()
            
        finally:
            os.unlink(temp_path)
    
    def test_load_all_stocks_mixed_validity(self):
        """Test 6: Load from temp STOCKS/ folder with 3 valid CSVs and 1 invalid"""
        with tempfile.TemporaryDirectory() as temp_dir:
            stocks_dir = os.path.join(temp_dir, 'STOCKS')
            os.makedirs(stocks_dir)
            
            # Create 3 valid CSVs
            for ticker in ['RELIANCE', 'TCS', 'INFY']:
                dates = pd.date_range('2020-01-01', periods=600, freq='D')
                data = {
                    'Date': dates,
                    'Open': [100.0 + i * 0.1 for i in range(600)],
                    'High': [101.0 + i * 0.1 for i in range(600)],
                    'Low': [99.0 + i * 0.1 for i in range(600)],
                    'Close': [100.5 + i * 0.1 for i in range(600)],
                    'Volume': [1000000 + i * 1000 for i in range(600)]
                }
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(stocks_dir, f'{ticker}.csv'), index=False)
            
            # Create 1 invalid CSV (missing volume column)
            dates = pd.date_range('2020-01-01', periods=600, freq='D')
            data = {
                'Date': dates,
                'Open': [100.0 + i * 0.1 for i in range(600)],
                'High': [101.0 + i * 0.1 for i in range(600)],
                'Low': [99.0 + i * 0.1 for i in range(600)],
                'Close': [100.5 + i * 0.1 for i in range(600)]
                # Missing Volume
            }
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(stocks_dir, 'INVALID.csv'), index=False)
            
            # Test load_all_stocks
            result = load_all_stocks(stocks_dir)
            
            assert len(result) == 3
            assert 'RELIANCE' in result
            assert 'TCS' in result
            assert 'INFY' in result
            assert 'INVALID' not in result
            
            # Verify each loaded stock has correct structure
            for ticker, df in result.items():
                assert len(df) == 600
                assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
                assert isinstance(df.index, pd.DatetimeIndex)
