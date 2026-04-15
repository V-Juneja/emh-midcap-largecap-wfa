import pytest
import os
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import openpyxl

class TestIntegration:
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment without cleaning up existing folders."""
        # Only create test directories if they don't exist
        if not os.path.exists("STOCKS"):
            os.makedirs("STOCKS", exist_ok=True)
        if not os.path.exists("STRATEGY"):
            os.makedirs("STRATEGY", exist_ok=True)
        
        yield
        
        # Don't clean up - preserve user's data folders
    
    def create_synthetic_stock_data(self, num_stocks: int = 10, num_rows: int = 1500):
        """Create synthetic stock CSV files for testing."""
        
        dates = pd.date_range('2015-01-01', periods=num_rows, freq='D')
        
        for i in range(num_stocks):
            ticker = f"STOCK{i:02d}"
            
            # Create synthetic price data
            np.random.seed(42 + i)
            base_price = 100.0 + i * 10
            returns = np.random.normal(0.001, 0.02, num_rows)
            prices = [base_price]
            
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            prices = prices[1:]  # Remove initial price
            
            # Create OHLCV data
            opens = np.array(prices)
            highs = opens * (1 + np.abs(np.random.normal(0, 0.01, num_rows)))
            lows = opens * (1 - np.abs(np.random.normal(0, 0.01, num_rows)))
            closes = opens + np.random.normal(0, 0.5, num_rows)
            volumes = np.random.randint(100000, 1000000, num_rows)
            
            # Create DataFrame
            df = pd.DataFrame({
                'date': dates,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
            
            # Save to CSV
            csv_path = os.path.join("STOCKS", f"{ticker}.csv")
            df.to_csv(csv_path, index=False)
    
    def create_strategy_file(self):
        """Create a simple strategy file for testing."""
        
        strategy_content = '''
import pandas as pd
import numpy as np

def get_signal(row: pd.Series) -> int:
    """
    Simple moving average crossover strategy.
    
    Args:
        row: DataFrame row with OHLCV and indicator data
        
    Returns:
        int: 1 (BUY), -1 (SELL), or 0 (HOLD)
    """
    
    # Basic MA crossover strategy
    if pd.isna(row['ma20']) or pd.isna(row['ma50']):
        return 0
    
    # Buy signal: short MA above long MA
    if row['ma20'] > row['ma50'] and row['volume'] > row['vol_ma20']:
        return 1
    
    # Sell signal: short MA below long MA
    elif row['ma20'] < row['ma50'] and row['volume'] > row['vol_ma20']:
        return -1
    
    # Hold signal
    return 0
'''
        
        strategy_path = os.path.join("STRATEGY", "strategy.py")
        with open(strategy_path, 'w') as f:
            f.write(strategy_content)
    
    def test_full_pipeline_integration(self):
        """Test complete pipeline integration."""
        
        # Check if we have real stock data
        stock_files = [f for f in os.listdir("STOCKS") if f.endswith('.csv')]
        
        if len(stock_files) == 0:
            # Create test data if no stocks exist
            print("Creating synthetic stock data...")
            self.create_synthetic_stock_data(num_stocks=10, num_rows=1500)
        else:
            print(f"Using existing {len(stock_files)} stock files")
        
        # Create strategy file if it doesn't exist
        if not os.path.exists("STRATEGY/strategy.py"):
            print("Creating strategy file...")
            self.create_strategy_file()
        else:
            print("Using existing strategy file")
        
        # Verify setup
        assert os.path.exists("STRATEGY/strategy.py"), "Strategy file should exist"
        assert len(stock_files) > 0, "Should have at least one stock file"
        
        # Run main.py as subprocess
        print("Running main.py...")
        try:
            result = subprocess.run(
                [sys.executable, "main.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
        except subprocess.TimeoutExpired:
            pytest.fail("main.py timed out after 5 minutes")
        
        # Check exit code
        assert result.returncode == 0, f"main.py should exit with code 0, got {result.returncode}"
        print(f"main.py exited with code: {result.returncode}")
        
        # Check output for expected content
        output = result.stdout
        print("main.py output:")
        print(output)
        
        assert "PIPELINE COMPLETED SUCCESSFULLY" in output, "Should show success message"
        assert "Total stocks processed:" in output, "Should show stocks processed count"
        assert "Average WFE:" in output, "Should show average WFE"
        assert "Final verdict:" in output, "Should show final verdict"
        
        # Check RESULTS folder was created
        assert os.path.exists("RESULTS"), "RESULTS directory should be created"
        
        # Check for run subfolder
        results_contents = os.listdir("RESULTS")
        run_folders = [f for f in results_contents if f.startswith("run_")]
        assert len(run_folders) == 1, f"Should have exactly 1 run folder, got {len(run_folders)}"
        
        run_folder = os.path.join("RESULTS", run_folders[0])
        assert os.path.isdir(run_folder), "Run folder should be a directory"
        
        # Check output files exist (no longer checking for PNG files)
        expected_files = [
            "summary.xlsx",
            "verdict.txt"
        ]
        
        for filename in expected_files:
            file_path = os.path.join(run_folder, filename)
            assert os.path.exists(file_path), f"File {filename} should exist"
            assert os.path.getsize(file_path) > 0, f"File {filename} should not be empty"
        
        # Check that summary.xlsx has exactly 4 sheets
        summary_path = os.path.join(run_folder, "summary.xlsx")
        wb = openpyxl.load_workbook(summary_path)
        sheet_names = wb.sheetnames
        
        assert len(sheet_names) == 4, f"summary.xlsx should have 4 sheets, got {len(sheet_names)}"
        expected_sheets = ["Equity & Drawdown", "IS vs OOS", "WFA Verdict", "Performance Summary"]
        for sheet_name in expected_sheets:
            assert sheet_name in sheet_names, f"Sheet '{sheet_name}' should exist in summary.xlsx"
        
        # Check verdict.txt content
        verdict_path = os.path.join(run_folder, "verdict.txt")
        with open(verdict_path, 'r') as f:
            verdict_content = f.read().strip()
        
        first_line = verdict_content.split('\n')[0]
        assert first_line in ["PASS", "FAIL"], f"First line should be PASS or FAIL, got '{first_line}'"
        
        print(f"Integration test passed! Results in: {run_folder}")
        print(f"Verdict: {first_line}")
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid data."""
        
        # Create invalid stock data (missing required columns)
        dates = pd.date_range('2015-01-01', periods=100, freq='D')
        invalid_df = pd.DataFrame({
            'invalid_column': np.random.normal(0, 1, 100)
        }, index=dates)
        
        invalid_df.to_csv("STOCKS/invalid.csv")
        
        # Create strategy file
        self.create_strategy_file()
        
        # Run main.py - should fail gracefully
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Should exit with error code
        assert result.returncode != 0, "Should exit with non-zero code on error"
        
        # Should show error message
        assert "ERROR: Pipeline failed" in result.stdout, "Should show error message"
        
        print("Error handling test passed - pipeline failed gracefully as expected")
    
    def test_pipeline_no_stocks_error(self):
        """Test pipeline behavior when no stocks are found."""
        
        # Don't create any stock files - STOCKS folder exists but is empty
        
        # Create strategy file
        self.create_strategy_file()
        
        # Run main.py - should fail gracefully
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Should exit with error code
        assert result.returncode != 0, "Should exit with non-zero code when no stocks found"
        
        # Should show error message
        assert "ERROR: Pipeline failed" in result.stdout, "Should show error message"
        assert "No stocks found" in result.stdout, "Should mention no stocks found"
        
        print("No stocks error test passed - pipeline failed gracefully as expected")
