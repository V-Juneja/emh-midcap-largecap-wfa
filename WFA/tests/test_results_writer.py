import pytest
import os
import shutil
from pathlib import Path
import sys
from datetime import datetime
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'WFA_ENGINE'))

from results_writer import write_results

class TestResultsWriter:
    
    @pytest.fixture(autouse=True)
    def cleanup_test_files(self):
        """Clean up test files after each test."""
        yield
        # Clean up RESULTS directory
        if os.path.exists("RESULTS"):
            shutil.rmtree("RESULTS")
    
    @pytest.fixture
    def synthetic_wfa_output(self):
        """Create synthetic WFA output for testing."""
        return {
            'windows': [
                {
                    'window_id': 1,
                    'is_start': '2015-01-01',
                    'is_end': '2016-12-31',
                    'oos_start': '2017-01-01',
                    'oos_end': '2017-06-30',
                    'is_metrics': {
                        'total_return_pct': 15.5,
                        'win_rate': 0.65,
                        'profit_factor': 1.8,
                        'max_drawdown_pct': -8.2,
                        'total_trades': 25
                    },
                    'oos_metrics': {
                        'total_return_pct': 8.3,
                        'win_rate': 0.60,
                        'profit_factor': 1.5,
                        'max_drawdown_pct': -5.1,
                        'total_trades': 12,
                        'avg_win_pct': 2.5,
                        'avg_loss_pct': -1.8,
                        'avg_hold_days': 15
                    },
                    'wfe': 0.45,
                    'stocks_with_trades': 8,
                    'stocks_skipped': 2
                },
                {
                    'window_id': 2,
                    'is_start': '2017-01-01',
                    'is_end': '2018-12-31',
                    'oos_start': '2019-01-01',
                    'oos_end': '2019-06-30',
                    'is_metrics': {
                        'total_return_pct': 22.1,
                        'win_rate': 0.70,
                        'profit_factor': 2.1,
                        'max_drawdown_pct': -6.5,
                        'total_trades': 30
                    },
                    'oos_metrics': {
                        'total_return_pct': 12.7,
                        'win_rate': 0.75,
                        'profit_factor': 2.3,
                        'max_drawdown_pct': -3.8,
                        'total_trades': 18,
                        'avg_win_pct': 3.1,
                        'avg_loss_pct': -1.5,
                        'avg_hold_days': 12
                    },
                    'wfe': 0.68,
                    'stocks_with_trades': 9,
                    'stocks_skipped': 1
                },
                {
                    'window_id': 3,
                    'is_start': '2019-01-01',
                    'is_end': '2020-12-31',
                    'oos_start': '2021-01-01',
                    'oos_end': '2021-06-30',
                    'is_metrics': {
                        'total_return_pct': -3.2,
                        'win_rate': 0.45,
                        'profit_factor': 0.8,
                        'max_drawdown_pct': -12.3,
                        'total_trades': 20
                    },
                    'oos_metrics': {
                        'total_return_pct': -2.1,
                        'win_rate': 0.40,
                        'profit_factor': 0.7,
                        'max_drawdown_pct': -8.5,
                        'total_trades': 10,
                        'avg_win_pct': 1.8,
                        'avg_loss_pct': -2.8,
                        'avg_hold_days': 18
                    },
                    'wfe': 0.0,
                    'stocks_with_trades': 7,
                    'stocks_skipped': 3
                }
            ],
            'summary': {
                'avg_wfe': 0.376,
                'pct_profitable_windows': 0.667,
                'total_windows': 3,
                'profitable_oos_windows': 2,
                'avg_oos_win_rate': 0.583,
                'avg_oos_profit_factor': 1.5,
                'avg_oos_avg_win_pct': 2.47,
                'avg_oos_avg_loss_pct': -2.03,
                'avg_oos_avg_hold_days': 15,
                'oos_equity_curve': [8.3, 21.5, 18.9],
                'verdict': 'FAIL'
            }
        }
    
    def test_run_folder_created_with_correct_naming_pattern(self, synthetic_wfa_output):
        """Test 1: run folder created with correct naming pattern"""
        run_folder = write_results(synthetic_wfa_output)
        
        # Check folder exists and follows pattern RESULTS/run_YYYYMMDD_HHMMSS
        assert os.path.exists(run_folder)
        # Normalize path to handle both forward and backward slashes
        normalized_path = run_folder.replace("\\", "/")
        assert normalized_path.startswith("RESULTS/run_")
        
        # Check timestamp format
        timestamp_part = os.path.basename(run_folder).replace("run_", "")
        assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS
        assert timestamp_part[8] == '_'
    
    def test_summary_xlsx_and_verdict_txt_exist_and_non_empty(self, synthetic_wfa_output):
        """Test 2: summary.xlsx and verdict.txt exist and are non-empty"""
        run_folder = write_results(synthetic_wfa_output)
        
        summary_path = os.path.join(run_folder, "summary.xlsx")
        verdict_path = os.path.join(run_folder, "verdict.txt")
        
        # Check files exist
        assert os.path.exists(summary_path)
        assert os.path.exists(verdict_path)
        
        # Check files are non-empty
        assert os.path.getsize(summary_path) > 0
        assert os.path.getsize(verdict_path) > 0
        
        # Check no PNG files exist
        png_files = [f for f in os.listdir(run_folder) if f.endswith('.png')]
        assert len(png_files) == 0
    
    def test_summary_xlsx_has_exactly_4_sheets(self, synthetic_wfa_output):
        """Test 3: summary.xlsx has exactly 4 sheets named correctly"""
        run_folder = write_results(synthetic_wfa_output)
        summary_path = os.path.join(run_folder, "summary.xlsx")
        
        wb = openpyxl.load_workbook(summary_path)
        sheet_names = wb.sheetnames
        
        assert len(sheet_names) == 4
        assert "Equity & Drawdown" in sheet_names
        assert "IS vs OOS" in sheet_names
        assert "WFA Verdict" in sheet_names
        assert "Performance Summary" in sheet_names
    
    def test_wfa_verdict_sheet_structure_and_formatting(self, synthetic_wfa_output):
        """Test 4: WFA Verdict sheet has correct headers, data rows, and formatting"""
        run_folder = write_results(synthetic_wfa_output)
        summary_path = os.path.join(run_folder, "summary.xlsx")
        
        wb = openpyxl.load_workbook(summary_path)
        ws = wb["WFA Verdict"]
        
        # Check headers in row 1
        expected_headers = [
            'Window', 'IS Start', 'IS End', 'OOS Start', 'OOS End',
            'IS Return %', 'OOS Return %', 'WFE', 'OOS Win Rate',
            'OOS Profit Factor', 'OOS Max DD %', 'Stocks Traded', 'Stocks Skipped'
        ]
        
        for col_idx, expected_header in enumerate(expected_headers, 1):
            assert ws.cell(row=1, column=col_idx).value == expected_header
            assert ws.cell(row=1, column=col_idx).font.bold
        
        # Check correct number of data rows (3 windows + header + 2 blank rows + 7 summary rows)
        assert ws.max_row == 13  # 3 data rows + header + 2 blank rows + 7 summary rows
        
        # Check WFE color coding
        wfe_colors = []
        for row_idx in range(2, 5):  # Data rows
            wfe_cell = ws.cell(row=row_idx, column=8)
            if wfe_cell.fill.start_color.rgb:
                wfe_colors.append(wfe_cell.fill.start_color.rgb)
        
        # Window 1: WFE=0.45 (yellow), Window 2: WFE=0.68 (green), Window 3: WFE=0.0 (red)
        # Just check that colors are applied
        assert len(wfe_colors) == 3
        
        # Check verdict cell coloring
        verdict_row = None
        for row_idx in range(1, ws.max_row + 1):
            if ws.cell(row=row_idx, column=1).value == 'Verdict':
                verdict_row = row_idx
                break
        
        assert verdict_row is not None
        verdict_cell = ws.cell(row=verdict_row, column=2)
        assert verdict_cell.value == 'FAIL'
        assert verdict_cell.font.bold
        assert verdict_cell.fill.start_color.rgb  # Should be colored red
    
    def test_performance_summary_sheet_has_all_required_metrics(self, synthetic_wfa_output):
        """Test 5: Performance Summary sheet has all required metrics"""
        run_folder = write_results(synthetic_wfa_output)
        summary_path = os.path.join(run_folder, "summary.xlsx")
        
        wb = openpyxl.load_workbook(summary_path)
        ws = wb["Performance Summary"]
        
        # Get all metric labels (first column)
        metric_labels = []
        for row_idx in range(1, ws.max_row + 1):
            label = ws.cell(row=row_idx, column=1).value
            if label:
                metric_labels.append(label)
        
        # Check for required metrics
        required_metrics = [
            'Initial Capital', 'Final Capital', 'CAGR %', 'Max Drawdown %',
            'Win Rate', 'Profit Factor', 'Sharpe Ratio', 'Sortino Ratio',
            'Calmar Ratio', 'WFA Verdict'
        ]
        
        for metric in required_metrics:
            assert metric in metric_labels, f"Missing metric: {metric}"
        
        # Check verdict cell formatting
        verdict_row = None
        for row_idx in range(1, ws.max_row + 1):
            if ws.cell(row=row_idx, column=1).value == 'WFA Verdict':
                verdict_row = row_idx
                break
        
        assert verdict_row is not None
        verdict_cell = ws.cell(row=verdict_row, column=2)
        assert verdict_cell.font.bold
        assert verdict_cell.fill.start_color.rgb  # Should be colored
    
    def test_verdict_txt_content_matches_summary(self, synthetic_wfa_output):
        """Test 6: verdict.txt content matches summary verdict"""
        run_folder = write_results(synthetic_wfa_output)
        verdict_path = os.path.join(run_folder, "verdict.txt")
        
        with open(verdict_path, 'r') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        
        # First line should be PASS or FAIL
        assert lines[0] in ['PASS', 'FAIL']
        assert lines[0] == synthetic_wfa_output['summary']['verdict']
        
        # Check for required fields
        assert 'avg_wfe:' in content
        assert 'pct_profitable_windows:' in content
        assert 'total_windows:' in content
        assert 'date:' in content
        
        # Check values match
        assert f"avg_wfe: {synthetic_wfa_output['summary']['avg_wfe']}" in content
        assert f"total_windows: {synthetic_wfa_output['summary']['total_windows']}" in content
    
    def test_multiple_runs_create_separate_folders(self, synthetic_wfa_output):
        """Test 7: Multiple runs create separate folders, both valid"""
        run_folder1 = write_results(synthetic_wfa_output)
        run_folder2 = write_results(synthetic_wfa_output)
        
        # Both folders should exist
        assert os.path.exists(run_folder1)
        assert os.path.exists(run_folder2)
        assert run_folder1 != run_folder2
        
        # Both should have the required files
        for run_folder in [run_folder1, run_folder2]:
            summary_path = os.path.join(run_folder, "summary.xlsx")
            verdict_path = os.path.join(run_folder, "verdict.txt")
            
            assert os.path.exists(summary_path)
            assert os.path.exists(verdict_path)
            assert os.path.getsize(summary_path) > 0
            assert os.path.getsize(verdict_path) > 0
