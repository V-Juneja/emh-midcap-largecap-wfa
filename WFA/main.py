#!/usr/bin/env python3
"""
Walk-Forward Analysis Pipeline Entry Point

This script orchestrates the complete WFA workflow:
1. Load stock data
2. Compute technical indicators
3. Load strategy and generate signals
4. Run walk-forward analysis
5. Write results to timestamped folder

Usage: python main.py [--us TICKER1,TICKER2,...]
"""

import os
import sys
import time
import traceback
import argparse
from pathlib import Path

# Add WFA_ENGINE to path
wfa_engine_path = Path(__file__).parent / "WFA_ENGINE"
sys.path.insert(0, str(wfa_engine_path.absolute()))

from data_loader import load_all_stocks, load_us_stocks_yfinance
from indicators import compute_indicators
from signal_generator import load_strategy, generate_signals
from wfa_runner import run_wfa
from results_writer import write_results
from bias_validator import run_all_checks
from backtest import run_backtest

# Edit these values to configure your WFA run
CONFIG = {
    "stocks_dir": "STOCKS",                    # folder containing stock CSVs
    "strategy_path": "STRATEGY/strategy.py",   # path to your strategy file
    "results_dir": "RESULTS",                  # folder for output results
    "start_date": "2015-01-01",                # start of full data range
    "end_date": "2024-12-31",                  # end of full data range
    "is_window_months": 24,                    # how many months to train on
    "oos_window_months": 6,                    # how many months to test on
    "initial_capital": 100000.0,               # starting capital in rupees
    "risk_per_trade": 0.02,                    # risk 2% of capital per trade
    "atr_stop_multiplier": 2.0,                # stop = entry - (2 x ATR)
    "slippage_pct": 0.0015                     # 0.15% slippage per side
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Walk-Forward Analysis Pipeline')
    parser.add_argument('--us', type=str, help='US stock tickers (comma-separated) for yfinance data')
    return parser.parse_args()

def main():
    print("=" * 60)
    print("WALK-FORWARD ANALYSIS PIPELINE")
    print("=" * 60)
    
    args = parse_arguments()
    start_time = time.time()
    
    try:
        # Step 1: Load stock data
        if args.us:
            print("Step 1: Loading US stock data from yfinance...")
            tickers = [ticker.strip().upper() for ticker in args.us.split(',')]
            stocks = load_us_stocks_yfinance(tickers, CONFIG["start_date"], CONFIG["end_date"])
            total_stocks = len(stocks)
            print(f"  Loaded {total_stocks} US stocks: {', '.join(tickers)}")
        else:
            print("Step 1: Loading stock data...")
            stocks = load_all_stocks(CONFIG["stocks_dir"])
            total_stocks = len(stocks)
            print(f"  Loaded {total_stocks} stocks from {CONFIG['stocks_dir']}")
        
        if total_stocks == 0:
            raise ValueError("No stocks loaded")
        
        print("Step 2: Computing technical indicators...")
        for ticker, df in stocks.items():
            stocks[ticker] = compute_indicators(df)
        print(f"  Computed indicators for {total_stocks} stocks")
        
        print("Step 3: Loading trading strategy...")
        strategy = load_strategy(CONFIG["strategy_path"])
        print(f"  Loaded strategy from {CONFIG['strategy_path']}")
        
        print("Step 4: Generating trading signals...")
        signals_dict = {}
        for ticker, df in stocks.items():
            signals = generate_signals(df, strategy)
            signals_dict[ticker] = signals
        print(f"  Generated signals for {total_stocks} stocks")
        
        print("Step 5: Running walk-forward analysis...")
        wfa_output = run_wfa(
            stocks=stocks,
            signals_dict=signals_dict,
            start_date=CONFIG["start_date"],
            end_date=CONFIG["end_date"],
            is_window_months=CONFIG["is_window_months"],
            oos_window_months=CONFIG["oos_window_months"],
            initial_capital=CONFIG["initial_capital"],
            risk_per_trade=CONFIG["risk_per_trade"],
            atr_stop_multiplier=CONFIG["atr_stop_multiplier"],
            slippage_pct=CONFIG["slippage_pct"]
        )
        
        total_windows = wfa_output['summary']['total_windows']
        avg_wfe = wfa_output['summary']['avg_wfe']
        verdict = wfa_output['summary']['verdict']
        
        print(f"  Analyzed {total_windows} windows")
        print(f"  Average WFE: {avg_wfe:.3f}")
        print(f"  Verdict: {verdict}")
        
        print("Step 6: Writing results...")
        run_folder = write_results(wfa_output, CONFIG["results_dir"])
        print(f"  Results written to: {run_folder}")
        
        print("Step 7: Running bias validation...")
        sample_ticker = list(stocks.keys())[0]
        sample_df = stocks[sample_ticker]
        
        sample_trades = run_backtest(
            sample_df, 
            signals_dict[sample_ticker],
            CONFIG["start_date"],
            CONFIG["end_date"],
            CONFIG["initial_capital"],
            CONFIG["risk_per_trade"],
            CONFIG["atr_stop_multiplier"],
            CONFIG["slippage_pct"]
        )
        
        bias_results = run_all_checks(
            df_sample=sample_df,
            strategy_module=strategy,
            wfa_output=wfa_output,
            trades_sample=sample_trades.get('trades', []),
            signals_sample=signals_dict[sample_ticker],
            stocks=stocks
        )
        
        if not bias_results["overall_clean"]:
            print("  WARNING: Bias issues detected - Review required!")
            print("  Check the bias validation report above for details")
        else:
            print("  Bias validation passed - No issues detected")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total stocks processed: {total_stocks}")
        print(f"Total windows analyzed: {total_windows}")
        print(f"Average WFE: {avg_wfe:.3f}")
        print(f"Final verdict: {verdict}")
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        print(f"Results saved to: {run_folder}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with exception:")
        print(f"  {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print("=" * 60)
        print("PIPELINE FAILED")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
