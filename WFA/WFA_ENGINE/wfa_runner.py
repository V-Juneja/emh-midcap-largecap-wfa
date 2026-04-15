import pandas as pd
import numpy as np
from datetime import datetime
import logging
from backtest import run_backtest

logger = logging.getLogger(__name__)


def run_wfa(
    stocks: dict[str, pd.DataFrame],
    signals_dict: dict[str, pd.Series],
    start_date: str,
    end_date: str,
    is_window_months: int = 24,
    oos_window_months: int = 6,
    initial_capital: float = 100000.0,
    risk_per_trade: float = 0.02,
    atr_stop_multiplier: float = 2.0,
    slippage_pct: float = 0.0015
) -> dict:
    """Run complete rolling walk-forward analysis across all stocks."""
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    windows = []
    current_start = start_dt
    
    while True:
        is_start = current_start
        is_end = is_start + pd.DateOffset(months=is_window_months) - pd.DateOffset(days=1)
        oos_start = is_end + pd.DateOffset(days=1)
        oos_end = oos_start + pd.DateOffset(months=oos_window_months) - pd.DateOffset(days=1)
        
        if oos_end > end_dt:
            break
        
        windows.append({
            'window_id': len(windows),
            'is_start': is_start.strftime('%Y-%m-%d'),
            'is_end': is_end.strftime('%Y-%m-%d'),
            'oos_start': oos_start.strftime('%Y-%m-%d'),
            'oos_end': oos_end.strftime('%Y-%m-%d')
        })
        
        current_start = current_start + pd.DateOffset(months=oos_window_months)
    
    if len(windows) < 3:
        raise ValueError(f"Insufficient data range: only {len(windows)} windows can be generated, minimum 3 required")
    
    logger.info(f"Generated {len(windows)} windows for WFA")
    
    window_results = []
    
    for window in windows:
        logger.info(f"Processing window {window['window_id'] + 1}/{len(windows)}")
        
        is_metrics = _aggregate_backtests(
            stocks, signals_dict,
            window['is_start'], window['is_end'],
            initial_capital, risk_per_trade, atr_stop_multiplier, slippage_pct
        )
        
        oos_metrics = _aggregate_backtests(
            stocks, signals_dict,
            window['oos_start'], window['oos_end'],
            initial_capital, risk_per_trade, atr_stop_multiplier, slippage_pct
        )
        
        wfe = _calculate_wfe(is_metrics, oos_metrics, is_window_months, oos_window_months)
        
        window_result = {
            'window_id': window['window_id'],
            'is_start': window['is_start'],
            'is_end': window['is_end'],
            'oos_start': window['oos_start'],
            'oos_end': window['oos_end'],
            'is_metrics': is_metrics,
            'oos_metrics': oos_metrics,
            'wfe': wfe,
            'stocks_with_trades': is_metrics.get('stocks_with_trades', 0),
            'stocks_skipped': is_metrics.get('stocks_skipped', 0)
        }
        
        window_results.append(window_result)
    
    raw_returns = [w['oos_metrics'].get('total_return_pct', 0.0) for w in window_results]
    compounded = []
    cumulative = 1.0
    for r in raw_returns:
        cumulative *= (1 + r / 100)
        compounded.append((cumulative - 1) * 100)
    oos_equity_curve = compounded
    
    summary = _calculate_summary(window_results, oos_equity_curve)
    
    return {
        'windows': window_results,
        'summary': summary
    }

def _aggregate_backtests(
    stocks: dict[str, pd.DataFrame],
    signals_dict: dict[str, pd.Series],
    start_date: str,
    end_date: str,
    initial_capital: float,
    risk_per_trade: float,
    atr_stop_multiplier: float,
    slippage_pct: float
) -> dict:
    total_trades = 0
    total_returns = []
    winning_trades = 0
    losing_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    max_drawdowns = []
    stocks_with_trades = 0
    stocks_skipped = 0
    
    for ticker, df in stocks.items():
        if ticker not in signals_dict:
            stocks_skipped += 1
            continue
            
        signals = signals_dict[ticker]
        
        # Run backtest for this stock
        result = run_backtest(
            df, signals, start_date, end_date,
            initial_capital, risk_per_trade, atr_stop_multiplier, slippage_pct
        )
        
        # Skip stocks with no trades or insufficient data
        if result['status'] in ['no_trades', 'insufficient_data']:
            stocks_skipped += 1
            continue
            
        # Aggregate metrics
        trades = result['total_trades']
        if trades > 0:
            stocks_with_trades += 1
            total_trades += trades
            total_returns.append(result['total_return_pct'])
            winning_trades += result['winning_trades']
            losing_trades += result['losing_trades']
            
            # Calculate gross profit/loss for this stock
            for trade in result['trades']:
                if trade['return_pct'] > 0:
                    gross_profit += trade['return_pct']
                else:
                    gross_loss += abs(trade['return_pct'])
            
            max_drawdowns.append(result['max_drawdown_pct'])
        else:
            stocks_skipped += 1
    
    # Calculate aggregated metrics
    if total_trades == 0:
        return {
            'status': 'no_trades',
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_return_pct': 0.0,
            'profit_factor': 0.0,
            'max_drawdown_pct': 0.0,
            'total_return_pct': 0.0,
            'stocks_with_trades': stocks_with_trades,
            'stocks_skipped': stocks_skipped
        }
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    avg_return_pct = np.mean(total_returns) if total_returns else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    max_drawdown_pct = np.max(max_drawdowns) if max_drawdowns else 0.0
    total_return_pct = np.mean(total_returns) if total_returns else 0.0
    
    return {
        'status': 'ok',
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_return_pct': avg_return_pct,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown_pct,
        'total_return_pct': total_return_pct,
        'stocks_with_trades': stocks_with_trades,
        'stocks_skipped': stocks_skipped
    }

def _calculate_wfe(
    is_metrics: dict,
    oos_metrics: dict,
    is_window_months: int,
    oos_window_months: int
) -> float:
    if is_metrics['status'] in ['no_trades', 'insufficient_data']:
        return 0.0
        
    if oos_metrics['status'] in ['no_trades', 'insufficient_data']:
        return 0.0
        
    is_return = is_metrics['total_return_pct'] / 100  # Convert to decimal
    oos_return = oos_metrics['total_return_pct'] / 100  # Convert to decimal
        
    if is_return <= 0:
        return 0.0
        
    # Proper annualization using compounding
    years_is = is_window_months / 12
    years_oos = oos_window_months / 12
    
    is_annual = (1 + is_return) ** (1 / years_is) - 1 if years_is > 0 else 0
    oos_annual = (1 + oos_return) ** (1 / years_oos) - 1 if years_oos > 0 else 0
    
    wfe = oos_annual / is_annual if is_annual > 0 else 0.0
        
    return wfe

def _calculate_summary(window_results: list[dict], oos_equity_curve: list[float]) -> dict:
    total_windows = len(window_results)
    profitable_oos_windows = sum(1 for w in window_results if w['oos_metrics'].get('total_return_pct', 0) > 0)
    pct_profitable_windows = profitable_oos_windows / total_windows if total_windows > 0 else 0.0
    
    wfes = [w['wfe'] for w in window_results]
    avg_wfe = np.mean(wfes) if wfes else 0.0
    
    oos_win_rates = [w['oos_metrics'].get('win_rate', 0) for w in window_results if w['oos_metrics'].get('status') == 'ok']
    avg_oos_win_rate = np.mean(oos_win_rates) if oos_win_rates else 0.0
    
    oos_profit_factors = [w['oos_metrics'].get('profit_factor', 0) for w in window_results if w['oos_metrics'].get('status') == 'ok']
    avg_oos_profit_factor = np.mean(oos_profit_factors) if oos_profit_factors else 0.0
    
    verdict = "PASS" if avg_wfe >= 0.5 and pct_profitable_windows >= 0.70 else "FAIL"
    
    return {
        'total_windows': total_windows,
        'profitable_oos_windows': profitable_oos_windows,
        'pct_profitable_windows': pct_profitable_windows,
        'avg_wfe': avg_wfe,
        'avg_oos_win_rate': avg_oos_win_rate,
        'avg_oos_profit_factor': avg_oos_profit_factor,
        'oos_equity_curve': oos_equity_curve,
        'verdict': verdict
    }
