import pandas as pd
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    risk_per_trade: float = 0.02,
    atr_stop_multiplier: float = 2.0,
    slippage_pct: float = 0.0015
) -> dict:
    """Simulate trade execution on a single stock over a specified date range."""
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df[mask].copy()
    signals_filtered = signals[mask]
    
    if len(df_filtered) < 10:
        return {"status": "insufficient_data"}
    
    trades = []
    current_position = None
    equity = initial_capital
    equity_curve = [equity]
    
    for i in range(len(df_filtered)):
        current_date = df_filtered.index[i]
        current_row = df_filtered.iloc[i]
        current_signal = signals_filtered.iloc[i]
        
        if current_position is not None:
            trade = current_position
            exit_triggered = False
            exit_price = None
            exit_reason = None
            
            if current_row['low'] <= trade['stop_price']:
                exit_price = trade['stop_price']
                exit_reason = "stop"
                exit_triggered = True
            
            elif current_signal == -1 and current_date != trade['entry_date']:
                exit_price = current_row['close'] * (1 - slippage_pct)
                exit_reason = "signal"
                exit_triggered = True
            
            if exit_triggered:
                commission_rate = 0.0015
                exit_commission = exit_price * trade['shares'] * commission_rate
                
                entry_cost = trade['entry_price'] * trade['shares'] + trade['entry_commission']
                exit_proceeds = exit_price * trade['shares'] - exit_commission
                trade_return = exit_proceeds - entry_cost
                trade_return_pct = (trade_return / entry_cost) * 100
                
                equity += trade_return
                equity_curve.append(equity)
                
                trade_record = {
                    'entry_date': trade['entry_date'],
                    'exit_date': current_date,
                    'entry_price': trade['entry_price'],
                    'exit_price': exit_price,
                    'stop_price': trade['stop_price'],
                    'shares': trade['shares'],
                    'return_pct': trade_return_pct,
                    'hold_days': (current_date - trade['entry_date']).days,
                    'exit_reason': exit_reason
                }
                trades.append(trade_record)
                
                current_position = None
        
        if current_position is None and current_signal == 1:
            # Only enter trade if we have data for next day
            if i + 1 < len(df_filtered):
                next_row = df_filtered.iloc[i + 1]
                entry_price = next_row['open'] * (1 + slippage_pct)
                
                # Use current_row's ATR for stop calculation (no lookahead bias)
                atr_value = current_row['atr14']
                if pd.isna(atr_value) or atr_value <= 0:
                    logger.warning(f"ATR is NaN or zero at {df_filtered.index[i]}, skipping trade")
                    continue
                stop_price = entry_price - (atr_value * atr_stop_multiplier)
                
                # Position sizing based on current capital and risk
                risk_amount = equity * risk_per_trade  # Use current equity, not initial
                price_risk = entry_price - stop_price
                if price_risk <= 0:
                    price_risk = entry_price * 0.02
                shares = max(1, int(risk_amount / price_risk))
                
                commission_rate = 0.0015
                entry_commission = entry_price * shares * commission_rate
                
                current_position = {
                    'entry_date': df_filtered.index[i + 1],
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'shares': shares,
                    'entry_commission': entry_commission
                }
                
                continue
    
    if not trades:
        return {
            "status": "no_trades",
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "total_return_pct": 0.0,
            "avg_hold_days": 0.0,
            "trades": []
        }
    
    returns = [trade['return_pct'] for trade in trades]
    winning_trades = [r for r in returns if r > 0]
    losing_trades = [r for r in returns if r < 0]
    
    total_trades = len(trades)
    winning_count = len(winning_trades)
    losing_count = len(losing_trades)
    
    win_rate = winning_count / total_trades if total_trades > 0 else 0.0
    avg_return_pct = np.mean(returns) if returns else 0.0
    avg_win_pct = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss_pct = np.mean(losing_trades) if losing_trades else 0.0
    
    gross_profit = sum(winning_trades) if winning_trades else 0.0
    gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    peak = equity_curve[0]
    max_drawdown = 0.0
    for equity_val in equity_curve:
        if equity_val > peak:
            peak = equity_val
        drawdown = (peak - equity_val) / peak
        max_drawdown = max(max_drawdown, drawdown)
    max_drawdown_pct = max_drawdown * 100
    
    if len(returns) > 1:
        avg_hold_days = np.mean([trade['hold_days'] for trade in trades])
        returns_std = np.std(returns)
        if returns_std > 0 and avg_hold_days > 0:
            sharpe = (avg_return_pct / returns_std) * math.sqrt(252 / avg_hold_days)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0
        avg_hold_days = 0.0
    
    total_return_pct = ((equity - initial_capital) / initial_capital) * 100
    
    return {
        "status": "ok",
        "total_trades": total_trades,
        "winning_trades": winning_count,
        "losing_trades": losing_count,
        "win_rate": win_rate,
        "avg_return_pct": avg_return_pct,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "total_return_pct": total_return_pct,
        "avg_hold_days": avg_hold_days,
        "trades": trades
    }
