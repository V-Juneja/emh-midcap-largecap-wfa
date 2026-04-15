import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image
import logging

# Try to import PIL for better image handling
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available, using basic image embedding")

# Configure matplotlib to prevent unicode issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

logger = logging.getLogger(__name__)


def write_results(wfa_output: dict, results_dir: str = "RESULTS", initial_capital: float = 100000.0) -> str:
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    
    wb = Workbook()
    _write_summary_excel(wb, wfa_output, run_folder, initial_capital)
    _write_verdict_file(wfa_output, run_folder)
    
    return run_folder


def _write_summary_excel(wb: Workbook, wfa_output: dict, run_folder: str, initial_capital: float):
    wb.remove(wb.active)
    
    _write_equity_drawdown_sheet(wb, wfa_output, run_folder)
    _write_is_vs_oos_sheet(wb, wfa_output, run_folder)
    _write_verdict_sheet(wb, wfa_output)
    _write_performance_summary_sheet(wb, wfa_output, initial_capital)
    
    excel_path = os.path.join(run_folder, "summary.xlsx")
    wb.save(excel_path)


def _write_equity_drawdown_sheet(wb: Workbook, wfa_output: dict, run_folder: str):
    ws = wb.create_sheet("Equity & Drawdown")
    
    equity_curve = wfa_output['summary']['oos_equity_curve']
    
    # Calculate drawdown
    drawdown = []
    peak = equity_curve[0] if equity_curve else 0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / (1 + peak/100) * 100 if peak >= 0 else 0
        drawdown.append(dd)
    
    # Create equity chart
    fig_equity, ax_equity = plt.subplots(figsize=(12, 6))
    window_indices = list(range(len(equity_curve)))
    ax_equity.plot(window_indices, equity_curve, 'b-', linewidth=2, label='Equity')
    ax_equity.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax_equity.set_title('Stitched OOS Equity Curve')
    ax_equity.set_xlabel('Window Index')
    ax_equity.set_ylabel('Cumulative Return %')
    ax_equity.grid(True, alpha=0.3)
    ax_equity.legend()
    
    # Create drawdown chart
    fig_drawdown, ax_drawdown = plt.subplots(figsize=(12, 6))
    ax_drawdown.fill_between(window_indices, drawdown, 0, color='red', alpha=0.3)
    ax_drawdown.plot(window_indices, drawdown, 'r-', linewidth=1)
    ax_drawdown.set_title('Drawdown')
    ax_drawdown.set_xlabel('Window Index')
    ax_drawdown.set_ylabel('Drawdown %')
    ax_drawdown.grid(True, alpha=0.3)
    
    # Add charts to sheet
    _add_chart_to_sheet(ws, fig_equity, 'A1', run_folder, 'equity_curve')
    _add_chart_to_sheet(ws, fig_drawdown, 'A35', run_folder, 'drawdown')


def _write_is_vs_oos_sheet(wb: Workbook, wfa_output: dict, run_folder: str):
    ws = wb.create_sheet("IS vs OOS")
    
    windows = wfa_output['windows']
    window_ids = [w['window_id'] for w in windows]
    is_returns = [w['is_metrics'].get('total_return_pct', 0) for w in windows]
    oos_returns = [w['oos_metrics'].get('total_return_pct', 0) for w in windows]
    is_win_rates = [w['is_metrics'].get('win_rate', 0) for w in windows]
    oos_win_rates = [w['oos_metrics'].get('win_rate', 0) for w in windows]
    
    # IS returns chart
    fig_is, ax_is = plt.subplots(figsize=(12, 4))
    ax_is.bar(window_ids, is_returns, color='blue', alpha=0.7)
    ax_is.set_title('IS Cumulative Return per Window')
    ax_is.set_xlabel('Window ID')
    ax_is.set_ylabel('Return %')
    ax_is.grid(True, alpha=0.3)
    
    # OOS returns chart
    fig_oos, ax_oos = plt.subplots(figsize=(12, 4))
    ax_oos.bar(window_ids, oos_returns, color='orange', alpha=0.7)
    ax_oos.set_title('OOS Cumulative Return per Window')
    ax_oos.set_xlabel('Window ID')
    ax_oos.set_ylabel('Return %')
    ax_oos.grid(True, alpha=0.3)
    
    # Win rate comparison chart
    fig_wr, ax_wr = plt.subplots(figsize=(12, 4))
    x = np.arange(len(window_ids))
    width = 0.35
    ax_wr.bar(x - width/2, is_win_rates, width, label='IS Win Rate', color='blue', alpha=0.7)
    ax_wr.bar(x + width/2, oos_win_rates, width, label='OOS Win Rate', color='orange', alpha=0.7)
    ax_wr.set_title('IS vs OOS Win Rate by Window')
    ax_wr.set_xlabel('Window ID')
    ax_wr.set_ylabel('Win Rate')
    ax_wr.set_xticks(x, window_ids)
    ax_wr.legend()
    ax_wr.grid(True, alpha=0.3)
    
    # Add charts to sheet
    _add_chart_to_sheet(ws, fig_is, 'A1', run_folder, 'is_returns')
    _add_chart_to_sheet(ws, fig_oos, 'A30', run_folder, 'oos_returns')
    _add_chart_to_sheet(ws, fig_wr, 'A60', run_folder, 'win_rate_comparison')


def _write_verdict_sheet(wb: Workbook, wfa_output: dict):
    ws = wb.create_sheet("WFA Verdict")
    
    # Data table
    headers = [
        'Window', 'IS Start', 'IS End', 'OOS Start', 'OOS End',
        'IS Return %', 'OOS Return %', 'WFE', 'OOS Win Rate',
        'OOS Profit Factor', 'OOS Max DD %', 'Stocks Traded', 'Stocks Skipped'
    ]
    ws.append(headers)
    
    # Style header
    header_font = Font(bold=True)
    for cell in ws[1]:
        cell.font = header_font
    
    # Add data rows
    for window in wfa_output['windows']:
        row = [
            window['window_id'],
            window['is_start'],
            window['is_end'],
            window['oos_start'],
            window['oos_end'],
            f"{window['is_metrics'].get('total_return_pct', 0):.2f}",
            f"{window['oos_metrics'].get('total_return_pct', 0):.2f}",
            f"{window['wfe']:.3f}",
            f"{window['oos_metrics'].get('win_rate', 0):.3f}",
            f"{window['oos_metrics'].get('profit_factor', 0):.2f}",
            f"{window['oos_metrics'].get('max_drawdown_pct', 0):.2f}",
            window['stocks_with_trades'],
            window['stocks_skipped']
        ]
        ws.append(row)
        
        # Color code WFE
        wfe_cell = ws.cell(row=ws.max_row, column=8)  # Column H
        wfe_value = window['wfe']
        if wfe_value >= 0.5:
            wfe_cell.fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
        elif wfe_value < 0.35:
            wfe_cell.fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")
        else:
            wfe_cell.fill = PatternFill(start_color="FFFFE0", end_color="FFFFE0", fill_type="solid")
    
    ws.freeze_panes = 'A2'
    
    # Summary block (2 blank rows gap)
    ws.append([])
    ws.append([])
    
    summary = wfa_output['summary']
    summary_data = [
        ['Verdict', summary['verdict']],
        ['Avg WFE', f"{summary['avg_wfe']:.3f}"],
        ['% Profitable Windows', f"{summary['pct_profitable_windows']:.2%}"],
        ['Avg OOS Win Rate', f"{summary['avg_oos_win_rate']:.3f}"],
        ['Avg OOS Profit Factor', f"{summary['avg_oos_profit_factor']:.2f}"],
        ['Total Windows', summary['total_windows']],
        ['Profitable OOS Windows', summary['profitable_oos_windows']]
    ]
    
    for label, value in summary_data:
        ws.append([label, value])
        
        if label == 'Verdict':
            verdict_cell = ws.cell(row=ws.max_row, column=2)
            if value == 'PASS':
                verdict_cell.fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
            else:
                verdict_cell.fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")
            verdict_cell.font = Font(bold=True)


def _write_performance_summary_sheet(wb: Workbook, wfa_output: dict, initial_capital: float):
    ws = wb.create_sheet("Performance Summary")
    
    summary = wfa_output['summary']
    equity_curve = summary['oos_equity_curve']
    
    # Calculate metrics
    final_return = equity_curve[-1] if equity_curve else 0
    final_capital = initial_capital * (1 + final_return/100)
    
    # CAGR calculation
    windows = summary['total_windows']
    oos_months = 6  # Default OOS window months
    total_months = windows * oos_months
    years = total_months / 12
    cagr = ((1 + final_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Max drawdown
    max_dd = 0
    peak = equity_curve[0] if equity_curve else 0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / (1 + peak/100) * 100 if peak >= 0 else 0
        max_dd = max(max_dd, dd)
    
    # Drawdown duration (consecutive windows in drawdown)
    dd_duration = 0
    current_dd_streak = 0
    for i in range(1, len(equity_curve)):
        if equity_curve[i] < equity_curve[i-1]:
            current_dd_streak += 1
        else:
            dd_duration = max(dd_duration, current_dd_streak)
            current_dd_streak = 0
    
    # Win rate
    win_rate = summary.get('avg_oos_win_rate', 0)
    
    # Avg win/loss
    avg_win = summary.get('avg_oos_avg_win_pct', 0)
    avg_loss = summary.get('avg_oos_avg_loss_pct', 0)
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 'N/A'
    
    # Profit factor
    profit_factor = summary.get('avg_oos_profit_factor', 0)
    
    # Sharpe and Sortino
    oos_returns = [w['oos_metrics'].get('total_return_pct', 0) for w in wfa_output['windows']]
    if len(oos_returns) > 1:
        mean_return = np.mean(oos_returns)
        std_return = np.std(oos_returns)
        sharpe = (mean_return / std_return) * np.sqrt(12/oos_months) if std_return > 0 else 0
        
        negative_returns = [r for r in oos_returns if r < 0]
        std_negative = np.std(negative_returns) if negative_returns else 0
        sortino = (mean_return / std_negative) * np.sqrt(12/oos_months) if std_negative > 0 else 0
    else:
        sharpe = 0
        sortino = 0
    
    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Total trades and hold days
    total_trades = sum(w['oos_metrics'].get('total_trades', 0) for w in wfa_output['windows'])
    avg_hold_days = summary.get('avg_oos_avg_hold_days', 'N/A')
    
    metrics = [
        ['Initial Capital', f"{initial_capital:,.2f}"],
        ['Final Capital', f"{final_capital:,.2f}"],
        ['Total OOS Return %', f"{final_return:.2f}"],
        ['CAGR %', f"{cagr:.2f}"],
        ['Max Drawdown %', f"{max_dd:.2f}"],
        ['Max Drawdown Duration', f"{dd_duration}"],
        ['Win Rate', f"{win_rate:.3f}"],
        ['Avg Win %', f"{avg_win:.2f}" if avg_win != 0 else 'N/A'],
        ['Avg Loss %', f"{avg_loss:.2f}" if avg_loss != 0 else 'N/A'],
        ['Avg R:R', f"{rr_ratio:.2f}" if isinstance(rr_ratio, (int, float)) else 'N/A'],
        ['Profit Factor', f"{profit_factor:.2f}"],
        ['Sharpe Ratio', f"{sharpe:.2f}"],
        ['Sortino Ratio', f"{sortino:.2f}"],
        ['Calmar Ratio', f"{calmar:.2f}"],
        ['Total Trades', f"{total_trades}"],
        ['Avg Hold Days', f"{avg_hold_days}" if isinstance(avg_hold_days, (int, float)) else 'N/A'],
        ['Total Windows', f"{windows}"],
        ['Profitable Windows', f"{summary['profitable_oos_windows']}"],
        ['WFA Verdict', f"{summary['verdict']}"]
    ]
    
    # Style headers
    header_font = Font(bold=True)
    
    for metric, value in metrics:
        row = [metric, value]
        ws.append(row)
        
        # Style header
        header_cell = ws.cell(row=ws.max_row, column=1)
        header_cell.font = header_font
        
        # Color verdict
        if metric == 'WFA Verdict':
            value_cell = ws.cell(row=ws.max_row, column=2)
            if value == 'PASS':
                value_cell.fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
            else:
                value_cell.fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")
            value_cell.font = Font(bold=True)


def _add_chart_to_sheet(ws, fig, cell_ref: str, run_folder: str, chart_name: str):
    try:
        # Save chart as separate PNG file
        chart_path = os.path.join(run_folder, f"{chart_name}.png")
        fig.savefig(chart_path, format='png', dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', pad_inches=0.1)
        
        # Close the figure to free memory
        plt.close(fig)
        
        # Add a hyperlink to the chart file
        chart_filename = f"{chart_name}.png"
        ws[cell_ref] = f"Chart: {chart_name}"
        ws[cell_ref].hyperlink = chart_filename
        ws[cell_ref].style = "Hyperlink"
        
    except Exception as e:
        logger.error(f"Error saving chart {chart_name}: {e}")
        plt.close(fig)
        ws[cell_ref] = f"Chart error: {chart_name}"


def _write_verdict_file(wfa_output: dict, run_folder: str):
    verdict_path = os.path.join(run_folder, "verdict.txt")
    
    with open(verdict_path, 'w') as f:
        f.write(f"{wfa_output['summary']['verdict']}\n")
        f.write(f"avg_wfe: {wfa_output['summary']['avg_wfe']}\n")
        f.write(f"pct_profitable_windows: {wfa_output['summary']['pct_profitable_windows']}\n")
        f.write(f"total_windows: {wfa_output['summary']['total_windows']}\n")
        f.write(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
