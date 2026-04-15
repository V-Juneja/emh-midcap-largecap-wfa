# Walk-Forward Analysis (WFA) Engine

A comprehensive Python framework for performing walk-forward analysis on trading strategies, specifically designed for NSE equities with robust backtesting, performance metrics, and detailed reporting.

## 🚀 Features

- **Walk-Forward Analysis**: Rolling window approach with In-Sample (IS) and Out-of-Sample (OOS) testing
- **Multi-Stock Backtesting**: Simultaneous analysis across multiple NSE stocks
- **Technical Indicators**: Built-in indicators (ATR, Moving Averages, Bollinger Bands, Volume Ratio)
- **Risk Management**: Position sizing, stop-loss, and slippage modeling
- **Comprehensive Reporting**: 4-sheet Excel output with charts and performance metrics
- **Strategy Validation**: Bias detection and statistical validation
- **High-Quality Charts**: Separate PNG files with Excel hyperlinks for easy access

## 📊 Output Format

The engine generates a timestamped results folder containing:

### 📈 Excel Report (`summary.xlsx`)
- **Sheet 1: Equity & Drawdown** - Stitched OOS equity curve and drawdown charts
- **Sheet 2: IS vs OOS** - Performance comparison charts across windows
- **Sheet 3: WFA Verdict** - Detailed window-by-window results with color-coded verdicts
- **Sheet 4: Performance Summary** - Comprehensive metrics and statistics

### 🖼️ Chart Files (PNG)
- `equity_curve.png` - Cumulative OOS equity curve
- `drawdown.png` - Drawdown visualization
- `is_returns.png` - In-sample returns per window
- `oos_returns.png` - Out-of-sample returns per window
- `win_rate_comparison.png` - IS vs OOS win rate comparison

### 📄 Verdict File (`verdict.txt`)
- Overall strategy verdict (PASS/FAIL)
- Key metrics: Average WFE, profitable windows percentage
- Total windows analyzed and timestamp

## 🏗️ Project Structure

```
WFA/
├── main.py                 # Main pipeline entry point
├── README.md              # This file
├── WFA_ENGINE/            # Core engine modules
│   ├── backtest.py        # Single-stock backtesting engine
│   ├── bias_validator.py  # Strategy bias detection
│   ├── data_loader.py     # Stock data loading utilities
│   ├── indicators.py      # Technical indicator calculations
│   ├── results_writer.py  # Excel report generation
│   ├── signal_generator.py # Trading signal generation
│   ├── wfa_runner.py      # Walk-forward analysis orchestrator
│   └── prop_test.py       # Proprietary validation tests
├── STRATEGY/              # Trading strategies
│   └── strategy.py        # Example MA crossover strategy
├── STOCKS/                # Stock price data (CSV format)
├── RESULTS/               # Generated analysis results
└── tests/                 # Comprehensive test suite
    ├── test_backtest.py
    ├── test_wfa_runner.py
    ├── test_results_writer.py
    ├── test_integration.py
    └── ...
```

## 🚦 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib openpyxl pytest
```

### 1. Prepare Stock Data

Place your NSE stock data in the `STOCKS/` directory as CSV files with the following format:
- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- Date format: `YYYY-MM-DD`
- File naming: `SYMBOL.csv` (e.g., `RELIANCE.csv`)

### 2. Configure Strategy

Edit `STRATEGY/strategy.py` to implement your trading logic:

```python
def get_signal(row, prev_row) -> int:
    """
    Returns: 1 (BUY), -1 (SELL), or 0 (HOLD)
    """
    # Your strategy logic here
    if condition:
        return 1
    elif condition:
        return -1
    return 0
```

### 3. Run Analysis

```bash
python main.py
```

### 4. View Results

Check the `RESULTS/run_YYYYMMDD_HHMMSS/` folder for your analysis report.

## ⚙️ Configuration

Edit the `CONFIG` dictionary in `main.py`:

```python
CONFIG = {
    "stocks_dir": "STOCKS",                    # Stock data folder
    "strategy_path": "STRATEGY/strategy.py",   # Strategy file
    "results_dir": "RESULTS",                  # Output folder
    "start_date": "2015-01-01",                # Analysis start date
    "end_date": "2024-12-31",                  # Analysis end date
    "is_window_months": 24,                    # Training window size
    "oos_window_months": 6,                    # Testing window size
    "initial_capital": 100000.0,               # Starting capital
    "risk_per_trade": 0.02,                    # Risk per trade (2%)
    "atr_stop_multiplier": 2.0,                # Stop-loss multiplier
    "slippage_pct": 0.0015                     # Slippage (0.15%)
}
```

## 📈 Built-in Indicators

The engine automatically calculates these technical indicators:

- **ATR14** - 14-day Average True Range
- **MA20/MA50** - 20/50-day Simple Moving Averages
- **BB_Upper/BB_Lower** - Bollinger Bands (20-day, 2 std dev)
- **Vol_Ratio** - Current volume / 20-day average volume

## 🎯 Key Metrics

### Walk-Forward Efficiency (WFE)
- Measures strategy consistency between IS and OOS periods
- Higher WFE indicates more robust strategy performance

### Performance Metrics
- **CAGR** - Compound Annual Growth Rate
- **Max Drawdown** - Maximum peak-to-trough decline
- **Sharpe Ratio** - Risk-adjusted returns
- **Sortino Ratio** - Downside risk-adjusted returns
- **Calmar Ratio** - CAGR / Max Drawdown
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Gross profit / Gross loss

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_backtest.py -v
pytest tests/test_wfa_runner.py -v
pytest tests/test_results_writer.py -v
```

### Test Coverage

- **Backtest Engine** - Trade execution, position sizing, risk management
- **WFA Runner** - Window generation, WFE calculations, verdict logic
- **Results Writer** - Excel generation, chart creation, formatting
- **Integration** - End-to-end pipeline validation

## 📋 Strategy Development Guidelines

### 1. Signal Generation
- Use `get_signal(row, prev_row)` function
- Return `1` for BUY, `-1` for SELL, `0` for HOLD
- Access indicators: `row['ma20']`, `row['atr14']`, `row['vol_ratio']`, etc.

### 2. Best Practices
- Avoid look-ahead bias
- Use volume confirmation for breakouts
- Implement proper risk management
- Test across multiple market conditions

### 3. Validation
- Check for survivorship bias
- Validate statistical significance
- Ensure sufficient sample size (>100 trades per window)

## 🔧 Advanced Features

### Bias Detection
The `bias_validator.py` module checks for:
- Look-ahead bias
- Survivorship bias
- Data-snooping bias
- Sample size adequacy

### Custom Indicators
Add custom indicators in `indicators.py`:

```python
def custom_indicator(df):
    # Your calculation logic
    df['custom_indicator'] = calculation
    return df
```

## 📊 Example Output

```
RESULTS/run_20240408_204549/
├── summary.xlsx          # Main report (4 sheets)
├── verdict.txt           # Strategy verdict
├── equity_curve.png      # Equity chart
├── drawdown.png          # Drawdown chart
├── is_returns.png        # IS returns chart
├── oos_returns.png       # OOS returns chart
└── win_rate_comparison.png # Win rate chart
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always validate strategies with out-of-sample testing before real-world application.

## 🆘 Support

For issues and questions:
1. Check the test files for usage examples
2. Review the configuration options
3. Ensure data format compliance
4. Validate strategy logic

---

**Built with ❤️ for quantitative trading research**
