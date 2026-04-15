import pandas as pd
import yfinance as yf
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_single_stock(filepath: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(filepath)
        
        required_columns = {'date', 'open', 'high', 'low', 'close', 'volume'}
        df_columns_lower = {col.lower() for col in df.columns}
        
        if not required_columns.issubset(df_columns_lower):
            missing = required_columns - df_columns_lower
            logger.warning(f"File {filepath} missing required columns: {missing}")
            return None
        
        df.columns = df.columns.str.lower()
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['volume'] = df['volume'].fillna(0).astype('int64')
        
        initial_rows = len(df)
        df = df[(df['open'] > 0) & (df['close'] > 0)]
        invalid_ohlc = initial_rows - len(df)
        
        if invalid_ohlc > 0:
            logger.info(f"File {filepath}: Dropped {invalid_ohlc} rows with invalid OHLC values")
        
        df = df[~df.index.duplicated(keep='last')]
        
        if len(df) < 500:
            logger.warning(f"File {filepath}: Only {len(df)} rows after cleaning (minimum 500 required)")
            return None
        
        if df.isnull().any().any():
            logger.warning(f"File {filepath}: Contains NaN values after cleaning")
            return None
        
        logger.info(f"Successfully loaded {filepath}: {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {str(e)}")
        return None

def load_all_stocks(stocks_dir: str) -> dict[str, pd.DataFrame]:
    stocks_path = Path(stocks_dir)
    
    if not stocks_path.exists():
        logger.error(f"Stocks directory {stocks_dir} does not exist")
        return {}
    
    stock_files = list(stocks_path.glob("*.csv"))
    
    if not stock_files:
        logger.warning(f"No CSV files found in {stocks_dir}")
        return {}
    
    loaded_stocks = {}
    skipped_count = 0
    
    for file_path in stock_files:
        ticker = file_path.stem
        df = load_single_stock(str(file_path))
        
        if df is not None:
            loaded_stocks[ticker] = df
        else:
            skipped_count += 1
    
    logger.info(f"Loaded {len(loaded_stocks)} stocks successfully")
    logger.info(f"Skipped {skipped_count} stocks")
    
    return loaded_stocks


def load_us_stocks_yfinance(tickers: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """
    Load US stock data using yfinance.
    
    Args:
        tickers: List of stock tickers (e.g., ['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        Dictionary of ticker -> DataFrame with OHLCV data
    """
    loaded_stocks = {}
    skipped_count = 0
    
    for ticker in tickers:
        try:
            logger.info(f"Downloading {ticker} data from {start_date} to {end_date}")
            
            # Download data using yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                skipped_count += 1
                continue
            
            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Ensure we have required columns
            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                logger.warning(f"{ticker} missing required columns: {missing}")
                skipped_count += 1
                continue
            
            # Clean data
            initial_rows = len(df)
            df = df[(df['open'] > 0) & (df['close'] > 0)]
            invalid_ohlc = initial_rows - len(df)
            
            if invalid_ohlc > 0:
                logger.info(f"{ticker}: Dropped {invalid_ohlc} rows with invalid OHLC values")
            
            # Remove any remaining NaN values
            df = df.dropna()
            
            if len(df) < 500:
                logger.warning(f"{ticker}: Only {len(df)} rows after cleaning (minimum 500 required)")
                skipped_count += 1
                continue
            
            logger.info(f"Successfully loaded {ticker}: {len(df)} rows")
            loaded_stocks[ticker] = df
            
        except Exception as e:
            logger.error(f"Failed to load {ticker}: {str(e)}")
            skipped_count += 1
    
    logger.info(f"Loaded {len(loaded_stocks)} US stocks successfully")
    logger.info(f"Skipped {skipped_count} stocks")
    
    return loaded_stocks
