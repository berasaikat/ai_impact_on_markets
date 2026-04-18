import logging
import pandas as pd
import yfinance as yf
from fredapi import Fred
from typing import List

from utils.cache import cached
from config import FRED_API_KEY, CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    pass

@cached(ttl=CACHE_TTL_SECONDS)
def fetch_vix(start: str, end: str) -> pd.DataFrame:
    """Fetch daily VIX close from FRED series VIXCLS. Fallback to yfinance if FRED API key is empty."""
    df = pd.DataFrame()
    if FRED_API_KEY:
        try:
            fred = Fred(api_key=FRED_API_KEY)
            series = fred.get_series('VIXCLS', start, end)
            df = pd.DataFrame(series, columns=['VIX'])
        except Exception as e:
            logger.warning(f"Error fetching VIX from FRED: {e}")
            
    if df.empty:
        logger.info("Falling back to yfinance for ^VIX")
        try:
            # yf.download may return multi-level columns in some yfinance versions
            vix_hf = yf.download('^VIX', start=start, end=end, progress=False)
            if isinstance(vix_hf.columns, pd.MultiIndex):
                vix_hf.columns = vix_hf.columns.get_level_values(0)
            
            if not vix_hf.empty and 'Close' in vix_hf.columns:
                df = pd.DataFrame(vix_hf['Close']).rename(columns={'Close': 'VIX'})
        except Exception as e:
            logger.warning(f"Error fetching VIX from yfinance: {e}")
            df = pd.DataFrame(columns=['VIX'])
            
    if not df.empty:
        # Ensure it has DatetimeIndex and forward-fill up to 5 days for weekends
        df.index = pd.to_datetime(df.index)
        df = df.ffill(limit=5)
        
    return df

@cached(ttl=86400)
def fetch_fed_funds(start: str, end: str) -> pd.DataFrame:
    """Fetch monthly Federal Funds Rate from FRED series FEDFUNDS and resample to daily."""
    if not FRED_API_KEY:
        raise ConfigurationError('FRED_API_KEY required for macro data')
        
    try:
        fred = Fred(api_key=FRED_API_KEY)
        series = fred.get_series('FEDFUNDS', start, end)
        df = pd.DataFrame(series, columns=['FedFunds'])
        df.index = pd.to_datetime(df.index)
        
        # Resample to daily and forward-fill
        if not df.empty:
            df = df.resample('D').ffill()
            
        return df
    except Exception as e:
        logger.warning(f"Error fetching FEDFUNDS from FRED: {e}")
        return pd.DataFrame(columns=['FedFunds'])

@cached(ttl=86400)
def fetch_cpi(start: str, end: str) -> pd.DataFrame:
    """Fetch monthly CPI (CPIAUCSL) from FRED, compute YoY % change, and resample to daily."""
    if not FRED_API_KEY:
        raise ConfigurationError('FRED_API_KEY required for macro data')
        
    try:
        fred = Fred(api_key=FRED_API_KEY)
        
        # Fetch an extra 15 months back to safely compute 12-month YoY change
        early_start = pd.to_datetime(start) - pd.DateOffset(months=15)
        
        series = fred.get_series('CPIAUCSL', early_start, end)
        df = pd.DataFrame(series, columns=['CPI'])
        df.index = pd.to_datetime(df.index)
        
        # Compute YoY before truncating
        df['CPI_YoY'] = df['CPI'].pct_change(periods=12)
        
        # Truncate to the requested date range
        df = df.loc[start:end]
        
        # Resample to daily and forward-fill
        if not df.empty:
            df = df.resample('D').ffill()
            
        return df
    except Exception as e:
        logger.warning(f"Error fetching CPI from FRED: {e}")
        return pd.DataFrame(columns=['CPI', 'CPI_YoY'])

def merge_macro_context(price_df: pd.DataFrame, *macro_dfs: pd.DataFrame) -> pd.DataFrame:
    """Left-join one or more macro DataFrames onto a price DataFrame by date index."""
    if not isinstance(price_df.index, pd.DatetimeIndex):
        raise TypeError("price_df must have a DatetimeIndex.")
        
    if not macro_dfs:
        return price_df
        
    # Join the macro dataframes
    merged_df = price_df.join(list(macro_dfs), how='left')
    
    # Identify macro columns to forward fill
    macro_cols = []
    for mdf in macro_dfs:
        macro_cols.extend(mdf.columns.tolist())
        
    # Forward-fill macro columns up to 5 days
    if macro_cols:
        merged_df[macro_cols] = merged_df[macro_cols].ffill(limit=5)
        
    return merged_df
