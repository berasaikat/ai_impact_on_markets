import logging
import pandas as pd
import yfinance as yf
from typing import Dict, List

from config import (
    AI_TICKERS, 
    AI_ADJACENT_TICKERS, 
    ALL_TICKERS, 
    BENCHMARK_TICKER, 
    AI_INDEX_TICKER,
    CACHE_TTL_SECONDS
)
from utils.cache import cached

logger = logging.getLogger(__name__)

@cached(ttl=CACHE_TTL_SECONDS)
def fetch_ohlcv(tickers: List[str], start: str, end: str, interval: str = '1d') -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data for a list of tickers using yf.download.
    Returns a dict mapping ticker -> DataFrame with title-cased columns and pd.DatetimeIndex.
    """
    result = {}
    for ticker in tickers:
        try:
            # Setting auto_adjust=True adjusts all OHLC automatically
            df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
            
            if df.empty:
                logger.warning(f"No data fetched for {ticker}")
                result[ticker] = pd.DataFrame()
                continue

            if isinstance(df.columns, pd.MultiIndex):
                # Flatten MultiIndex: keep only the OHLCV field, drop ticker level
                df.columns = df.columns.get_level_values(0)
                
            # Rename 'Adj Close' to 'Adj_Close' if it exists (though auto_adjust=True might drop it)
            df = df.rename(columns={'Adj Close': 'Adj_Close'})
            
            # Ensure columns are title-cased
            df.columns = [str(c).title().replace('Adj_close', 'Adj_Close') for c in df.columns]
            
            # Additional rename for specific naming match
            df = df.rename(columns={'Adj_close': 'Adj_Close'})
                
            result[ticker] = df
        except Exception as e:
            logger.warning(f"Exception while fetching OHLCV for {ticker}: {e}")
            result[ticker] = pd.DataFrame()
    return result

@cached(ttl=86400)
def fetch_ticker_info(ticker: str) -> dict:
    """
    Return a flat dict of key metadata for a single ticker.
    """
    keys_to_extract = ['symbol', 'longName', 'sector', 'industry', 'marketCap', 'trailingPE', 'beta', 'country']
    try:
        info = yf.Ticker(ticker).info
        result = {k: info.get(k) for k in keys_to_extract}
        return result
    except Exception as e:
        logger.warning(f"Exception while fetching info for {ticker}: {e}")
        return {k: None for k in keys_to_extract}

def get_ai_basket(basket_name: str = 'AI Pure-Play') -> List[str]:
    """
    Return the appropriate ticker list from config based on basket_name.
    """
    if basket_name == 'AI Pure-Play':
        return AI_TICKERS
    elif basket_name == 'AI Adjacent':
        return AI_ADJACENT_TICKERS
    elif basket_name == 'Both':
        return ALL_TICKERS
    else:
        raise ValueError(f"Invalid basket_name: '{basket_name}'. Valid options are: 'AI Pure-Play', 'AI Adjacent', 'Both'")

@cached(ttl=CACHE_TTL_SECONDS)
def get_benchmark(start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV for BENCHMARK_TICKER (SPY) and AI_INDEX_TICKER (QQQ).
    """
    res = fetch_ohlcv([BENCHMARK_TICKER, AI_INDEX_TICKER], start, end)
    
    spy_df = res.get(BENCHMARK_TICKER, pd.DataFrame())
    qqq_df = res.get(AI_INDEX_TICKER, pd.DataFrame())
    
    if spy_df.empty:
        raise RuntimeError('Benchmark data unavailable')
        
    df = pd.DataFrame(index=spy_df.index)
    
    if 'Close' in spy_df.columns and 'Volume' in spy_df.columns:
        df['SPY_Close'] = spy_df['Close']
        df['SPY_Volume'] = spy_df['Volume']
    else:
        raise RuntimeError('Benchmark data unavailable: missing required fields')
        
    if not qqq_df.empty and 'Close' in qqq_df.columns and 'Volume' in qqq_df.columns:
        # Align QQQ data to SPY index
        df['QQQ_Close'] = qqq_df['Close']
        df['QQQ_Volume'] = qqq_df['Volume']
        
    # Forward-fill max 2 days for market holidays
    df = df.ffill(limit=2)
    return df
