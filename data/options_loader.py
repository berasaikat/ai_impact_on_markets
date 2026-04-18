import logging
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple

from utils.cache import cached

logger = logging.getLogger(__name__)

@cached(ttl=3600)
def fetch_all_expiries(ticker: str) -> List[str]:
    """Return all available option expiry date strings for a ticker in YYYY-MM-DD format."""
    try:
        tk_obj = yf.Ticker(ticker)
        expiries = list(tk_obj.options)
        return sorted(expiries)
    except Exception as e:
        logger.warning(f"Error fetching expiries for {ticker}: {e}")
        return []

@cached(ttl=1800)
def fetch_options_chain(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch one expiry's options chain. Returns (calls, puts)."""
    try:
        tk_obj = yf.Ticker(ticker)
        chain = tk_obj.option_chain(expiry)
        
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        
        desired_cols = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'inTheMoney']
        
        calls = calls[[c for c in desired_cols if c in calls.columns]]
        puts = puts[[c for c in desired_cols if c in puts.columns]]
        
        calls = calls.reset_index(drop=True)
        puts = puts.reset_index(drop=True)
        
        numeric_cols = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
        for col in numeric_cols:
            if col in calls.columns:
                calls[col] = pd.to_numeric(calls[col], errors='coerce').astype('float64')
            if col in puts.columns:
                puts[col] = pd.to_numeric(puts[col], errors='coerce').astype('float64')
                
        # Drop rows where IV == 0 or IV <= 0
        if 'impliedVolatility' in calls.columns:
            calls = calls[calls['impliedVolatility'] > 0]
        if 'impliedVolatility' in puts.columns:
            puts = puts[puts['impliedVolatility'] > 0]
            
        return calls, puts
    except Exception as e:
        logger.warning(f"Error fetching options chain for {ticker} at {expiry}: {e}")
        return pd.DataFrame(), pd.DataFrame()

@cached(ttl=1800)
def build_iv_matrix(ticker: str, max_expiries: int = 8) -> pd.DataFrame:
    """Build a 2D matrix of implied volatility with strikes as index and expiry dates as columns."""
    expiries = fetch_all_expiries(ticker)
    if not expiries:
        return pd.DataFrame()
        
    expiries = expiries[:max_expiries]
    
    try:
        spot_history = yf.Ticker(ticker).history(period='1d')
        if spot_history.empty:
            return pd.DataFrame()
        spot_price = float(spot_history['Close'].iloc[-1])
    except Exception as e:
        logger.warning(f"Error fetching spot price for {ticker}: {e}")
        return pd.DataFrame()

    iv_data = []
    
    for exp in expiries:
        calls, puts = fetch_options_chain(ticker, exp)
        
        combined = pd.concat([calls, puts], ignore_index=True)
        if combined.empty or 'strike' not in combined.columns or 'impliedVolatility' not in combined.columns:
            continue
            
        avg_iv = combined.groupby('strike')['impliedVolatility'].mean().reset_index()
        avg_iv['expiry'] = exp
        iv_data.append(avg_iv)
        
    if not iv_data:
        return pd.DataFrame()
        
    all_iv = pd.concat(iv_data, ignore_index=True)
    
    iv_matrix = all_iv.pivot(index='strike', columns='expiry', values='impliedVolatility')
    iv_matrix = iv_matrix.dropna(how='all')
    
    lower_bound = spot_price * 0.4
    upper_bound = spot_price * 1.6
    
    iv_matrix = iv_matrix[(iv_matrix.index >= lower_bound) & (iv_matrix.index <= upper_bound)]
    
    iv_matrix.index = iv_matrix.index.astype('float64')
    iv_matrix = iv_matrix.sort_index(ascending=True)
    
    return iv_matrix

@cached(ttl=1800)
def get_put_call_ratio(ticker: str, expiry: str) -> float:
    """Compute put-call ratio by volume for a given expiry."""
    calls, puts = fetch_options_chain(ticker, expiry)
    if calls.empty and puts.empty:
        return np.nan
        
    calls_vol = calls['volume'].sum() if 'volume' in calls.columns else 0
    puts_vol = puts['volume'].sum() if 'volume' in puts.columns else 0
    
    if calls_vol == 0 and puts_vol == 0:
        calls_vol = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
        puts_vol = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
        
    if calls_vol == 0 or pd.isna(calls_vol):
        logger.warning(f"Calls volume/OI is 0 for {ticker} at {expiry}. PCR is undefined.")
        return np.nan
        
    return float(puts_vol / calls_vol)
