import pandas as pd
import numpy as np
import logging
from config import ROLLING_BETA_WINDOW

def rolling_beta(stock_ret: pd.Series, mkt_ret: pd.Series, window: int = ROLLING_BETA_WINDOW) -> pd.Series:
    """
    Compute rolling OLS beta of stock vs market over a trailing window.
    """
    cov = stock_ret.rolling(window=window).cov(mkt_ret)
    var = mkt_ret.rolling(window=window).var()
    
    # To handle zero variance
    beta = cov / var
    beta = beta.where(var != 0, np.nan)
    
    if not beta.empty:
        nan_ratio = beta.isna().sum() / len(beta)
        if nan_ratio > 0.05:
            logging.warning(f"More than 5% ({nan_ratio:.1%}) of beta values are NaN for {stock_ret.name}.")
            
    beta.name = f"{stock_ret.name}_beta"
    return beta.astype('float64')

def beta_vs_ai_index(ticker_ret: pd.Series, ai_index_ret: pd.Series, window: int = ROLLING_BETA_WINDOW) -> pd.Series:
    """
    Same computation as rolling_beta but names the result clearly as AI-index beta.
    """
    beta = rolling_beta(ticker_ret, ai_index_ret, window)
    beta.name = f"{ticker_ret.name}_ai_beta"
    return beta

def beta_change_event(beta_series: pd.Series, event_date: str, pre_days: int = 30, post_days: int = 30) -> dict:
    """
    Quantify how a stock's beta changed around an event. 
    Compares the mean beta in the pre_days window before the event to the mean beta in the post_days window after.
    """
    if beta_series.empty:
        return {
            'pre_beta': np.nan, 'post_beta': np.nan, 'delta': np.nan, 
            'event_date': event_date, 'pct_change': np.nan, 'insufficient_data': True
        }
        
    target_date = pd.to_datetime(event_date)
    
    try:
        time_diff = np.abs((beta_series.index - target_date).days)
    except AttributeError:
        # Fallback if index is not naturally yielding .days
        time_diff = np.abs((pd.to_datetime(beta_series.index) - target_date).days)
        
    min_diff = time_diff.min()
    
    if min_diff > 3:
        return {
            'pre_beta': np.nan, 'post_beta': np.nan, 'delta': np.nan, 
            'event_date': event_date, 'pct_change': np.nan, 'insufficient_data': True
        }
        
    pos = time_diff.argmin()
    actual_event_date = beta_series.index[pos]
    
    start_pre = max(0, pos - pre_days)
    pre_slice = beta_series.iloc[start_pre:pos]
    post_slice = beta_series.iloc[pos+1:pos+1+post_days]
    
    pre_count = pre_slice.count()
    post_count = post_slice.count()
    
    insufficient = False
    
    if pre_count < 10:
        pre_mean = np.nan
        insufficient = True
    else:
        pre_mean = pre_slice.mean()
        
    if post_count < 10:
        post_mean = np.nan
        insufficient = True
    else:
        post_mean = post_slice.mean()
        
    delta = post_mean - pre_mean if not insufficient else np.nan
    pct_change = (delta / pre_mean) if (not insufficient and pre_mean != 0 and pd.notna(pre_mean)) else np.nan
    
    res = {
        'pre_beta': float(pre_mean) if pd.notna(pre_mean) else np.nan,
        'post_beta': float(post_mean) if pd.notna(post_mean) else np.nan,
        'delta': float(delta) if pd.notna(delta) else np.nan,
        'event_date': actual_event_date.strftime('%Y-%m-%d') if hasattr(actual_event_date, 'strftime') else str(actual_event_date),
        'pct_change': float(pct_change) if pd.notna(pct_change) else np.nan
    }
    
    if insufficient:
        res['insufficient_data'] = True
        
    return res
