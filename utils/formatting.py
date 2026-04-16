"""
utils/formatting.py

All numeric display formatting lives here. Pages import these functions 
instead of formatting inline. This ensures consistency across all metric 
cards and table displays.
"""

import pandas as pd
import numpy as np
from typing import Union, Any

def fmt_pct(value: Any, decimals: int = 2, sign: bool = True) -> str:
    """
    Format a decimal float as a percentage string. 
    0.0342 → '+3.42%'. sign=False suppresses the leading '+'.
    Handles None or np.nan -> '—'. Handles negative values: -0.05 -> '-5.00%'.
    """
    if pd.isna(value):
        return '—'
    
    val_pct = float(value) * 100
    format_str = f"{{:{'+' if sign else ''}.{decimals}f}}%"
    return format_str.format(val_pct)

def fmt_dollar(value: Any, decimals: int = 2) -> str:
    """
    Format a float as a dollar string. 
    1234567.8 → '$1,234,567.80'
    Handles None or np.nan -> '—'. Negative: -500 -> '-$500.00'
    """
    if pd.isna(value):
        return '—'
    
    val = float(value)
    is_negative = val < 0
    abs_val = abs(val)
    
    format_str = f"${abs_val:,.{decimals}f}"
    if is_negative:
        return f"-{format_str}"
    return format_str

def fmt_large(value: Any) -> str:
    """
    Abbreviate large numbers. 
    1_234_567 → '$1.23M'
    Thresholds: B (1e9), M (1e6), K (1e3).
    Edge cases: value < 1000 -> fmt_dollar(value). value=None -> '—'.
    """
    if pd.isna(value):
        return '—'
    
    val = float(value)
    is_negative = val < 0
    abs_val = abs(val)
    
    if abs_val < 1000:
        return fmt_dollar(val)
        
    num = abs_val
    suffix = ''
    if num >= 1e9:
        num /= 1e9
        suffix = 'B'
    elif num >= 1e6:
        num /= 1e6
        suffix = 'M'
    elif num >= 1e3:
        num /= 1e3
        suffix = 'K'
    
    formatted_num = f"${num:,.2f}{suffix}"
    if is_negative:
        return f"-{formatted_num}"
    return formatted_num

def fmt_date(dt: Any) -> str:
    """
    Return a human-readable date string. 
    datetime → 'Mar 14, 2023'.
    Accepts str (parses with pd.to_datetime). Returns '—' for NaT/None.
    """
    if pd.isna(dt):
        return '—'
        
    try:
        timestamp = pd.to_datetime(dt)
        if pd.isna(timestamp):
            return '—'
        # Extract date parts manually if needed, or use strftime
        # Example output: Mar 14, 2023
        formatted = timestamp.strftime('%b %d, %Y')
        # Remove zero padding on day
        parts = formatted.split(' ')
        if parts[1].startswith('0'):
            parts[1] = parts[1][1:]
        return ' '.join(parts)
    except Exception:
        return '—'

def fmt_ratio(value: Any, decimals: int = 2) -> str:
    """
    Format a plain ratio (e.g. Sharpe). 
    1.435 → '1.44'. Adds no units.
    Edge cases: None/nan -> '—'.
    """
    if pd.isna(value):
        return '—'
        
    val = float(value)
    format_str = f"{{:.{decimals}f}}"
    return format_str.format(val)
