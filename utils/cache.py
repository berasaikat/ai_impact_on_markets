"""
utils/cache.py

Provides caching wrappers to abstract away Streamlit context where necessary,
and to manage data cache lifecycles consistently.
"""

import streamlit as st
from typing import Callable

# Import the default TTL from config to ensure consistency across the app
from config import CACHE_TTL_SECONDS

def cached(ttl: int = CACHE_TTL_SECONDS) -> Callable:
    """
    Returns st.cache_data(ttl=ttl) — a pass-through decorator factory.
    
    Import and re-export so pages only import from utils.cache, never from
    streamlit directly.

    Usage:
        @cached()
        def fetch_ohlcv(...):
            ...
            
        @cached(ttl=300)
        def fetch_fast(...):
            ...
    """
    return st.cache_data(ttl=ttl)

def clear_all_caches() -> None:
    """
    Calls st.cache_data.clear().
    
    Used by the sidebar 'Refresh data' button in app.py to clear
    all memoised Streamlit cache entries.
    """
    st.cache_data.clear()
