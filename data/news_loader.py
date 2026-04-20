import logging
import pandas as pd
from typing import List, Dict, Optional

from utils.cache import cached
from config import NEWS_API_KEY, AI_EVENTS

logger = logging.getLogger(__name__)

@cached(ttl=3600)
def fetch_headlines(query: str, from_date: str, to_date: str, page_size: int = 100) -> pd.DataFrame:
    """Fetch news articles from NewsAPI /v2/everything endpoint matching query string."""
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY is empty. Cannot fetch headlines.")
        return pd.DataFrame()
        
    try:
        from newsapi import NewsApiClient
        
        api = NewsApiClient(api_key=NEWS_API_KEY)
        response = api.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            sort_by='relevancy',
            page_size=page_size,
            language='en'
        )
        
        articles = response.get('articles', [])
        if not articles:
             return pd.DataFrame()
             
        df = pd.DataFrame(articles)
        
        # Extract source name
        if 'source' in df.columns:
            df['source'] = df['source'].apply(lambda x: x.get('name') if isinstance(x, dict) else x)
            
        # Select target columns
        target_cols = ['publishedAt', 'title', 'description', 'source', 'url']
        df = df[[c for c in target_cols if c in df.columns]]
        
        # Process publishedAt Date index
        if 'publishedAt' in df.columns:
            df['publishedAt'] = pd.to_datetime(df['publishedAt'], utc=True).dt.date
            df = df.set_index('publishedAt')
            
        return df
        
    except Exception as e:
        logger.warning(f"Exception while fetching headlines from NewsAPI: {e}")
        return pd.DataFrame()

def get_ai_events() -> List[Dict]:
    """Return the AI_EVENTS list from config."""
    return AI_EVENTS

def tag_event_dates(price_df: pd.DataFrame, events: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Add boolean and categorical columns to a price DataFrame marking which rows coincide with an AI event."""
    df = price_df.copy()
    
    if df.empty:
        return df
        
    if events is None:
        events = get_ai_events()
        
    df['is_event_day'] = False
    df['event_label'] = None
    df['event_category'] = None
    
    # Use normalized dates for matching to avoid time component mismatches
    index_dates = df.index.normalize() if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index).normalize()
    
    for ev in events:
        ev_date = pd.to_datetime(ev['date']).normalize()
        
        # Order of checking: Exact day, previous day, 2 days prior (for weekend->Friday), next day
        for offset in [0, -1, -2, 1]:
            candidate = ev_date + pd.Timedelta(days=offset)
            matches = df.index[index_dates == candidate]
            
            if len(matches) > 0:
                target_idx = matches[0]
                df.loc[target_idx, 'is_event_day'] = True
                df.loc[target_idx, 'event_label'] = ev['label']
                df.loc[target_idx, 'event_category'] = ev.get('category')
                break
                
    return df
