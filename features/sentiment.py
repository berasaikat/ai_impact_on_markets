"""
Sentiment analysis using VADER: score headlines, aggregate daily sentiment,
detect AI events from news, and compute correlation with prices.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Module-level singleton VADER analyzer
_VADER_ANALYZER = SentimentIntensityAnalyzer()

# Default AI keywords for NLP event detection
_DEFAULT_AI_KEYWORDS: List[str] = [
    "ai",
    "artificial intelligence",
    "chatgpt",
    "gpt",
    "llm",
    "nvidia",
    "deep learning",
    "foundation model",
]


def score_headlines(headlines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply VADER sentiment scoring to a headlines DataFrame.

    Adds four columns: vader_pos, vader_neg, vader_neu, vader_compound.

    Parameters
    ----------
    headlines_df : pd.DataFrame
        DataFrame with at least 'title' and 'description' columns.

    Returns
    -------
    pd.DataFrame
        Copy of input with sentiment columns appended. If input is empty,
        returns unchanged empty DataFrame.
    """
    if headlines_df.empty:
        return headlines_df.copy()

    # Work on a copy
    df = headlines_df.copy()

    # Create combined text: title + ' ' + description, fill NaN with empty string
    title_col = "title" if "title" in df.columns else None
    desc_col = "description" if "description" in df.columns else None

    if title_col is None:
        # If no title, use description only
        combined = df[desc_col].fillna("") if desc_col else pd.Series([""] * len(df))
    else:
        if desc_col:
            combined = df[title_col].fillna("") + " " + df[desc_col].fillna("")
        else:
            combined = df[title_col].fillna("")

    # Apply VADER to each row
    def get_scores(text: str) -> dict:
        try:
            return _VADER_ANALYZER.polarity_scores(text)
        except Exception:
            return {"pos": 0.0, "neg": 0.0, "neu": 0.0, "compound": 0.0}

    scores = combined.apply(get_scores)
    df["vader_pos"] = scores.apply(lambda x: x["pos"]).astype(float)
    df["vader_neg"] = scores.apply(lambda x: x["neg"]).astype(float)
    df["vader_neu"] = scores.apply(lambda x: x["neu"]).astype(float)
    df["vader_compound"] = scores.apply(lambda x: x["compound"]).astype(float)

    return df


def daily_sentiment(
    scored_df: pd.DataFrame, date_col: str = "publishedAt"
) -> pd.Series:
    """
    Aggregate per-article VADER scores to daily mean sentiment.

    Parameters
    ----------
    scored_df : pd.DataFrame
        DataFrame with a date column and a 'vader_compound' column.
    date_col : str, default='publishedAt'
        Name of the column containing article dates.

    Returns
    -------
    pd.Series
        Daily mean sentiment, indexed by datetime (date only, no time).
        Name = 'daily_sentiment', sorted ascending.
        Empty Series if scored_df is empty.
    """
    if scored_df.empty:
        return pd.Series(dtype=float, name="daily_sentiment")

    # Ensure date column exists
    if date_col not in scored_df.columns:
        logger.warning(f"Column '{date_col}' not found in DataFrame")
        return pd.Series(dtype=float, name="daily_sentiment")

    # Ensure vader_compound exists
    if "vader_compound" not in scored_df.columns:
        logger.warning("'vader_compound' column not found; run score_headlines first")
        return pd.Series(dtype=float, name="daily_sentiment")

    # Convert to datetime and extract date part
    dates = pd.to_datetime(scored_df[date_col])
    grouped = scored_df.groupby(dates.dt.date)["vader_compound"].mean()

    # Convert index back to datetime
    grouped.index = pd.to_datetime(grouped.index)
    grouped = grouped.sort_index()
    grouped.name = "daily_sentiment"

    return grouped


def detect_ai_events_nlp(
    scored_df: pd.DataFrame,
    sentiment_threshold: float = 0.3,
    keywords: Optional[List[str]] = None,
) -> List[str]:
    """
    Identify dates where high sentiment magnitude and AI keywords co-occur.

    Parameters
    ----------
    scored_df : pd.DataFrame
        DataFrame with 'title' and 'vader_compound' columns.
    sentiment_threshold : float, default=0.3
        Minimum absolute compound score to consider an event.
    keywords : list[str] | None, default=None
        List of keywords to match in titles (case-insensitive).
        If None, uses default AI keywords.

    Returns
    -------
    list[str]
        Sorted list of date strings (YYYY-MM-DD) of detected events.
    """
    if scored_df.empty:
        return []

    # Use default keywords if none provided
    if keywords is None:
        keywords = _DEFAULT_AI_KEYWORDS

    # Ensure required columns exist
    if "title" not in scored_df.columns:
        logger.warning("'title' column missing, cannot detect keyword events")
        return []

    if "vader_compound" not in scored_df.columns:
        logger.warning("'vader_compound' missing; run score_headlines first")
        return []

    # Filter rows where any keyword appears in title (case-insensitive)
    title_lower = scored_df["title"].fillna("").str.lower()
    keyword_pattern = "|".join(keywords)
    mask_keyword = title_lower.str.contains(keyword_pattern, na=False)

    if not mask_keyword.any():
        return []

    # Subset to keyword-matching articles
    keyword_df = scored_df[mask_keyword].copy()

    # Convert publishedAt to date if present
    date_col = "publishedAt" if "publishedAt" in keyword_df.columns else None
    if date_col is None:
        logger.warning("No 'publishedAt' column, cannot aggregate by date")
        return []

    # Group by date and compute max absolute compound score
    dates = pd.to_datetime(keyword_df[date_col]).dt.date
    daily_max_abs = keyword_df.groupby(dates)["vader_compound"].apply(
        lambda x: np.abs(x).max()
    )

    # Keep dates where max abs sentiment >= threshold
    event_dates = daily_max_abs[daily_max_abs >= sentiment_threshold].index
    # Convert back to YYYY-MM-DD strings and sort
    result = sorted([d.strftime("%Y-%m-%d") for d in event_dates])

    return result


def sentiment_price_corr(
    sentiment_series: pd.Series,
    price_series: pd.Series,
    lag: int = 0,
) -> float:
    """
    Compute Pearson correlation between sentiment and stock returns with optional lag.

    Parameters
    ----------
    sentiment_series : pd.Series
        Daily sentiment scores (e.g., from daily_sentiment), indexed by datetime.
    price_series : pd.Series
        Daily stock prices (close), indexed by datetime.
    lag : int, default=0
        If positive: sentiment leads price (shift sentiment forward).
        If negative: price leads sentiment (shift price forward).

    Returns
    -------
    float
        Correlation coefficient in [-1, 1]. Returns np.nan if fewer than 10
        overlapping observations.
    """
    # Ensure both series are aligned to daily frequency
    # Convert price_series to log returns
    if price_series.isnull().any():
        price_series = price_series.dropna()

    if len(price_series) < 2:
        logger.warning("Insufficient price data to compute returns")
        return np.nan

    # Compute log returns
    price_returns = np.log(price_series / price_series.shift(1)).dropna()

    # Align sentiment and returns by date index
    combined = pd.DataFrame(
        {"sentiment": sentiment_series, "returns": price_returns}
    ).dropna()

    if combined.empty:
        logger.warning("No overlapping dates between sentiment and price returns")
        return np.nan

    # Apply lag
    if lag > 0:
        # sentiment leads price: shift sentiment forward so that sentiment[t] aligns with returns[t+lag]
        combined["sentiment"] = combined["sentiment"].shift(lag)
    elif lag < 0:
        # price leads sentiment: shift returns forward so that returns[t] aligns with sentiment[t-lag]
        combined["returns"] = combined["returns"].shift(-lag)

    # Drop rows with NaN after shift
    combined = combined.dropna()

    n_obs = len(combined)
    if n_obs < 10:
        logger.warning(f"Only {n_obs} overlapping observations; insufficient for correlation")
        return np.nan

    if n_obs < 30:
        logger.warning(f"Correlation computed on only {n_obs} observations (recommend >=30)")

    # Compute Pearson correlation
    corr, _ = pearsonr(combined["sentiment"], combined["returns"])
    return float(corr)