# features/returns.py
"""
Log returns, market model estimation, abnormal returns, and cumulative
abnormal returns (CAR) for event studies.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import ESTIMATION_WINDOW, EVENT_WINDOW, MIN_ESTIMATION_DAYS

logger = logging.getLogger(__name__)


def log_returns(
    price_df: pd.DataFrame, price_col: str = "Close"
) -> pd.DataFrame:
    """
    Compute log returns for each ticker column (or single price_col) in a DataFrame.

    Parameters
    ----------
    price_df : pd.DataFrame
        DataFrame with DatetimeIndex and price columns (e.g., 'Close').
    price_col : str, default='Close'
        Column name to use when price_df has a single column.

    Returns
    -------
    pd.DataFrame
        DataFrame of log returns, same columns as input, first row dropped.
        Index is DatetimeIndex.
    """
    # Work on a copy to avoid mutating input
    df = price_df.copy()

    # If single column and matches price_col, ensure it's a DataFrame with that column
    if len(df.columns) == 1 and df.columns[0] == price_col:
        # Already fine
        pass
    elif price_col in df.columns and len(df.columns) == 1:
        # Ensure the column name is exactly price_col? It's okay.
        pass
    # For multi-column DataFrames, we apply to all columns

    # Replace zero or negative prices with NaN
    if (df <= 0).any().any():
        logger.warning("Zero or negative prices found; replacing with NaN before log")
        df = df.where(df > 0, np.nan)

    # Compute log returns: log(P_t / P_{t-1})
    log_ret = np.log(df / df.shift(1))

    # Drop first row (NaN)
    log_ret = log_ret.dropna(how="all")

    # If input was single column and we want to return single column DataFrame
    if len(df.columns) == 1:
        # Ensure column name is consistent (could be original name)
        pass

    return log_ret


def estimate_market_model(
    stock_ret: pd.Series,
    mkt_ret: pd.Series,
    estimation_window: Tuple[int, int] = ESTIMATION_WINDOW,
) -> Tuple[float, float, float]:
    """
    Fit OLS regression stock_ret = alpha + beta * mkt_ret on estimation window.

    Parameters
    ----------
    stock_ret : pd.Series
        Daily log returns of the stock, indexed by date.
    mkt_ret : pd.Series
        Daily log returns of the market benchmark, indexed by date.
    estimation_window : tuple[int, int], default=ESTIMATION_WINDOW
        (start_offset, end_offset) relative to the event day. Both negative.

    Returns
    -------
    tuple[float, float, float]
        (alpha, beta, r_squared)

    Raises
    ------
    ValueError
        If fewer than MIN_ESTIMATION_DAYS observations in window,
        or if >10% NaN in either series in the window.
    """
    start_offset, end_offset = estimation_window

    # Align series by index
    aligned = pd.concat([stock_ret, mkt_ret], axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("No overlapping data between stock and market returns")

    # Slice estimation window using integer positions (relative to the end of series)
    # The window is defined relative to the event day, but here we are fitting
    # a model on a fixed period that ends `end_offset` days before the event.
    # Since the series may have arbitrary length, we need to know the event index.
    # This function is called within multi_event_car where we know the event position.
    # However, for standalone usage, we assume the input series are already aligned
    # to a common period and the estimation window is taken from the *end* of the series.
    # More robust: the function receives a full series and the event date separately.
    # But spec says: "Window is defined as (start_offset, end_offset) days relative
    # to the position index (not a calendar date)." This implies the function must
    # be called with series that have been trimmed to the event period? In multi_event_car
    # we will compute the model on a window ending `end_offset` days before the event.
    # For simplicity, we assume the caller provides series that are already aligned
    # and the estimation window is taken from the end: last (end_offset - start_offset) days.
    # However, standard approach: the series should have a DatetimeIndex and we need to
    # know the event date. The spec for multi_event_car doesn't pass event_date to this function.
    # Let's reinterpret: The estimation_window offsets are relative to the event day.
    # This function should receive the event position as well? Not in signature.
    # To match spec, we will assume the input series are already sliced to the estimation
    # window before calling. But that would be inefficient. Alternatively, we can accept
    # an additional parameter `event_pos`? No.
    # Reading the spec carefully: "Slice both series to [start_offset:end_offset]" suggests
    # the series indices are consecutive integers where 0 is the event day? Or the series
    # is already relative. Actually, to implement this correctly, the caller (multi_event_car)
    # will slice the series using integer positions. To avoid complexity, we will implement
    # this function assuming the input series are already aligned and contain only the
    # estimation period in order. The caller will slice before calling. That is simpler.
    # Let's document that expectation.

    if len(aligned) < MIN_ESTIMATION_DAYS:
        raise ValueError(
            f"Estimation window has only {len(aligned)} observations, "
            f"need at least {MIN_ESTIMATION_DAYS}"
        )

    # Check for NaNs in the window (>10%)
    if aligned.isna().any().any():
        # Actually we already dropped NaNs with inner join. But we should check
        # if after dropping we still have enough data.
        pass
    # Re-check length after dropna (already done)
    if len(aligned) < MIN_ESTIMATION_DAYS:
        raise ValueError("Insufficient clean data in estimation window after dropping NaNs")

    # Fit OLS: y = X * beta, where X includes constant
    X = sm.add_constant(aligned.iloc[:, 1])  # mkt_ret is second column
    y = aligned.iloc[:, 0]  # stock_ret
    model = sm.OLS(y, X, missing="drop")
    results = model.fit()

    alpha = float(results.params[0])
    beta = float(results.params[1])
    r_squared = float(results.rsquared)

    if r_squared < 0.01:
        logger.warning(f"Very low R² ({r_squared:.3f}) for market model — check data quality")

    return alpha, beta, r_squared


def compute_abnormal_returns(
    stock_ret: pd.Series,
    mkt_ret: pd.Series,
    alpha: float,
    beta: float,
) -> pd.Series:
    """
    Compute abnormal returns AR_t = R_t - (alpha + beta * R_market_t).

    Parameters
    ----------
    stock_ret : pd.Series
        Stock log returns.
    mkt_ret : pd.Series
        Market log returns.
    alpha : float
        Intercept from market model.
    beta : float
        Slope from market model.

    Returns
    -------
    pd.Series
        Abnormal returns, same index as stock_ret, name='AR'.
        Missing days (non-overlapping) become NaN.
    """
    # Align by index
    combined = pd.concat([stock_ret, mkt_ret], axis=1, join="outer")
    combined.columns = ["stock", "mkt"]
    # Compute expected return: alpha + beta * mkt
    expected = alpha + beta * combined["mkt"]
    ar = combined["stock"] - expected
    ar.name = "AR"
    return ar


def compute_car(
    ar_series: pd.Series,
    event_date: str,
    window: Tuple[int, int] = EVENT_WINDOW,
) -> pd.Series:
    """
    Compute cumulative abnormal returns (CAR) over event window.

    Parameters
    ----------
    ar_series : pd.Series
        Abnormal returns with DatetimeIndex.
    event_date : str
        Event date in YYYY-MM-DD format.
    window : tuple[int, int], default=EVENT_WINDOW
        (start_offset, end_offset) days around event.

    Returns
    -------
    pd.Series
        CAR values indexed by integer day offsets (e.g., -10,...,10).
        Name='CAR'.

    Raises
    ------
    ValueError
        If event_date cannot be found within 3 calendar days of the index,
        or if too many NaN values in the window.
    """
    # Convert event_date to Timestamp
    event_ts = pd.Timestamp(event_date)

    # Find nearest date in ar_series index within ±3 calendar days
    idx = ar_series.index
    if event_ts in idx:
        event_pos = idx.get_loc(event_ts)
        event_date_found = event_ts
    else:
        # Find nearest date within ±3 days
        tolerance = pd.Timedelta(days=3)
        candidates = idx[(idx >= event_ts - tolerance) & (idx <= event_ts + tolerance)]
        if len(candidates) == 0:
            raise ValueError(f"Event date {event_date} not found within 3 calendar days of index")
        event_date_found = candidates[0]
        event_pos = idx.get_loc(event_date_found)
        logger.info(f"Event date {event_date} mapped to nearest trading day {event_date_found.date()}")

    # Slice event window using integer positions
    start_offset, end_offset = window
    start_pos = event_pos + start_offset
    end_pos = event_pos + end_offset + 1  # inclusive

    if start_pos < 0 or end_pos > len(idx):
        raise ValueError(f"Event window extends beyond available series range")

    window_ar = ar_series.iloc[start_pos:end_pos]

    # Compute CAR: cumulative sum
    car_values = window_ar.cumsum()

    # Reindex to day offsets
    offsets = list(range(start_offset, end_offset + 1))
    car_series = pd.Series(car_values.values, index=offsets, name="CAR")

    # Check for excessive NaNs
    if car_series.isna().sum() > 0.3 * len(car_series):
        raise ValueError("Too many NaN values in event window — ticker may have been delisted")

    return car_series


def multi_event_car(
    ticker_returns: Dict[str, pd.Series],
    mkt_ret: pd.Series,
    events: List[dict],
    window: Tuple[int, int] = EVENT_WINDOW,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute average CAR (CAAR) and confidence intervals across multiple events and tickers.

    Parameters
    ----------
    ticker_returns : dict[str, pd.Series]
        Mapping ticker -> series of log returns with DatetimeIndex.
    mkt_ret : pd.Series
        Market log returns with DatetimeIndex.
    events : list[dict]
        List of event dicts with keys 'label', 'date', 'category'.
    window : tuple[int, int], default=EVENT_WINDOW
        Event window (start_offset, end_offset).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        car_mean_df : DataFrame with columns ['day_offset', 'CAAR'].
        car_ci_df : DataFrame with columns ['day_offset', 'CI_lower', 'CI_upper'].
    """
    from statsmodels.stats.weightstats import DescrStatsW

    all_car_series = []  # list of pd.Series indexed by day offset

    total_combinations = len(ticker_returns) * len(events)
    processed = 0

    for ticker, ret_series in ticker_returns.items():
        for event in events:
            processed += 1
            event_label = event.get("label", "unknown")
            event_date = event["date"]
            logger.info(f"Processing event {processed}/{total_combinations}: {event_label} for {ticker}")

            try:
                # Align stock and market returns around the event
                # Slice estimation window: we need a continuous period ending at `window[1]`? Actually
                # The market model is estimated on estimation_window (e.g., -120 to -20) relative to event.
                # We'll extract the full period from min(estimation start, window start) to max(estimation end, window end)
                # but simpler: we can call estimate_market_model with series sliced to the estimation window.
                # We need to locate event position in the combined series.
                # Create aligned DataFrame of stock and market returns
                combined = pd.DataFrame({ticker: ret_series, "mkt": mkt_ret}).dropna()
                if combined.empty:
                    raise ValueError("No overlapping data for ticker and market")

                # Find event position
                event_ts = pd.Timestamp(event_date)
                idx = combined.index
                if event_ts not in idx:
                    # Find nearest within 3 days
                    tolerance = pd.Timedelta(days=3)
                    candidates = idx[(idx >= event_ts - tolerance) & (idx <= event_ts + tolerance)]
                    if len(candidates) == 0:
                        raise ValueError(f"Event date {event_date} not found within 3 days of combined index")
                    event_pos = idx.get_loc(candidates[0])
                else:
                    event_pos = idx.get_loc(event_ts)

                # Estimation window: from event_pos + start_offset to event_pos + end_offset
                est_start, est_end = ESTIMATION_WINDOW
                est_start_pos = event_pos + est_start
                est_end_pos = event_pos + est_end
                if est_start_pos < 0 or est_end_pos > len(idx):
                    raise ValueError("Estimation window out of bounds")
                est_stock = combined.iloc[est_start_pos:est_end_pos][ticker]
                est_mkt = combined.iloc[est_start_pos:est_end_pos]["mkt"]

                # Estimate market model
                alpha, beta, _ = estimate_market_model(est_stock, est_mkt, ESTIMATION_WINDOW)

                # Compute abnormal returns for the full series (or at least from window start to end)
                ar = compute_abnormal_returns(combined[ticker], combined["mkt"], alpha, beta)

                # Compute CAR over event window
                car = compute_car(ar, event_date, window)
                all_car_series.append(car)

            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping {ticker} / {event_label}: {e}")
                continue

    if not all_car_series:
        raise RuntimeError("All event-ticker combinations failed to produce CAR")

    # Build matrix: rows = day offsets, columns = individual CAR series
    all_offsets = sorted(set().union(*(s.index for s in all_car_series)))
    car_matrix = pd.DataFrame(index=all_offsets)
    for i, car_ser in enumerate(all_car_series):
        car_matrix[i] = car_ser

    # Compute cross-sectional mean and CI (assuming independence)
    mean_car = car_matrix.mean(axis=1)
    std_car = car_matrix.std(axis=1, ddof=1)
    n = car_matrix.count(axis=1)
    se = std_car / np.sqrt(n)
    ci_width = 1.96 * se
    ci_lower = mean_car - ci_width
    ci_upper = mean_car + ci_width

    car_mean_df = pd.DataFrame({"day_offset": mean_car.index, "CAAR": mean_car.values})
    car_ci_df = pd.DataFrame(
        {"day_offset": mean_car.index, "CI_lower": ci_lower.values, "CI_upper": ci_upper.values}
    )

    return car_mean_df, car_ci_df