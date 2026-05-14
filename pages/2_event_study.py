# pages/2_event_study.py
"""
Event Study page: Cumulative abnormal returns around AI events.
"""

import logging
import traceback
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.car_chart import car_chart, caar_bar_chart
from config import AI_TICKERS, BENCHMARK_TICKER, ESTIMATION_WINDOW
from data import fetch_ohlcv, get_ai_basket, get_ai_events, get_benchmark
from features.returns import log_returns, multi_event_car
from features.volatility import realized_vol
from utils.formatting import fmt_pct, fmt_ratio

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(layout="wide")

# ── Read global session state ────────────────────────────────────────────────
start = st.session_state.get("start_date", pd.Timestamp("2022-01-01").date())
end = st.session_state.get("end_date", pd.Timestamp.today().date())
basket = st.session_state.get("basket", "AI Pure-Play")

# ── Page-local sidebar controls ──────────────────────────────────────────────
all_events = get_ai_events()
event_labels = [e["label"] for e in all_events]
default_events = event_labels[:5] if len(event_labels) >= 5 else event_labels

selected_event_labels = st.sidebar.multiselect(
    "Select events", options=event_labels, default=default_events
)

all_tickers = get_ai_basket(basket)
default_tickers = all_tickers[:5] if len(all_tickers) >= 5 else all_tickers
selected_tickers = st.sidebar.multiselect(
    "Tickers for averaging", options=all_tickers, default=default_tickers
)

window_days = st.sidebar.slider("Event window (days)", min_value=5, max_value=20, value=10)
show_individual = st.sidebar.checkbox("Show individual CARs", value=False)

# Ensure we have at least one event and one ticker
if not selected_event_labels:
    st.warning("Please select at least one event.")
    st.stop()
if not selected_tickers:
    st.warning("Please select at least one ticker.")
    st.stop()

# ── Data loading ─────────────────────────────────────────────────────────────
with st.spinner("Loading price data and computing event study..."):
    # Fetch OHLCV for selected tickers plus benchmark
    tickers_to_fetch = list(set(selected_tickers + [BENCHMARK_TICKER]))
    price_dict = fetch_ohlcv(tickers_to_fetch, start, end)

    # Filter out tickers with empty data
    valid_tickers = [t for t in selected_tickers if not price_dict.get(t, pd.DataFrame()).empty]
    if BENCHMARK_TICKER not in price_dict or price_dict[BENCHMARK_TICKER].empty:
        st.error(f"Benchmark {BENCHMARK_TICKER} data unavailable.")
        st.stop()
    if not valid_tickers:
        st.error("No valid price data for selected tickers.")
        st.stop()

    # Compute log returns
    ticker_returns = {}
    for t in valid_tickers:
        rets = log_returns(price_dict[t][["Close"]])["Close"].dropna()
        if not rets.empty:
            ticker_returns[t] = rets

    mkt_ret = log_returns(price_dict[BENCHMARK_TICKER][["Close"]])["Close"].dropna()
    if mkt_ret.empty:
        st.error("Benchmark returns empty.")
        st.stop()

    # Filter selected events
    selected_events = [e for e in all_events if e["label"] in selected_event_labels]

    # Event window tuple
    event_window = (-window_days, window_days)

    # Run multi_event_car
    try:
        car_mean_df, car_ci_df = multi_event_car(
            ticker_returns=ticker_returns,
            mkt_ret=mkt_ret,
            events=selected_events,
            window=event_window,
        )
    except Exception as e:
        st.error(f"Event study computation failed: {e}")
        st.stop()

    # For individual CARs (if requested)
    individual_cars = None
    if show_individual:
        # Recompute but store each CAR series (simplified: we could have multi_event_car return them,
        # but spec says to collect within page. Since multi_event_car only returns mean and CI,
        # we need to compute individual CARs separately. We'll re-run a loop similar to multi_event_car
        # but storing each CAR series.
        # To avoid code duplication, we'll compute individually here.
        individual_cars = []
        for ticker, ret_ser in ticker_returns.items():
            for event in selected_events:
                try:
                    from features.returns import estimate_market_model, compute_abnormal_returns, compute_car
                    # Align series
                    combined = pd.DataFrame({ticker: ret_ser, "mkt": mkt_ret}).dropna()
                    if combined.empty:
                        continue
                    event_ts = pd.Timestamp(event["date"])
                    idx = combined.index
                    # Find nearest trading day
                    if event_ts not in idx:
                        tolerance = pd.Timedelta(days=3)
                        candidates = idx[(idx >= event_ts - tolerance) & (idx <= event_ts + tolerance)]
                        if len(candidates) == 0:
                            continue
                        event_pos = idx.get_loc(candidates[0])
                    else:
                        event_pos = idx.get_loc(event_ts)
                    est_start, est_end = ESTIMATION_WINDOW
                    est_start_pos = event_pos + est_start
                    est_end_pos = event_pos + est_end
                    if est_start_pos < 0 or est_end_pos > len(idx):
                        continue
                    est_stock = combined.iloc[est_start_pos:est_end_pos][ticker]
                    est_mkt = combined.iloc[est_start_pos:est_end_pos]["mkt"]
                    alpha, beta, _ = estimate_market_model(est_stock, est_mkt)
                    ar = compute_abnormal_returns(combined[ticker], combined["mkt"], alpha, beta)
                    car = compute_car(ar, event["date"], window=event_window)
                    individual_cars.append(car)
                except Exception:
                    continue
        if not individual_cars:
            individual_cars = None

# Compute metrics from car_mean_df
if car_mean_df is not None and not car_mean_df.empty:
    # Map day_offset to values
    caar_dict = car_mean_df.set_index("day_offset")["CAAR"].to_dict()
    caar_at_0 = caar_dict.get(0, np.nan)
    caar_at_10 = caar_dict.get(window_days, np.nan)  # t=+window_days
    # Pre-event drift: mean CAR from t=-5 to t=-1
    pre_days = [-5, -4, -3, -2, -1]
    pre_values = [caar_dict.get(d, np.nan) for d in pre_days if d in caar_dict]
    pre_drift = np.nanmean(pre_values) if pre_values else np.nan
    # CI width at t=0
    ci_at_0 = None
    if car_ci_df is not None:
        ci_row = car_ci_df[car_ci_df["day_offset"] == 0]
        if not ci_row.empty:
            ci_width = ci_row["CI_upper"].values[0] - ci_row["CI_lower"].values[0]
        else:
            ci_width = np.nan
    else:
        ci_width = np.nan
else:
    caar_at_0 = caar_at_10 = pre_drift = ci_width = np.nan

# Row 1: Metric cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("CAAR at t=0", fmt_pct(caar_at_0) if not np.isnan(caar_at_0) else "---")
with col2:
    st.metric(f"CAAR at t=+{window_days}", fmt_pct(caar_at_10) if not np.isnan(caar_at_10) else "---")
with col3:
    st.metric("Pre-event drift (t=-5 to -1)", fmt_pct(pre_drift) if not np.isnan(pre_drift) else "---")
with col4:
    st.metric("CI width at t=0", fmt_pct(ci_width) if not np.isnan(ci_width) else "---")

# Row 2: CAR chart
st.subheader("Average Cumulative Abnormal Return (CAAR)")
if car_mean_df is not None and not car_mean_df.empty:
    try:
        fig_car = car_chart(
            car_df=car_mean_df,
            ci_df=car_ci_df if car_ci_df is not None else pd.DataFrame(),
            event_label=f"{len(selected_events)} events / {len(valid_tickers)} tickers",
            show_individual=show_individual,
            individual_cars=individual_cars,
        )
        st.plotly_chart(fig_car, use_container_width=True)
    except Exception as e:
        st.error("CAR chart failed to render")
        logging.error(traceback.format_exc())
else:
    st.info("No CAR data to display.")

# Row 3: Two columns – CAAR bar chart and per-event table
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("CAAR at t=+5 by Event")
    # Build dict of event label -> CAAR at day +5 (or +window_days). Use day +5 if window_days >=5 else final day.
    target_day = min(5, window_days) if window_days >= 5 else window_days
    # Recompute per-event CAAR (since multi_event_car gave average across all events, not per event)
    # We need to compute per-event CAAR for the bar chart. We'll loop through events and compute CAAR for that event alone.
    event_caars = {}
    for event in selected_events:
        try:
            # Single-event CAR
            # Use the same multi_event_car but with a single event
            single_car_mean, _ = multi_event_car(
                ticker_returns=ticker_returns,
                mkt_ret=mkt_ret,
                events=[event],
                window=event_window,
            )
            if single_car_mean is not None and not single_car_mean.empty:
                caar_val = single_car_mean.set_index("day_offset")["CAAR"].get(target_day, np.nan)
                if not np.isnan(caar_val):
                    event_caars[event["label"]] = caar_val
        except Exception:
            continue
    if event_caars:
        try:
            fig_bar = caar_bar_chart(event_caars, title=f"CAAR at t=+{target_day} by Event")
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.error("CAAR bar chart failed")
            logging.error(traceback.format_exc())
    else:
        st.info("No per-event CAAR data available.")

with col_right:
    st.subheader("Per-Event CAAR Table")
    if event_caars:
        table_df = pd.DataFrame([
            {"Event": label, "Date": next((e["date"] for e in selected_events if e["label"] == label), ""),
             f"CAAR (t=+{target_day})": val}
            for label, val in event_caars.items()
        ])
        # Add approximate t-stat (CAAR / std across tickers? we don't have std per event.
        # Simpler: just show the CAAR.
        st.dataframe(table_df, use_container_width=True)
    else:
        st.info("No per-event CAAR data.")

# Row 4: Pre/post realised vol comparison
st.subheader("Volatility: Event Window vs Estimation Window")
# For each ticker, compute mean 21-day realised vol during event window and during estimation window
# We'll use the same event windows for all events? Average across selected events.
if valid_tickers and selected_events:
    vol_data = []
    for ticker in valid_tickers:
        rets = ticker_returns[ticker]
        # Compute rolling 21-day realized vol (annualized)
        rv = realized_vol(rets.to_frame(name=ticker), window=21, annualize=True)
        rv = rv[ticker + "_RVol"].dropna()
        if rv.empty:
            continue
        # For each event, get vol during event window and estimation window
        event_vols = []
        est_vols = []
        for event in selected_events:
            # Find event position in rets index
            event_ts = pd.Timestamp(event["date"])
            idx = rets.index
            if event_ts not in idx:
                tolerance = pd.Timedelta(days=3)
                candidates = idx[(idx >= event_ts - tolerance) & (idx <= event_ts + tolerance)]
                if len(candidates) == 0:
                    continue
                event_pos = idx.get_loc(candidates[0])
            else:
                event_pos = idx.get_loc(event_ts)
            # Event window indices
            event_start = max(0, event_pos + event_window[0])
            event_end = min(len(idx)-1, event_pos + event_window[1])
            if event_end <= event_start:
                continue
            event_vol = rv.iloc[event_start:event_end+1].mean()
            if not np.isnan(event_vol):
                event_vols.append(event_vol)
            # Estimation window (ESTIMATION_WINDOW, e.g., -120 to -20)
            est_start = event_pos + ESTIMATION_WINDOW[0]
            est_end = event_pos + ESTIMATION_WINDOW[1]
            if est_start < 0 or est_end >= len(idx) or est_end <= est_start:
                continue
            est_vol = rv.iloc[est_start:est_end+1].mean()
            if not np.isnan(est_vol):
                est_vols.append(est_vol)
        if event_vols and est_vols:
            vol_data.append({
                "Ticker": ticker,
                "Event Window Vol": np.mean(event_vols),
                "Estimation Window Vol": np.mean(est_vols),
            })
    if vol_data:
        vol_df = pd.DataFrame(vol_data)
        # Bar chart with two bars per ticker
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=vol_df["Ticker"],
            y=vol_df["Event Window Vol"],
            name="Event Window (mean)",
            marker_color="#E74C3C"
        ))
        fig_vol.add_trace(go.Bar(
            x=vol_df["Ticker"],
            y=vol_df["Estimation Window Vol"],
            name="Estimation Window (mean)",
            marker_color="#3498DB"
        ))
        fig_vol.update_layout(
            title="Realised Volatility Comparison (Annualized)",
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial", size=12, color="#E0E0E0"),
            margin=dict(l=40, r=40, t=60, b=40),
            yaxis_tickformat=".1%",
        )
        fig_vol.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
        fig_vol.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("Insufficient data for volatility comparison.")
else:
    st.info("Select tickers and events to see volatility comparison.")