# pages/3_derivatives.py
"""
Derivatives Surface page: Implied volatility surface, volatility smile,
volatility spread, and put-call ratios.
"""

import logging
import traceback
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from components.iv_surface import iv_smile, iv_surface_3d
from config import REALIZED_VOL_WINDOW
from data import (
    build_iv_matrix,
    fetch_all_expiries,
    fetch_ohlcv,
    fetch_options_chain,
    fetch_vix,
    get_put_call_ratio,
)
from features.volatility import realized_vol, vol_spread
from utils.formatting import fmt_pct, fmt_ratio

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(layout="wide")

# ── Read global session state ────────────────────────────────────────────────
start = st.session_state.get("start_date", pd.Timestamp("2022-01-01").date())
end = st.session_state.get("end_date", pd.Timestamp.today().date())
basket = st.session_state.get("basket", "AI Pure-Play")  # not used here

# ── Page-local sidebar controls ──────────────────────────────────────────────
ticker_options = ["NVDA", "MSFT", "GOOGL", "META", "AMD"]
selected_ticker = st.sidebar.selectbox("Ticker", options=ticker_options, index=0)

view_mode = st.sidebar.radio("View", ["3D Surface", "IV Smile", "Both"])
max_expiries = st.sidebar.slider("Max expiries", min_value=3, max_value=10, value=6)
show_vol_spread = st.sidebar.checkbox("Show volatility spread and put-call ratio", value=True)

# ── Data fetching with caching ───────────────────────────────────────────────
@st.cache_data(ttl=1800)
def get_expiries(ticker):
    return fetch_all_expiries(ticker)

@st.cache_data(ttl=1800)
def get_options_chain(ticker, expiry):
    return fetch_options_chain(ticker, expiry)

@st.cache_data(ttl=1800)
def get_iv_matrix(ticker, max_exp):
    return build_iv_matrix(ticker, max_expiries=max_exp)

@st.cache_data(ttl=1800)
def get_price_data(ticker, start_date, end_date):
    return fetch_ohlcv([ticker], start_date, end_date).get(ticker, pd.DataFrame())

@st.cache_data(ttl=3600)
def get_vix_series(start_date, end_date):
    vix_df = fetch_vix(start_date, end_date)
    if vix_df is not None and not vix_df.empty:
        return vix_df["VIX"]
    return pd.Series(dtype=float)

@st.cache_data(ttl=1800)
def get_historical_atm_iv(ticker, max_days=60):
    """
    Build approximate ATM IV time series using current options chain.
    Limitation: uses current chain for all historical dates.
    """
    # Fetch historical prices
    price_df = fetch_ohlcv([ticker], pd.Timestamp.today() - timedelta(days=max_days+30), pd.Timestamp.today())
    if ticker not in price_df or price_df[ticker].empty:
        return pd.Series(dtype=float)
    prices = price_df[ticker][["Close"]].dropna()
    if len(prices) < 10:
        return pd.Series(dtype=float)
    # Get current front-month expiry
    expiries = fetch_all_expiries(ticker)
    if not expiries:
        return pd.Series(dtype=float)
    front_month = expiries[0]
    calls, puts = fetch_options_chain(ticker, front_month)
    if calls.empty and puts.empty:
        return pd.Series(dtype=float)
    # Combine calls and puts, keep strikes and IV
    combined = pd.concat([calls[["strike", "impliedVolatility"]], puts[["strike", "impliedVolatility"]]])
    combined = combined.dropna().groupby("strike").mean()  # average if both exist
    atm_iv_series = []
    for date, close in prices["Close"].items():
        # Find strike closest to close price
        if combined.empty:
            iv = np.nan
        else:
            idx = np.abs(combined.index - close).argmin()
            iv = combined.iloc[idx]["impliedVolatility"]
        atm_iv_series.append({"date": date, "atm_iv": iv})
    result = pd.DataFrame(atm_iv_series).set_index("date")["atm_iv"]
    return result

@st.cache_data(ttl=1800)
def get_put_call_ratio_by_expiry(ticker, expiries):
    """Return dict expiry -> put-call ratio (volume) for multiple expiries."""
    ratios = {}
    for exp in expiries[:10]:  # limit
        try:
            ratio = get_put_call_ratio(ticker, exp)
            if not np.isnan(ratio):
                ratios[exp] = ratio
        except Exception:
            continue
    return ratios

# ── Main data loading ────────────────────────────────────────────────────────
with st.spinner("Loading derivatives data..."):
    # Expiries and IV matrix
    expiries = get_expiries(selected_ticker)
    if not expiries:
        st.error(f"No option expiries found for {selected_ticker}")
        st.stop()
    front_month = expiries[0]
    iv_matrix = get_iv_matrix(selected_ticker, max_expiries)
    # Options chain for front-month
    calls_df, puts_df = get_options_chain(selected_ticker, front_month)
    # ATM IV from front-month chain
    if not calls_df.empty or not puts_df.empty:
        # Combine calls and puts to find ATM strike
        combined = pd.concat([calls_df[["strike", "impliedVolatility"]],
                              puts_df[["strike", "impliedVolatility"]]])
        combined = combined.dropna().groupby("strike").mean()
        # Current spot price
        # ticker_obj = yf.Ticker(selected_ticker)
        # spot = ticker_obj.history(period="1d")["Close"].iloc[-1]
        # idx = (combined.index - spot).abs().argmin()
        # atm_iv = combined.iloc[idx]["impliedVolatility"]
        # atm_iv_pct = atm_iv
        # Current spot price
        ticker_obj = yf.Ticker(selected_ticker)
        spot = ticker_obj.history(period="1d")["Close"].iloc[-1]
        # Find strike closest to spot price
        strike_diff = np.abs(combined.index - spot)
        idx = strike_diff.argmin()
        atm_iv = combined.iloc[idx]["impliedVolatility"]
        atm_iv_pct = atm_iv
    else:
        atm_iv_pct = np.nan
    # Put-call ratio for front-month
    pcr = get_put_call_ratio(selected_ticker, front_month)
    # Realized volatility (21-day)
    price_df = get_price_data(selected_ticker, start, end)
    if not price_df.empty:
        returns = np.log(price_df["Close"] / price_df["Close"].shift(1)).dropna()
        realized = realized_vol(returns.to_frame(name=selected_ticker), window=REALIZED_VOL_WINDOW, annualize=True)
        realized_latest = realized.iloc[-1, 0] if not realized.empty else np.nan
    else:
        realized_latest = np.nan
    # Vol spread
    vol_spread_val = atm_iv_pct - realized_latest if not np.isnan(atm_iv_pct) and not np.isnan(realized_latest) else np.nan

    # Historical ATM IV series (approximate)
    atm_iv_series = get_historical_atm_iv(selected_ticker, max_days=60)
    # Realized vol series for same period
    if not price_df.empty and len(price_df) > REALIZED_VOL_WINDOW:
        rv_series = realized_vol(returns.to_frame(name=selected_ticker), window=REALIZED_VOL_WINDOW, annualize=True)
        rv_series = rv_series.iloc[:, 0]  # Series
    else:
        rv_series = pd.Series(dtype=float)
    # Put-call ratio by expiry (for bar chart)
    pcr_by_expiry = get_put_call_ratio_by_expiry(selected_ticker, expiries)

# ── Layout ────────────────────────────────────────────────────────────────────
st.title(f"📊 Derivatives Analytics – {selected_ticker}")

# Row 1: Metric cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("ATM IV (Front-month)", fmt_pct(atm_iv_pct) if not np.isnan(atm_iv_pct) else "---")
with col2:
    st.metric("Put-Call Ratio (Vol)", f"{pcr:.2f}" if not np.isnan(pcr) else "---")
with col3:
    st.metric(f"21d Realized Vol", fmt_pct(realized_latest) if not np.isnan(realized_latest) else "---")
with col4:
    if not np.isnan(vol_spread_val):
        color = "red" if vol_spread_val > 0.05 else "green" if vol_spread_val < -0.05 else "gray"
        st.metric("Vol Spread (IV - RVol)", fmt_pct(vol_spread_val), delta_color="off")
        # Custom HTML color via markdown? Streamlit doesn't support color directly, but we can use st.markdown with custom CSS.
        # Alternatively just show the value.
        # We'll add a small note:
        if vol_spread_val > 0.05:
            st.caption("⚠️ Options overpriced")
        elif vol_spread_val < -0.05:
            st.caption("✅ Options underpriced")
    else:
        st.metric("Vol Spread (IV - RVol)", "---")

# Row 2: IV Surface / Smile based on view mode
if view_mode in ["3D Surface", "Both"]:
    st.subheader("Implied Volatility Surface")
    if iv_matrix is not None and not iv_matrix.empty:
        try:
            fig_surface = iv_surface_3d(iv_matrix, ticker=selected_ticker)
            st.plotly_chart(fig_surface, use_container_width=True)
        except Exception as e:
            st.error("Failed to render 3D surface")
            logging.error(traceback.format_exc())
    else:
        st.info("IV matrix not available for this ticker/expiries.")

if view_mode in ["IV Smile", "Both"]:
    st.subheader(f"Volatility Smile – {front_month}")
    if iv_matrix is not None and front_month in iv_matrix.columns:
        try:
            fig_smile = iv_smile(iv_matrix, expiry=front_month, ticker=selected_ticker)
            st.plotly_chart(fig_smile, use_container_width=True)
        except Exception as e:
            st.error("Failed to render IV smile")
            logging.error(traceback.format_exc())
    else:
        st.info(f"No data for expiry {front_month}")

# Row 3: Vol spread and put-call ratio time series (if toggled)
if show_vol_spread:
    st.subheader("Volatility Spread & Put-Call Ratio")
    col_left, col_right = st.columns(2)
    with col_left:
        # Combine ATM IV series and realized vol series
        if not atm_iv_series.empty and not rv_series.empty:
            # Align indices
            combined_vol = pd.DataFrame({"ATM IV": atm_iv_series, "Realized Vol": rv_series}).dropna()
            if not combined_vol.empty:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=combined_vol.index, y=combined_vol["ATM IV"], mode="lines",
                                             name="ATM IV (front-month)", line=dict(color="#4B9CD3")))
                fig_vol.add_trace(go.Scatter(x=combined_vol.index, y=combined_vol["Realized Vol"], mode="lines",
                                             name="Realized Vol (21d)", line=dict(color="#F5A623")))
                fig_vol.update_layout(
                    title="ATM IV vs Realized Volatility (last 60 days)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Arial", size=12, color="#E0E0E0"),
                    margin=dict(l=40, r=40, t=60, b=40),
                    yaxis_tickformat=".1%",
                    xaxis_title="Date",
                    yaxis_title="Volatility (Annualized)",
                )
                fig_vol.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
                fig_vol.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
                st.plotly_chart(fig_vol, use_container_width=True)
            else:
                st.info("Insufficient overlapping data for volatility time series.")
        else:
            st.info("Historical ATM IV or realized vol series not available (requires at least 21 trading days).")
    with col_right:
        # Put-call ratio bar chart by expiry
        if pcr_by_expiry:
            # Convert expiry strings to something readable (keep first 10)
            exp_list = list(pcr_by_expiry.keys())[:10]
            ratios = [pcr_by_expiry[e] for e in exp_list]
            # Truncate expiry labels to YYYY-MM-DD
            exp_labels = [e[:10] if len(e) > 10 else e for e in exp_list]
            fig_pcr = go.Figure(data=[
                go.Bar(x=exp_labels, y=ratios, marker_color="#7ED321")
            ])
            fig_pcr.update_layout(
                title="Put-Call Ratio by Expiry (Volume)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Arial", size=12, color="#E0E0E0"),
                margin=dict(l=40, r=40, t=60, b=80),
                xaxis_title="Expiry Date",
                yaxis_title="Put-Call Ratio",
                xaxis_tickangle=45,
            )
            fig_pcr.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
            fig_pcr.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
            st.plotly_chart(fig_pcr, use_container_width=True)
        else:
            st.info("Put-call ratios not available for this ticker.")

    # Limitation note for ATM IV series
    st.info("📌 Note: Historical ATM IV uses the *current* options chain for all dates (yfinance limitation). Real intraday snapshots are not available in free tier.")

# Row 4: Raw options chain expander
with st.expander("Raw options chain data – front-month expiry"):
    if not calls_df.empty:
        st.subheader("Call Options")
        st.dataframe(calls_df, use_container_width=True)
    else:
        st.info("No call options data.")
    if not puts_df.empty:
        st.subheader("Put Options")
        st.dataframe(puts_df, use_container_width=True)
    else:
        st.info("No put options data.")