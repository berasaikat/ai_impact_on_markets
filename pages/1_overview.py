# pages/1_overview.py
"""
Market Overview page: price charts, performance metrics, and volatility regime.
"""

import logging
import traceback

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.candlestick import annotated_candlestick
from config import AI_TICKERS, BENCHMARK_TICKER
from data import (
    fetch_ohlcv,
    fetch_ticker_info,
    fetch_vix,
    get_ai_basket,
    get_ai_events,
    get_benchmark,
    tag_event_dates,
)
from features.returns import log_returns
from features.volatility import vol_regime
from utils.formatting import fmt_dollar, fmt_large, fmt_pct, fmt_ratio

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(layout="wide")

# ── Read global session state ────────────────────────────────────────────────
start = st.session_state.get("start_date", pd.Timestamp("2022-01-01").date())
end = st.session_state.get("end_date", pd.Timestamp.today().date())
basket = st.session_state.get("basket", "AI Pure-Play")

# ── Page-local sidebar controls ──────────────────────────────────────────────
all_tickers = get_ai_basket(basket)
selected_tickers = st.sidebar.multiselect(
    "Select tickers", options=all_tickers, default=all_tickers[:5]
)

chart_type = st.sidebar.radio("Chart type", ["Candlestick", "Line"])
show_events = st.sidebar.checkbox("Show AI events", value=True)

# ── Helper: fetch data with error handling ───────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_price_data(tickers, start_date, end_date):
    """Fetch OHLCV for tickers, handling empty results."""
    return fetch_ohlcv(tickers, start_date, end_date)


@st.cache_data(ttl=3600)
def fetch_benchmark_data(start_date, end_date):
    return get_benchmark(start_date, end_date)


@st.cache_data(ttl=3600)
def fetch_vix_data(start_date, end_date):
    return fetch_vix(start_date, end_date)


@st.cache_data(ttl=86400)
def fetch_events():
    return get_ai_events()


# ── Data loading ─────────────────────────────────────────────────────────────
with st.spinner("Loading market data..."):
    # Price data for selected tickers
    price_dict = fetch_price_data(selected_tickers, start, end)

    # Benchmark (SPY)
    benchmark_df = fetch_benchmark_data(start, end)

    # VIX data
    vix_df = fetch_vix_data(start, end)

    # Events
    events = fetch_events() if show_events else None

# Filter out tickers with empty data
valid_tickers = [t for t in selected_tickers if not price_dict.get(t, pd.DataFrame()).empty]
if not valid_tickers:
    st.error("No valid price data for selected tickers.")
    st.stop()

# For metrics, use the first valid ticker
primary_ticker = valid_tickers[0]
primary_df = price_dict[primary_ticker]

# Compute daily returns for primary ticker
primary_returns = log_returns(primary_df[["Close"]])["Close"].dropna()
if primary_returns.empty:
    st.warning(f"Insufficient price data for {primary_ticker} to compute metrics.")
    primary_returns = pd.Series(dtype=float)

# Metrics
if not primary_returns.empty:
    total_return = (primary_df["Close"].iloc[-1] / primary_df["Close"].iloc[0] - 1)
    cum_returns = (1 + primary_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns / rolling_max - 1)
    max_drawdown = drawdown.min()
    ann_vol = primary_returns.std() * (252 ** 0.5)
    sharpe = primary_returns.mean() / primary_returns.std() * (252 ** 0.5) if primary_returns.std() > 0 else 0.0
else:
    total_return = max_drawdown = ann_vol = sharpe = 0.0

# Row 1: Metric cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Return (selected period)", fmt_pct(total_return))
with col2:
    st.metric("Max Drawdown", fmt_pct(max_drawdown))
with col3:
    st.metric("Annualised Volatility", fmt_pct(ann_vol))
with col4:
    st.metric("Sharpe Ratio", fmt_ratio(sharpe))

# Row 2: Price chart (candlestick or line)
st.subheader(f"{primary_ticker} - Price Chart")
try:
    if chart_type == "Candlestick":
        fig = annotated_candlestick(
            primary_df,
            events=events if show_events else None,
            ticker=primary_ticker,
            add_volume=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:  # Line chart
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=primary_df.index, y=primary_df["Close"], mode="lines", name=primary_ticker))
        if show_events and events:
            for ev in events:
                fig_line.add_vline(x=ev["date"], line_dash="dot", line_color="orange",
                                   annotation_text=ev["label"], annotation_position="top left")
        fig_line.update_layout(
            title=f"{primary_ticker} - Close Price",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial", size=12, color="#E0E0E0"),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig_line.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
        fig_line.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
        st.plotly_chart(fig_line, use_container_width=True)
except Exception as e:
    st.error("Chart failed to render")
    logging.error(traceback.format_exc())

# Row 3: Cumulative return comparison (selected tickers + SPY)
st.subheader("Cumulative Return Comparison")
try:
    # Build returns for all valid selected tickers and benchmark
    cum_fig = go.Figure()
    # Benchmark (SPY) adjusted close
    if benchmark_df is not None and not benchmark_df.empty and "SPY_Close" in benchmark_df.columns:
        spy_close = benchmark_df["SPY_Close"].dropna()
        spy_returns = spy_close.pct_change().dropna()
        if not spy_returns.empty:
            spy_cum = (1 + spy_returns).cumprod()
            spy_cum_normalized = spy_cum / spy_cum.iloc[0] * 100
            cum_fig.add_trace(go.Scatter(x=spy_cum_normalized.index, y=spy_cum_normalized.values,
                                         mode="lines", name="SPY (Benchmark)", line=dict(dash="dash")))

    for ticker in valid_tickers:
        df = price_dict[ticker]
        if df.empty:
            continue
        close = df["Close"].dropna()
        if len(close) < 2:
            continue
        rets = close.pct_change().dropna()
        cum = (1 + rets).cumprod()
        cum_norm = cum / cum.iloc[0] * 100
        cum_fig.add_trace(go.Scatter(x=cum_norm.index, y=cum_norm.values, mode="lines", name=ticker))

    cum_fig.update_layout(
        title="Cumulative Return (Normalized to 100)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#E0E0E0"),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis_title="Date",
        yaxis_title="Index (Start = 100)",
    )
    cum_fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    cum_fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    st.plotly_chart(cum_fig, use_container_width=True)
except Exception as e:
    st.error("Cumulative return chart failed")
    logging.error(traceback.format_exc())

# Row 4: Two columns – ticker info table and volatility regime bar chart
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Ticker Information")
    info_rows = []
    for ticker in valid_tickers[:10]:  # limit to avoid clutter
        info = fetch_ticker_info(ticker)
        sector = info.get("sector", "N/A")
        beta = info.get("beta", "N/A")
        pe = info.get("trailingPE", "N/A")
        mcap = info.get("marketCap", None)
        mcap_fmt = fmt_large(mcap) if mcap else "N/A"
        info_rows.append({
            "Ticker": ticker,
            "Sector": sector,
            "Beta": beta if beta == "N/A" else round(beta, 2),
            "P/E": pe if pe == "N/A" else round(pe, 2),
            "Market Cap": mcap_fmt,
        })
    if info_rows:
        st.dataframe(pd.DataFrame(info_rows), use_container_width=True)
    else:
        st.info("No ticker info available")

with col_right:
    st.subheader("Volatility Regime (VIX)")
    if vix_df is not None and not vix_df.empty and "VIX" in vix_df.columns:
        vix_series = vix_df["VIX"].dropna()
        regime = vol_regime(vix_series)
        regime_counts = regime.value_counts()
        # Ensure all categories present
        all_cats = ["low", "normal", "high"]
        counts = {cat: regime_counts.get(cat, 0) for cat in all_cats}
        try:
            bar_fig = go.Figure(data=[
                go.Bar(x=list(counts.keys()), y=list(counts.values()),
                       marker_color=["#2ECC71", "#F1C40F", "#E74C3C"])
            ])
            bar_fig.update_layout(
                title="Number of Days by VIX Regime",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Arial", size=12, color="#E0E0E0"),
                margin=dict(l=40, r=40, t=60, b=40),
                xaxis_title="Regime",
                yaxis_title="Count",
            )
            bar_fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
            bar_fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
            st.plotly_chart(bar_fig, use_container_width=True)
        except Exception as e:
            st.error("Vol regime chart failed")
            logging.error(traceback.format_exc())
    else:
        st.info("VIX data not available")