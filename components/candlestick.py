# components/candlestick.py
"""
Annotated candlestick chart with event markers and volume subplot.
"""

from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Category colors for event annotations
CATEGORY_COLOURS = {
    "model_release": "#4B9CD3",
    "earnings": "#F5A623",
    "partnership": "#7ED321",
    "regulatory": "#D0021B",
    "hardware": "#9B59B6",
}


def annotated_candlestick(
    ohlcv_df: pd.DataFrame,
    events: Optional[List[dict]] = None,
    ticker: str = "",
    add_volume: bool = True,
    highlight_window: Optional[Tuple[str, str]] = None,
) -> go.Figure:
    """
    Build an annotated candlestick chart with vertical event lines and volume subplot.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume.
        Must have a DatetimeIndex.
    events : list[dict] | None, default=None
        List of event dicts with keys: date (str YYYY-MM-DD), label (str), category (str).
    ticker : str, default=''
        Ticker symbol for chart title.
    add_volume : bool, default=True
        Whether to include a volume subplot below the candlesticks.
    highlight_window : tuple[str, str] | None, default=None
        (start_date, end_date) to shade as a vertical rectangle.

    Returns
    -------
    go.Figure
        Plotly figure ready for st.plotly_chart.
    """
    # Prepare data
    df = ohlcv_df.copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{ticker} - No Data Available")
        return fig

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Define color for volume bars: green if close >= open, else red
    volume_color = [
        "#2ECC71" if close >= open_ else "#E74C3C"
        for close, open_ in zip(df["Close"], df["Open"])
    ]

    # Create subplots
    if add_volume:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.75, 0.25],
        )
        # Candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )
        # Volume trace
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=volume_color,
                opacity=0.6,
            ),
            row=2,
            col=1,
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            )
        )

    # Title
    title_text = f"{ticker} - Price & Volume" if add_volume else f"{ticker} - Price"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#E0E0E0"),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
    )

    # Gridlines
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")

    # Highlight window (vertical rectangle)
    if highlight_window is not None:
        start, end = highlight_window
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(255,255,100,0.07)",
            line_width=0,
            layer="below",
        )

    # Add event lines
    if events:
        for event in events:
            date = event.get("date")
            if date is None:
                continue
            
            # Convert date to unix timestamp in milliseconds for plotly datetime axis
            date_ms = pd.to_datetime(date).timestamp() * 1000
            
            label = event.get("label", "")
            category = event.get("category", "default")
            color = CATEGORY_COLOURS.get(category, "#888888")

            fig.add_vline(
                x=date_ms,
                line_dash="dot",
                line_color=color,
                annotation_text=label,
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color=color,
            )

    return fig