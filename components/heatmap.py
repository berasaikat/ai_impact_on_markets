# components/heatmap.py
"""
Correlation heatmap and sector return heatmap using Plotly.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Default period list for sector return heatmap
_DEFAULT_PERIODS = ["1W", "1M", "3M", "6M", "YTD", "1Y"]


def correlation_heatmap(
    returns_df: pd.DataFrame, title: str = "Correlation Matrix"
) -> go.Figure:
    """
    Render a Plotly correlation heatmap of log returns across tickers.

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of log returns with DatetimeIndex and tickers as columns.
    title : str, default='Correlation Matrix'
        Chart title.

    Returns
    -------
    go.Figure
        Heatmap figure. Square aspect ratio if ≤15 tickers.
    """
    # Compute correlation matrix
    corr_matrix = returns_df.corr()

    # Mask upper triangle (k=1 excludes diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    masked_corr = corr_matrix.where(~mask, np.nan)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=masked_corr.values,
            x=masked_corr.columns.tolist(),
            y=masked_corr.index.tolist(),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=masked_corr.round(2).values,
            texttemplate="%{text}",
            textfont_size=10,
            textfont_color="black",
            showscale=True,
            colorbar=dict(title="Correlation", tickformat=".2f"),
            hoverongaps=False,
        )
    )

    # Layout with global style
    n_tickers = len(returns_df.columns)
    layout_width = 700 if n_tickers <= 15 else None
    layout_height = 700 if n_tickers <= 15 else None

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#E0E0E0"),
        margin=dict(l=40, r=40, t=60, b=40),
        width=layout_width,
        height=layout_height,
        xaxis=dict(
            title="Ticker",
            tickangle=45,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
        ),
        yaxis=dict(
            title="Ticker",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
            scaleanchor="x" if n_tickers <= 15 else None,
        ),
    )

    return fig


def sector_return_heatmap(
    returns_df: pd.DataFrame,
    sector_map: Dict[str, str],
    periods: Optional[List[str]] = None,
) -> go.Figure:
    """
    Render a heatmap of period returns per ticker, grouped by sector.

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of log returns with DatetimeIndex and tickers as columns.
    sector_map : dict[str, str]
        Mapping from ticker to sector name.
    periods : list[str] | None, default=None
        List of periods to compute (e.g., '1W', '1M', '3M', '6M', 'YTD', '1Y').
        If None, uses default periods.

    Returns
    -------
    go.Figure
        Heatmap figure with sector divider lines.
    """
    if periods is None:
        periods = _DEFAULT_PERIODS

    # Convert log returns to simple returns for cumulative product calculation
    # (log returns sum to log total return, then expm1)
    # We'll compute total return over each period ending at the last date in returns_df
    end_date = returns_df.index.max()

    # Helper to compute total return over a period
    def period_total_return(period: str) -> pd.Series:
        if period == "YTD":
            start_date = pd.Timestamp(end_date.year, 1, 1)
        else:
            # Parse period like '1W', '1M', '3M', '6M', '1Y'
            offset = period[-1]
            try:
                num = int(period[:-1])
            except ValueError:
                num = 1
            if offset == "W":
                start_date = end_date - pd.Timedelta(weeks=num)
            elif offset == "M":
                start_date = end_date - pd.DateOffset(months=num)
            elif offset == "Y":
                start_date = end_date - pd.DateOffset(years=num)
            else:
                start_date = returns_df.index.min()

        # Slice returns between start_date and end_date
        mask = (returns_df.index >= start_date) & (returns_df.index <= end_date)
        period_returns = returns_df[mask]

        if period_returns.empty:
            return pd.Series(index=returns_df.columns, dtype=float).fillna(np.nan)

        # Sum log returns over the period (log total return)
        log_total = period_returns.sum()
        # Convert to simple total return
        total_return = np.exp(log_total) - 1
        return total_return

    # Compute total returns for each period
    period_returns_dict = {}
    for p in periods:
        period_returns_dict[p] = period_total_return(p)

    # Build DataFrame: rows = tickers, columns = periods
    result_df = pd.DataFrame(period_returns_dict)

    # Add sector information and sort
    result_df["sector"] = result_df.index.map(lambda t: sector_map.get(t, "Unknown"))
    # Sort by sector, then by ticker name
    result_df = result_df.sort_values(["sector", result_df.index.name or "ticker"])
    sectors = result_df["sector"]
    result_df = result_df.drop(columns=["sector"])

    # Prepare z matrix (values) and row labels
    tickers = result_df.index.tolist()
    z = result_df.values

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=periods,
            y=tickers,
            colorscale="RdYlGn",
            zmin=-0.5,
            zmax=0.5,
            text=np.round(z * 100, 1),  # show as percentage
            texttemplate="%{text}%",
            textfont_size=10,
            showscale=True,
            colorbar=dict(title="Return", tickformat=".0%"),
            hovertemplate="Ticker: %{y}<br>Period: %{x}<br>Return: %{z:.1%}<extra></extra>",
        )
    )

    # Add sector divider lines (horizontal lines between groups)
    # Find indices where sector changes
    unique_sectors = []
    current = None
    sector_boundaries = []
    for i, (ticker, sector) in enumerate(zip(tickers, sectors)):
        if sector != current:
            if current is not None:
                # Boundary after previous sector
                sector_boundaries.append(i - 0.5)
            current = sector
    # Add the last boundary if needed? Actually we want lines between groups.
    # We'll add horizontal lines at y = boundary value (between rows)
    # Plotly heatmap y-axis is categorical with tickers in order, but coordinates are 0-indexed.
    # So a line at y = index - 0.5 separates row index-1 and index.
    for boundary in sector_boundaries:
        fig.add_hline(
            y=boundary,
            line_dash="dash",
            line_color="rgba(255,255,255,0.5)",
            line_width=1.5,
            annotation_text="",
        )

    # Layout with global style
    fig.update_layout(
        title=dict(text="Sector Return Heatmap", font=dict(size=15)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#E0E0E0"),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            title="Period",
            tickangle=45,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
        ),
        yaxis=dict(
            title="Ticker (grouped by sector)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
            autorange="reversed",  # so first sector is at top
        ),
        height=400 + 20 * len(tickers),  # dynamic height
    )

    return fig