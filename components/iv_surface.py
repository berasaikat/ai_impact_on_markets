# components/iv_surface.py
"""
3D implied volatility surface and 2D volatility smile charts.
"""

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf


def iv_surface_3d(
    iv_surface_df: pd.DataFrame, ticker: str = "", colorscale: str = "RdYlGn"
) -> go.Figure:
    """
    Render a 3D implied volatility surface with strike on X axis,
    DTE on Y axis, and IV on Z axis.

    Parameters
    ----------
    iv_surface_df : pd.DataFrame
        DataFrame with strike as index, expiry days-to-expiry as columns,
        and implied volatility as values (decimals, e.g., 0.45 = 45%).
    ticker : str, default=''
        Ticker symbol for chart title.
    colorscale : str, default='RdYlGn'
        Plotly colorscale name.

    Returns
    -------
    go.Figure
        Plotly 3D surface figure with height 600.
    """
    if iv_surface_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{ticker} IV Surface - No Data",
            height=600,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Prepare data: x = strikes, y = days to expiry (as ints), z = IV values transposed
    strikes = iv_surface_df.index.values
    # Convert column names (may be strings like '30') to integers
    dte = [int(c) for c in iv_surface_df.columns]
    z = iv_surface_df.values.T  # shape (len(dte), len(strikes))

    # Create 3D surface
    fig = go.Figure(
        data=[
            go.Surface(
                x=strikes,
                y=dte,
                z=z,
                colorscale=colorscale,
                opacity=0.9,
                colorbar=dict(title="Implied Volatility", tickformat=".0%"),
            )
        ]
    )

    # Add horizontal plane at mean IV for reference
    mean_iv = iv_surface_df.values.mean()
    # Create a mesh for the plane: same x and y range
    x_range = [strikes.min(), strikes.max()]
    y_range = [min(dte), max(dte)]
    # Plane data: z constant = mean_iv
    plane_z = [[mean_iv, mean_iv], [mean_iv, mean_iv]]
    fig.add_trace(
        go.Surface(
            x=x_range,
            y=y_range,
            z=plane_z,
            showscale=False,
            opacity=0.3,
            colorscale=[[0, "gray"], [1, "gray"]],
            name=f"Avg IV: {mean_iv:.1%}",
        )
    )

    # Camera angle
    camera = dict(eye=dict(x=1.6, y=1.6, z=0.8))

    # Layout with global style
    title_text = f"{ticker} Implied Volatility Surface" if ticker else "Implied Volatility Surface"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#E0E0E0"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=600,
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Volatility",
            zaxis_tickformat=".0%",
            camera=camera,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
            zaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
        ),
    )

    return fig


def iv_smile(iv_matrix_df: pd.DataFrame, expiry: str, ticker: str = "") -> go.Figure:
    """
    2D chart of IV vs strike for a single expiry (volatility smile/skew).

    Parameters
    ----------
    iv_matrix_df : pd.DataFrame
        DataFrame with strike as index, expiry dates as columns,
        implied volatility as values (decimals).
    expiry : str
        Column name (expiry date string) to slice.
    ticker : str, default=''
        Ticker symbol to fetch current spot price for ATM marker.
        If empty, ATM marker is omitted.

    Returns
    -------
    go.Figure
        Plotly line chart.
    """
    if iv_matrix_df.empty or expiry not in iv_matrix_df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=f"IV Smile - No Data for {expiry}",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Extract strikes and IV values, drop NaN
    series = iv_matrix_df[expiry].dropna()
    strikes = series.index.values
    iv_values = series.values

    # Create figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=iv_values,
            mode="lines+markers",
            line=dict(color="#4B9CD3", width=2.5),
            marker=dict(size=6, color="#4B9CD3"),
            name="Implied Volatility",
        )
    )

    # ATM marker if ticker provided
    if ticker:
        try:
            # Fetch current spot price
            spot_data = yf.Ticker(ticker).history(period="1d")
            if not spot_data.empty:
                spot = spot_data["Close"].iloc[-1]
                # Find strike closest to spot
                closest_idx = (strikes - spot).abs().argmin()
                closest_strike = strikes[closest_idx]
                atm_iv = iv_values[closest_idx]
                fig.add_trace(
                    go.Scatter(
                        x=[closest_strike],
                        y=[atm_iv],
                        mode="markers+text",
                        marker=dict(size=12, color="#F5A623", symbol="star"),
                        text=["ATM"],
                        textposition="top center",
                        textfont=dict(size=11, color="#F5A623"),
                        name=f"ATM (Spot: ${spot:.2f})",
                    )
                )
        except Exception:
            # Silently skip ATM marker if fetch fails
            pass

    # Layout with global style
    title_text = f"{ticker} Volatility Smile - {expiry}" if ticker else f"Volatility Smile - {expiry}"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#E0E0E0"),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False)

    return fig