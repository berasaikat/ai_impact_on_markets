# components/car_chart.py
"""
Cumulative abnormal return (CAR) chart with confidence intervals
and bar chart for CAAR comparisons.
"""

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go

# Muted color palette for individual CAR lines
_INDIVIDUAL_COLORS = [
    "#A6CEE3",
    "#B2DF8A",
    "#FB9A99",
    "#FDBF6F",
    "#CAB2D6",
    "#FFFF99",
    "#B15928",
]


def car_chart(
    car_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    event_label: str = "",
    show_individual: bool = False,
    individual_cars: Optional[List[pd.Series]] = None,
) -> go.Figure:
    """
    Render cumulative abnormal return line chart with 95% confidence interval shading.

    Parameters
    ----------
    car_df : pd.DataFrame
        DataFrame with columns ['day_offset', 'CAAR'].
    ci_df : pd.DataFrame
        DataFrame with columns ['day_offset', 'CI_lower', 'CI_upper'].
    event_label : str, default=''
        Label to display in the chart title.
    show_individual : bool, default=False
        Whether to plot individual CAR series in the background.
    individual_cars : list[pd.Series] | None, default=None
        List of pd.Series indexed by day_offset with CAR values for individual events.

    Returns
    -------
    go.Figure
        Plotly figure ready for st.plotly_chart.
    """
    # Merge and sort by day_offset
    car_sorted = car_df.sort_values("day_offset")
    ci_sorted = ci_df.sort_values("day_offset")

    # Ensure day_offset is integer
    car_sorted["day_offset"] = car_sorted["day_offset"].astype(int)
    ci_sorted["day_offset"] = ci_sorted["day_offset"].astype(int)

    day_offsets = car_sorted["day_offset"].values
    caar_values = car_sorted["CAAR"].values

    # Build figure
    fig = go.Figure()

    # 1. Confidence band (shaded area between lower and upper)
    ci_upper = ci_sorted.set_index("day_offset").reindex(day_offsets)["CI_upper"].values
    ci_lower = ci_sorted.set_index("day_offset").reindex(day_offsets)["CI_lower"].values

    fig.add_trace(
        go.Scatter(
            x=day_offsets,
            y=ci_upper,
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="CI Upper",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=day_offsets,
            y=ci_lower,
            fill="tonexty",
            fillcolor="rgba(55,138,221,0.15)",
            mode="lines",
            line=dict(width=0),
            name="95% Confidence Interval",
            showlegend=True,
        )
    )

    # 2. Main CAR line
    fig.add_trace(
        go.Scatter(
            x=day_offsets,
            y=caar_values,
            mode="lines+markers",
            line=dict(color="#378ADD", width=2.5),
            marker=dict(size=4, color="#378ADD"),
            name="CAAR",
        )
    )

    # 3. Individual CAR lines (if requested)
    if show_individual and individual_cars:
        for i, car_series in enumerate(individual_cars):
            if car_series is None or car_series.empty:
                continue
            # Ensure index is integer day offsets
            idx = car_series.index.astype(int)
            vals = car_series.values
            color = _INDIVIDUAL_COLORS[i % len(_INDIVIDUAL_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=vals,
                    mode="lines",
                    line=dict(color=color, width=1.2, dash="dot"),
                    opacity=0.4,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        # Add a dummy legend entry for "Individual events"
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="gray", width=1.2, dash="dot"),
                opacity=0.4,
                name=f"Individual CARs (n={len(individual_cars)})",
            )
        )

    # Day 0 vertical line
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=1.5)

    # Zero horizontal line
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)", line_width=1)

    # X-axis formatting with custom labels
    tick_vals = []
    tick_text = []
    for offset in day_offsets:
        # Show every 5 days +/-10, plus Event at 0
        if abs(offset) % 5 == 0 or offset == 0:
            tick_vals.append(offset)
            if offset == 0:
                tick_text.append("Event")
            elif offset > 0:
                tick_text.append(f"+{offset}")
            else:
                tick_text.append(str(offset))

    # Layout styling
    title_text = f"Cumulative Abnormal Return - {event_label}" if event_label else "Cumulative Abnormal Return"
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#E0E0E0"),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(
        title_text="Days Relative to Event",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Cumulative Abnormal Return",
        tickformat=".1%",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        zeroline=False,
    )

    return fig


def caar_bar_chart(event_cars: dict, title: str = "CAAR by Event") -> go.Figure:
    """
    Bar chart comparing CAAR at day +5 (or final day) across multiple events.

    Parameters
    ----------
    event_cars : dict[str, float]
        Dictionary mapping event label -> CAAR value (fractional, e.g., 0.05 = 5%).
    title : str, default='CAAR by Event'
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not event_cars:
        fig = go.Figure()
        fig.update_layout(title="No CAAR data available")
        return fig

    # Sort by value descending
    sorted_items = sorted(event_cars.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Define bar colors: positive green, negative red
    colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in values]

    fig = go.Figure(
        go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.2%}" for v in values], textposition="outside")
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color="#E0E0E0"),
        margin=dict(l=40, r=40, t=60, b=80),
        xaxis_title="Event",
        yaxis_title="CAAR (t=+5)",
        yaxis_tickformat=".1%",
        showlegend=False,
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)", tickangle=45)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")

    return fig