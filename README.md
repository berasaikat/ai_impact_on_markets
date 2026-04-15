# AI Impact on Markets Dashboard

A professional Streamlit dashboard analyzing the impact of AI announcements and model releases on equity prices and derivatives markets.

## Features

- **Market Overview** – Annotated candlestick charts with AI event markers
- **Event Study** – Cumulative Abnormal Returns (CAR) with confidence intervals
- **Derivatives Surface** – 3D implied volatility surfaces, put-call ratios, and vol spreads
- **Sector Heatmap** – Correlation matrices and rolling beta analysis
- **Sentiment Overlay** – News sentiment scoring with lead-lag correlation
- **Portfolio Simulator** – Interactive backtesting with risk metrics

## Tech Stack

- Python 3.11+
- Streamlit 1.35+
- Plotly for interactive visualizations
- yfinance for market data
- statsmodels for event study methodology
- VADER for sentiment analysis

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/ai_market_dashboard.git
cd ai_market_dashboard

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys to .env (NewsAPI, FRED)

# Run the dashboard
streamlit run app.py
```

## Required API Keys (Free Tier)

| Service | Purpose | Sign-up |
|---------|---------|---------|
| NewsAPI | News headlines & sentiment | [newsapi.org](https://newsapi.org) |
| FRED | VIX, Fed Funds, CPI data | [fred.stlouisfed.org](https://fred.stlouisfed.org) |

