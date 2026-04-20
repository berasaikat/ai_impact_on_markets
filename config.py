"""
config.py - Configuration constants for AI Market Dashboard

All hardcoded values, ticker lists, event dates, and API keys live here.
Import from this module across all other files.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# TICKER LISTS
# ============================================================================

# Core AI companies (primary focus)
AI_TICKERS = [
    'NVDA',   # NVIDIA
    'MSFT',   # Microsoft
    'GOOGL',  # Google/Alphabet
    'META',   # Meta Platforms
    'AMD',    # AMD
    'AMZN',   # Amazon
    'TSLA',   # Tesla
    'PLTR',   # Palantir
    'SMCI',   # Super Micro Computer
    'ARM',    # Arm Holdings
    'INTC',   # Intel
    'QCOM'    # Qualcomm
]

# Adjacent AI companies (semiconductors, infrastructure)
AI_ADJACENT_TICKERS = [
    'AVGO',   # Broadcom
    'ASML',   # ASML Holding
    'TSM',    # Taiwan Semiconductor
    'AMAT',   # Applied Materials
    'KLAC',   # KLA Corporation
    'LRCX',   # Lam Research
    'MU',     # Micron Technology
    'WDC',    # Western Digital
    'QCOM',   # Qualcomm Inc
]

# Combined universe for sector heatmap
ALL_TICKERS = AI_TICKERS + AI_ADJACENT_TICKERS

# Benchmark tickers
BENCHMARK_TICKER: str = 'SPY'      # S&P 500 ETF (market proxy for OLS regressions)
AI_INDEX_TICKER: str = 'QQQ'       # Nasdaq 100 ETF (AI sector benchmark)

# ============================================================================
# AI EVENT DATES
# ============================================================================
# Format: {'label': str, 'date': 'YYYY-MM-DD', 'category': str}
# Categories: 'model_release', 'earnings', 'partnership', 'regulatory'

AI_EVENTS = [
    # 2022 - ChatGPT Launch Era
    {'label': 'ChatGPT Launch', 'date': '2022-11-30', 'category': 'model_release'},
    {'label': 'GPT-4 Release', 'date': '2023-03-14', 'category': 'model_release'},
    {'label': 'Bing AI Launch', 'date': '2023-02-07', 'category': 'model_release'},
    {'label': 'Bard/Gemini Launch', 'date': '2023-03-21', 'category': 'model_release'},
    {'label': 'Meta Llama Release', 'date': '2023-02-24', 'category': 'model_release'},
    {'label': 'Midjourney V5', 'date': '2023-03-15', 'category': 'model_release'},
    {'label': 'Adobe Firefly', 'date': '2023-03-21', 'category': 'model_release'},
    
    # 2023 - Major Model Updates
    {'label': 'GPT-4 Turbo', 'date': '2023-11-06', 'category': 'model_release'},
    {'label': 'Elon Musk xAI Launch', 'date': '2023-07-12', 'category': 'model_release'},
    {'label': 'Claude 2 Release', 'date': '2023-07-11', 'category': 'model_release'},
    {'label': 'Meta Llama 2', 'date': '2023-07-18', 'category': 'model_release'},
    {'label': 'Google Gemini Pro', 'date': '2023-12-06', 'category': 'model_release'},
    {'label': 'Mistral 7B Release', 'date': '2023-09-27', 'category': 'model_release'},
    
    # 2024 - Continued Innovation
    {'label': 'Sora Launch', 'date': '2024-02-15', 'category': 'model_release'},
    {'label': 'Claude 3 Release', 'date': '2024-03-04', 'category': 'model_release'},
    {'label': 'Grok-1.5', 'date': '2024-03-28', 'category': 'model_release'},
    {'label': 'Meta Llama 3', 'date': '2024-04-18', 'category': 'model_release'},
    {'label': 'GPT-4o', 'date': '2024-05-13', 'category': 'model_release'},
    {'label': 'Apple AI Announcement', 'date': '2024-06-10', 'category': 'model_release'},
    
    # Key Earnings Events
    {'label': 'NVDA Earnings Q1 2023', 'date': '2023-05-24', 'category': 'earnings'},
    {'label': 'NVDA Earnings Q2 2023', 'date': '2023-08-23', 'category': 'earnings'},
    {'label': 'NVDA Earnings Q3 2023', 'date': '2023-11-21', 'category': 'earnings'},
    {'label': 'NVDA Earnings Q4 2023', 'date': '2024-02-21', 'category': 'earnings'},
    {'label': 'MSFT Earnings (AI focus)', 'date': '2023-04-25', 'category': 'earnings'},
    {'label': 'MSFT Earnings (Copilot)', 'date': '2023-10-24', 'category': 'earnings'},
    {'label': 'GOOGL Earnings (Gemini)', 'date': '2023-10-24', 'category': 'earnings'},
    
    # Regulatory Events
    {'label': 'EU AI Act Draft', 'date': '2023-06-14', 'category': 'regulatory'},
    {'label': 'Biden AI Executive Order', 'date': '2023-10-30', 'category': 'regulatory'},
    {'label': 'EU AI Act Final', 'date': '2024-03-13', 'category': 'regulatory'},
    
    # Partnerships & Acquisitions
    {'label': 'Microsoft-OpenAI Deepening', 'date': '2023-01-23', 'category': 'partnership'},
    {'label': 'Google-Anthropic Investment', 'date': '2023-10-27', 'category': 'partnership'},
    {'label': 'AWS-Anthropic Partnership', 'date': '2023-09-25', 'category': 'partnership'},
    {'label': 'Microsoft-Inflection Deal', 'date': '2024-03-19', 'category': 'partnership'},
]

# ============================================================================
# FRED ECONOMIC SERIES
# ============================================================================

FRED_SERIES = {
    'vix': 'VIXCLS',           # CBOE Volatility Index (daily)
    'fed_funds': 'FEDFUNDS',   # Effective Federal Funds Rate (daily)
    'cpi': 'CPIAUCSL',         # Consumer Price Index for All Urban Consumers (monthly)
    'treasury_10y': 'DGS10',   # 10-Year Treasury Constant Maturity Rate (daily)
    'sp500_pe': 'SP500',       # S&P 500 P/E Ratio (monthly, requires separate calc)
}

# ============================================================================
# EVENT STUDY PARAMETERS
# ============================================================================

ESTIMATION_WINDOW: tuple[int, int] = (-120, -20)   # Trading days before event for OLS estimation
EVENT_WINDOW: tuple[int, int] = (-10, 10)           # Trading days around event for CAR computation
MIN_ESTIMATION_DAYS: int = 60

# Confidence level for CAR bands (95% = 1.96, 90% = 1.645)
CAR_CONFIDENCE_LEVEL = 0.95

# ============================================================================
# DATA CACHING
# ============================================================================

CACHE_TTL_SECONDS: int = 3600          # Cache time-to-live in seconds (1 hour)   
MAX_RETRIES: int = 3                   # Max retries for API calls
RETRY_BACKOFF_FACTOR: int = 2          # Exponential backoff multiplier
PRICE_INTERVAL: str = '1d'

# ============================================================================
# Volatility thresholds
# ============================================================================

REALIZED_VOL_WINDOW: int = 21
ROLLING_BETA_WINDOW: int = 63
VIX_LOW_THRESHOLD: float = 15.0
VIX_HIGH_THRESHOLD: float = 25.0

# ============================================================================
# API KEYS (loaded from .env file)
# ============================================================================

NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
FRED_API_KEY = os.getenv('FRED_API_KEY', '')

# Validate API keys (warn but don't crash)
if not NEWS_API_KEY:
    print("Warning: NEWS_API_KEY not found in .env file. News features will be limited.")
if not FRED_API_KEY:
    print("Warning: FRED_API_KEY not found in .env file. Macro data features will be limited.")

# ============================================================================
# SENTIMENT ANALYSIS PARAMETERS
# ============================================================================

SENTIMENT_THRESHOLD = 0.3         # Threshold for detecting significant sentiment events
SENTIMENT_LAG_RANGE = (-5, 5)     # Days to analyze lead-lag correlation

# Keywords for AI event detection in news
AI_KEYWORDS = [
    'artificial intelligence', 'ai', 'machine learning', 'deep learning',
    'llm', 'language model', 'gpt', 'chatgpt', 'bard', 'gemini', 'claude',
    'llama', 'mistral', 'generative ai', 'foundation model', 'neural network',
    'transformer', 'openai', 'anthropic', 'inflection', 'xai'
]

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Color schemes
COLOR_CATEGORIES = {
    'model_release': '#2E86AB',   # Blue
    'earnings': '#A23B72',         # Purple
    'partnership': '#F18F01',      # Orange
    'regulatory': '#C73E1D',       # Red
}

# Default date range (last 2 years)
DEFAULT_START_DATE = '2022-01-01'
DEFAULT_END_DATE = '2025-01-01'

# Chart defaults
CHART_HEIGHT = 600
CHART_WIDTH = None  # Auto-width with container

# ============================================================================
# PORTFOLIO SIMULATOR DEFAULTS
# ============================================================================

DEFAULT_INITIAL_CAPITAL = 100000.0  # $100,000
DEFAULT_REBALANCE_FREQUENCY = 'Monthly'  # Options: 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Annually', 'Never'

# Risk-free rate for Sharpe ratio (current ~4.5% annualized)
RISK_FREE_RATE = 0.045

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# def get_tickers_by_category(category: str) -> list:
#     """Return tickers by category."""
#     categories = {
#         'ai_pure': AI_TICKERS,
#         'ai_adjacent': AI_ADJACENT_TICKERS,
#         'all': ALL_TICKERS,
#         'benchmark': [BENCHMARK_TICKER],
#         'ai_index': [AI_INDEX_TICKER]
#     }
#     return categories.get(category, [])

# def get_events_by_category(category: str = None) -> list:
#     """Return events filtered by category."""
#     if category is None:
#         return AI_EVENTS
#     return [e for e in AI_EVENTS if e.get('category') == category]

# def get_event_dates() -> list:
#     """Return just the dates of all AI events."""
#     return [e['date'] for e in AI_EVENTS]