"""
Smoke tests: Project Skeleton & Configuration

Run with: pytest tests/smoke_test_project_configuration.py -v

"""

# tests/smoke_test_project_configuration.py
import pytest
import subprocess
import sys
from pathlib import Path
import importlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestProjectConfiguration:
    """Smoke tests for Phase 1: Project Skeleton & Configuration"""

    def test_requirements_installation(self):
        """Test that all required packages can be imported"""
        required_packages = [
            'streamlit',
            'plotly',
            'yfinance',
            'pandas',
            'numpy',
            'statsmodels',
            'scipy',
            'vaderSentiment',
            'fredapi',
            'newsapi',
            'dotenv',
            'requests'
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError as e:
                pytest.fail(f"Failed to import {package}: {e}")

    def test_config_module(self):
        """Test config.py constants and structure"""
        from config import (
            AI_TICKERS,
            AI_ADJACENT_TICKERS,
            BENCHMARK_TICKER,
            AI_INDEX_TICKER,
            AI_EVENTS,
            FRED_SERIES,
            ESTIMATION_WINDOW,
            EVENT_WINDOW,
            MIN_ESTIMATION_DAYS,
            CACHE_TTL_SECONDS,
            DEFAULT_START_DATE,
            PRICE_INTERVAL,
            REALIZED_VOL_WINDOW,
            ROLLING_BETA_WINDOW,
            VIX_LOW_THRESHOLD,
            VIX_HIGH_THRESHOLD,
            FRED_SERIES,
            NEWS_API_KEY,
            FRED_API_KEY
        )
        
        # Test ticker baskets
        assert len(AI_TICKERS) >= 10, "AI_PURE_TICKERS should have at least 10 tickers"
        assert len(AI_ADJACENT_TICKERS) >= 9, "AI_ADJACENT_TICKERS should have at least 9 tickers"
        assert BENCHMARK_TICKER == 'SPY'
        assert AI_INDEX_TICKER == 'QQQ'
        
        # Test event structure
        assert len(AI_EVENTS) >= 20, "AI_EVENTS should have at least 20 events"
        for event in AI_EVENTS:
            assert 'label' in event, "Each event must have a 'label'"
            assert 'date' in event, "Each event must have a 'date'"
            assert 'category' in event, "Each event must have a 'category'"
            assert event['category'] in ['model_release', 'earnings', 'partnership', 'regulatory', 'hardware']
        
        # Test window parameters
        assert ESTIMATION_WINDOW[0] < ESTIMATION_WINDOW[1] < 0
        assert EVENT_WINDOW[0] < EVENT_WINDOW[1]
        assert MIN_ESTIMATION_DAYS > 0
        
        # Test data settings
        assert CACHE_TTL_SECONDS > 0
        assert DEFAULT_START_DATE == '2022-01-01'
        assert PRICE_INTERVAL == '1d'
        
        # Test thresholds
        assert REALIZED_VOL_WINDOW == 21
        assert ROLLING_BETA_WINDOW == 63
        assert VIX_LOW_THRESHOLD == 15.0
        assert VIX_HIGH_THRESHOLD == 25.0
        
        # Test FRED series
        assert 'vix' in FRED_SERIES
        assert 'fed_funds' in FRED_SERIES
        assert 'cpi' in FRED_SERIES
        assert FRED_SERIES['vix'] == 'VIXCLS'
        
        # Test API keys (should be loaded from .env, can be empty strings)
        assert isinstance(NEWS_API_KEY, str)
        assert isinstance(FRED_API_KEY, str)

    def test_cache_utils(self):
        """Test utils/cache.py exports"""
        from utils.cache import cached, clear_all_caches
        
        # Test cached decorator exists and is callable
        assert callable(cached)
        
        # Test that cached can be called with or without arguments
        @cached()
        def test_func():
            return "test"
        
        # Test clear_all_caches exists and is callable
        assert callable(clear_all_caches)
        
        # Test that cached returns a decorator
        decorated = cached(ttl=300)
        assert callable(decorated)

    def test_formatting_utils(self):
        """Test utils/formatting.py functions"""
        from utils.formatting import fmt_pct, fmt_dollar, fmt_large, fmt_date, fmt_ratio
        import pandas as pd
        import numpy as np
        
        # Test fmt_pct
        assert fmt_pct(0.0342) == '+3.42%'
        assert fmt_pct(0.0342, sign=False) == '3.42%'
        assert fmt_pct(-0.05) == '-5.00%'
        assert fmt_pct(0.05, decimals=1) == '+5.0%'
        assert fmt_pct(None) == '—'
        assert fmt_pct(np.nan) == '—'
        
        # Test fmt_dollar
        assert fmt_dollar(1234567.8) == '$1,234,567.80'
        assert fmt_dollar(1000) == '$1,000.00'
        assert fmt_dollar(-500) == '-$500.00'
        assert fmt_dollar(None) == '—'
        assert fmt_dollar(np.nan) == '—'
        
        # Test fmt_large
        assert fmt_large(1_500_000) == '$1.50M'
        assert fmt_large(2_500_000_000) == '$2.50B'
        assert fmt_large(750_000) == '$750.00K'
        assert fmt_large(500) == '$500.00'
        assert fmt_large(None) == '—'
        
        # Test fmt_date
        from datetime import datetime
        test_date = datetime(2023, 3, 14)
        assert fmt_date(test_date) == 'Mar 14, 2023'
        assert fmt_date('2023-03-14') == 'Mar 14, 2023'
        assert fmt_date(pd.Timestamp('2023-03-14')) == 'Mar 14, 2023'
        assert fmt_date(None) == '—'
        assert fmt_date(pd.NaT) == '—'
        
        # Test fmt_ratio
        assert fmt_ratio(1.435) == '1.44'
        assert fmt_ratio(1.435, decimals=3) == '1.435'
        assert fmt_ratio(None) == '—'
        assert fmt_ratio(np.nan) == '—'

    def test_packages_importable(self):
        """Test that data and features packages are importable"""
        try:
            import data
            import features
        except ImportError as e:
            pytest.fail(f"Failed to import data or features packages: {e}")
        
        # Test that __init__.py files exist
        data_init = Path(PROJECT_ROOT) / 'data' / '__init__.py'
        features_init = Path(PROJECT_ROOT) / 'features' / '__init__.py'
        
        assert data_init.exists(), "data/__init__.py not found"
        assert features_init.exists(), "features/__init__.py not found"

    def test_directory_structure(self):
        """Test that all required directories and files exist"""
        required_dirs = [
            'data',
            'features',
            'components',
            'pages',
            'utils'
        ]
        
        required_files = [
            'app.py',
            'config.py',
            'requirements.txt',
            '.env.example',
            'data/__init__.py',
            'features/__init__.py',
            'utils/cache.py',
            'utils/formatting.py'
        ]
        
        for dir_name in required_dirs:
            dir_path = Path(PROJECT_ROOT) / dir_name
            assert dir_path.exists(), f"Directory {dir_name} not found"
            assert dir_path.is_dir(), f"{dir_name} is not a directory"
        
        for file_name in required_files:
            file_path = Path(PROJECT_ROOT) / file_name
            assert file_path.exists(), f"File {file_name} not found"
            assert file_path.is_file(), f"{file_name} is not a file"

    def test_env_example_format(self):
        """Test that .env.example has correct format"""
        env_example = Path(PROJECT_ROOT) / '.env.example'
        assert env_example.exists(), ".env.example not found"
        
        content = env_example.read_text()
        assert 'NEWS_API_KEY=' in content
        assert 'FRED_API_KEY=' in content

    def test_requirements_format(self):
        """Test that requirements.txt has correct format"""
        requirements = Path(PROJECT_ROOT) / 'requirements.txt'
        assert requirements.exists(), "requirements.txt not found"
        
        content = requirements.read_text()
        
        # Check required packages are listed
        required_packages = [
            'streamlit>=1.35.0',
            'plotly>=5.20.0',
            'yfinance>=0.2.40',
            'pandas>=2.2.0',
            'numpy>=1.26.0',
            'statsmodels>=0.14.0',
            'scipy>=1.12.0',
            'vaderSentiment>=3.3.2',
            'fredapi>=0.5.1',
            'newsapi-python>=0.2.7',
            'python-dotenv>=1.0.0',
            'requests>=2.31.0'
        ]
        
        for package in required_packages:
            assert package in content, f"{package} not found in requirements.txt"

    def test_app_py_structure(self):
        """Test that app.py has required imports and structure"""
        app_py = Path(PROJECT_ROOT) / 'app.py'
        assert app_py.exists(), "app.py not found"
        
        content = app_py.read_text()
        
        # Check required imports
        assert 'import streamlit as st' in content or 'from streamlit' in content
        assert 'load_dotenv' in content
        assert 'set_page_config' in content
        
        # Check sidebar elements
        assert 'sidebar' in content.lower()
        assert 'date_input' in content
        assert 'button' in content
        
        # Check session state initialization
        assert 'session_state' in content
        assert 'if' in content and 'not in st.session_state' in content

    def test_streamlit_app_runs(self):
        """Test that Streamlit app can run without errors"""
        import subprocess
        import time
        import requests
        
        # Start Streamlit app
        process = subprocess.Popen(
            ['streamlit', 'run', 'app.py', '--server.headless', 'true', '--server.port', '8502'],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for app to start
        time.sleep(5)
        
        try:
            # Check if app is responding
            response = requests.get('http://localhost:8502', timeout=5)
            assert response.status_code == 200
            # Just verify the app loaded successfully (any content is fine)
            assert len(response.text) > 100  # App should return substantial HTML
        except requests.RequestException:
            pytest.fail("Streamlit app failed to start or respond")
        finally:
            process.terminate()
            process.wait(timeout=5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])