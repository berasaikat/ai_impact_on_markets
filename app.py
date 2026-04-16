import streamlit as st
from dotenv import load_dotenv

# Import and call load_dotenv() from python-dotenv before any config imports
load_dotenv()

import datetime
from config import (
    AI_EVENTS, 
    AI_TICKERS, 
    AI_ADJACENT_TICKERS, 
    ALL_TICKERS, 
    DEFAULT_START_DATE
)
from utils.cache import clear_all_caches

# Call st.set_page_config as the very first Streamlit call
st.set_page_config(page_title='AI Market Impact', page_icon='📈', layout='wide')

def main():
    # Initialise session_state defaults
    if 'basket' not in st.session_state:
        st.session_state['basket'] = 'AI Pure-Play'
        
    start_dt = datetime.datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d").date()
    today_dt = datetime.date.today()
    
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = start_dt
        
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = today_dt

    # Sidebar
    st.sidebar.markdown("# AI Market Impact")
    st.sidebar.markdown("Tracking the singularity across financial markets")
    st.sidebar.markdown("---")
    
    # Ticker basket selector (setting key='basket' stores it directly in session_state)
    st.sidebar.selectbox(
        'Basket', 
        ['AI Pure-Play', 'AI Adjacent', 'Both'],
        key='basket'
    )
    
    # Global date range selector
    selected_range = st.sidebar.date_input(
        'Date range',
        value=(st.session_state['start_date'], st.session_state['end_date']),
        key='global_date_range'
    )
    
    # Safely unpack the tuple from date_input and store in session_state
    if isinstance(selected_range, tuple):
        if len(selected_range) == 2:
            st.session_state['start_date'] = selected_range[0]
            st.session_state['end_date'] = selected_range[1]
            delta_days = (selected_range[1] - selected_range[0]).days
        elif len(selected_range) == 1:
            st.session_state['start_date'] = selected_range[0]
            st.session_state['end_date'] = selected_range[0]
            delta_days = 0
        else:
            delta_days = 0
    else:
        st.session_state['start_date'] = selected_range
        st.session_state['end_date'] = selected_range
        delta_days = 0
        
    st.sidebar.markdown("---")
    
    # Refresh button
    if st.sidebar.button('🔄 Refresh data'):
        clear_all_caches()
        st.sidebar.success("Caches cleared!")
        
    # Main Area
    st.title("AI Market Impact Dashboard")
    st.markdown("""
        Welcome to the AI Market Impact Dashboard. This application tracks the overarching
        effects of artificial intelligence advancements on the financial markets, highlighting 
        correlations between major AI product releases, regulatory actions, earnings reports, 
        and the performance of key industry players. Use the sidebar to configure your portfolio 
        basket and target date range.
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total AI Events Tracked", len(AI_EVENTS))
        
    with col2:
        # Determine tickers in current basket
        basket_sel = st.session_state['basket']
        if basket_sel == 'AI Pure-Play':
            num_tickers = len(AI_TICKERS)
        elif basket_sel == 'AI Adjacent':
            num_tickers = len(AI_ADJACENT_TICKERS)
        else:
            num_tickers = len(ALL_TICKERS)
            
        st.metric("Tickers in Current Basket", num_tickers)
        
    with col3:
        st.metric("Date Range Length (Days)", f"{delta_days} days")

if __name__ == "__main__":
    main()
