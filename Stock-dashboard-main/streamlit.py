import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import json
import os


# Set page config
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide"
)

# Alpha Vantage API configuration
API_KEY = os.getenv("ALPHA_VANTAGE_KEY")  # Replace with your Alpha Vantage API key
BASE_URL = "https://www.alphavantage.co/query"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol):
    """Fetch stock data from Alpha Vantage API"""
    try:
        # Get daily data
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': API_KEY,
            'outputsize': 'compact'  # Last 100 data points
        }
        
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if "Error Message" in data:
            st.error(f"Error fetching data for {symbol}: {data['Error Message']}")
            return None
            
        if "Note" in data:
            st.warning("API call frequency limit reached. Please wait a moment.")
            return None
            
        time_series = data.get('Time Series (Daily)', {})
        
        if not time_series:
            st.error(f"No data found for symbol: {symbol}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        return df
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_metrics(df):
    """Calculate key stock metrics"""
    if df is None or df.empty:
        return {}
    
    latest_price = df['Close'].iloc[-1]
    previous_price = df['Close'].iloc[-2] if len(df) > 1 else latest_price
    
    # Calculate moving averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    metrics = {
        'current_price': latest_price,
        'daily_change': latest_price - previous_price,
        'daily_change_pct': ((latest_price - previous_price) / previous_price) * 100,
        'volume': df['Volume'].iloc[-1],
        'high_52w': df['High'].max(),
        'low_52w': df['Low'].min(),
        'avg_volume': df['Volume'].mean(),
        'volatility': df['Daily_Return'].std() * 100,
        'ma_20': df['MA_20'].iloc[-1] if not pd.isna(df['MA_20'].iloc[-1]) else None,
        'ma_50': df['MA_50'].iloc[-1] if not pd.isna(df['MA_50'].iloc[-1]) else None,
    }
    
    return metrics, df

def create_price_chart(df, symbol):
    """Create interactive price chart"""
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add moving averages if available
    if 'MA_20' in df.columns and not df['MA_20'].isna().all():
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA_20'],
            mode='lines',
            name='20-Day MA',
            line=dict(color='orange', width=1, dash='dash')
        ))
    
    if 'MA_50' in df.columns and not df['MA_50'].isna().all():
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA_50'],
            mode='lines',
            name='50-Day MA',
            line=dict(color='red', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title=f'{symbol} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=400,
        showlegend=True
    )
    
    return fig

def create_volume_chart(df, symbol):
    """Create volume chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f'{symbol} Trading Volume',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=300
    )
    
    return fig

def main():
    st.title("📈 Stock Market Dashboard")
    st.markdown("Track your favorite stocks with real-time data and analytics")
    
    # Sidebar for stock selection
    st.sidebar.header("Stock Selection")
    
    # Predefined popular stocks
    popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    
    # Stock symbol input
    symbol = st.sidebar.selectbox(
        "Choose a stock symbol:",
        options=popular_stocks,
        index=0
    )
    
    # Custom symbol input
    custom_symbol = st.sidebar.text_input(
        "Or enter a custom symbol:",
        placeholder="e.g., AAPL"
    ).upper()
    
    if custom_symbol:
        symbol = custom_symbol
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Time Period:",
        options=['30 Days', '60 Days', '90 Days', 'All Available'],
        index=0
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # API Key check
    if API_KEY == "your_api_key_here":
        st.error("⚠️ Please add your Alpha Vantage API key to use this dashboard.")
        st.info("Get your free API key from: https://www.alphavantage.co/support/#api-key")
        st.stop()
    
    # Main dashboard
    if symbol:
        st.header(f"Stock Analysis: {symbol}")
        
        # Fetch data
        with st.spinner(f"Fetching data for {symbol}..."):
            df = fetch_stock_data(symbol)
        
        if df is not None and not df.empty:
            # Filter data based on time period
            if time_period != 'All Available':
                days = int(time_period.split()[0])
                df = df.tail(days)
            
            # Calculate metrics
            metrics, df_with_indicators = calculate_metrics(df)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${metrics['current_price']:.2f}",
                    f"{metrics['daily_change']:+.2f} ({metrics['daily_change_pct']:+.2f}%)"
                )
            
            with col2:
                st.metric(
                    "Volume",
                    f"{metrics['volume']:,}",
                    f"Avg: {metrics['avg_volume']:,.0f}"
                )
            
            with col3:
                st.metric(
                    "52W High",
                    f"${metrics['high_52w']:.2f}"
                )
            
            with col4:
                st.metric(
                    "52W Low",
                    f"${metrics['low_52w']:.2f}"
                )
            
            # Additional metrics
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                if metrics['ma_20']:
                    st.metric("20-Day MA", f"${metrics['ma_20']:.2f}")
            
            with col6:
                if metrics['ma_50']:
                    st.metric("50-Day MA", f"${metrics['ma_50']:.2f}")
            
            with col7:
                st.metric("Volatility", f"{metrics['volatility']:.2f}%")
            
            # Charts
            st.subheader("Price Chart")
            price_chart = create_price_chart(df_with_indicators, symbol)
            st.plotly_chart(price_chart, use_container_width=True)
            
            st.subheader("Volume Chart")
            volume_chart = create_volume_chart(df, symbol)
            st.plotly_chart(volume_chart, use_container_width=True)
            
            # Data table
            with st.expander("📊 Raw Data"):
                st.dataframe(df.tail(10).round(2))
            
            # Download data
            csv = df.to_csv()
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{symbol}_stock_data.csv",
                mime="text/csv"
            )
        
        else:
            st.error(f"Unable to fetch data for {symbol}. Please check the symbol and try again.")

if __name__ == "__main__":
    main()