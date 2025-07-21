# Stock Market Dashboard 📈

A real-time stock market dashboard built with Python and Streamlit that provides comprehensive stock analysis and visualization tools.

![Dashboard Preview](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

### 📊 Real-Time Data
- Live stock price tracking with Alpha Vantage API
- Current price, daily changes, and percentage movements
- Trading volume analysis with historical averages
- 52-week high and low tracking

### 📈 Technical Analysis
- 20-day and 50-day moving averages
- Volatility calculations
- Daily return analysis
- Interactive price charts with technical indicators

### 🎯 User-Friendly Interface
- Pre-loaded popular stocks (AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, NFLX)
- Custom stock symbol input
- Multiple time period views (30, 60, 90 days, or all available data)
- Responsive design with organized metrics display

### 📋 Data Management
- Raw data table view
- CSV export functionality
- Data caching to optimize API usage
- Error handling and user feedback

## Screenshots

### Main Dashboard
The dashboard displays key metrics in an organized layout with real-time price updates and change indicators.

### Interactive Charts
- **Price Chart**: Candlestick-style visualization with moving averages overlay
- **Volume Chart**: Trading volume analysis with historical context

## Installation

### Prerequisites
- Python 3.8 or higher
- Alpha Vantage API key (free registration required)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd stock-market-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get API Key**
   - Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Sign up for a free account
   - Copy your API key

4. **Configure API Key**
   - Open `stock_dashboard.py`
   - Set API key as environment variable
   ```bash
   export ALPHA_VANTAGE_KEY={your API key} #bash
   set ALPHA_VANTAGE_KEY={your API key} #windows CLI
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit.py
   ```

6. **Access the dashboard**
   - Open your browser to `http://localhost:8501`
   - Start exploring stock data!

## Usage

### Basic Navigation
1. **Select Stock**: Choose from popular stocks or enter a custom symbol
2. **Time Period**: Select your preferred analysis timeframe
3. **Refresh**: Use the refresh button to get the latest data
4. **Download**: Export data as CSV for further analysis

### Understanding the Metrics
- **Current Price**: Latest closing price with daily change
- **Volume**: Current trading volume vs. historical average
- **52W High/Low**: Highest and lowest prices in the past year
- **Moving Averages**: 20-day and 50-day trend indicators
- **Volatility**: Price fluctuation measurement (higher = more volatile)

### Chart Interpretation
- **Blue Line**: Current stock price
- **Orange Dashed**: 20-day moving average (short-term trend)
- **Red Dashed**: 50-day moving average (long-term trend)
- **Volume Bars**: Trading activity levels

## Technical Details

### Architecture
```
stock_dashboard.py          # Main application file
├── Data Layer             # API integration and caching
├── Processing Layer       # Metrics calculation and analysis
├── Visualization Layer    # Charts and UI components
└── User Interface         # Streamlit components
```

### Key Technologies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive charting library
- **Requests**: HTTP library for API calls

### API Integration
- **Provider**: Alpha Vantage (free tier)
- **Rate Limits**: 5 calls/minute, 500 calls/day
- **Data Coverage**: Daily stock prices, volume, and company info
- **Caching**: 5-minute cache to optimize API usage

## Project Structure

```
stock-market-dashboard/
│
├── stock_dashboard.py      # Main application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── .gitignore            # Git ignore file (recommended)
```

## Customization Options

### Adding New Stocks
Edit the `popular_stocks` list in the code:
```python
popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'YOUR_STOCK']
```

### Modifying Time Periods
Update the time period options:
```python
time_period = st.sidebar.selectbox(
    "Time Period:",
    options=['7 Days', '30 Days', '60 Days', '1 Year'],
    index=0
)
```

### Custom Metrics
Add new calculations in the `calculate_metrics()` function:
```python
metrics['your_metric'] = your_calculation
```

## API Rate Limits

**Alpha Vantage Free Tier:**
- 5 API calls per minute
- 500 API calls per day
- If you hit limits, the app will display appropriate warnings

**Optimization Features:**
- 5-minute data caching
- Error handling for rate limits
- Efficient data processing

## Troubleshooting

### Common Issues

**"Error fetching data"**
- Check your API key is correctly entered
- Verify the stock symbol exists
- Ensure internet connection is stable

**"API call frequency limit reached"**
- Wait 1 minute before making new requests
- Consider upgrading to Alpha Vantage premium for higher limits

**Charts not displaying**
- Ensure all dependencies are installed
- Check that data was successfully fetched
- Verify Plotly is working: `import plotly` in Python

**Application won't start**
- Verify Python version (3.8+)
- Install requirements: `pip install -r requirements.txt`
- Check for port conflicts (default: 8501)

## Future Enhancements

### Potential Features
- [ ] Portfolio tracking with multiple stocks
- [ ] Price alerts and notifications
- [ ] Advanced technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Comparison charts for multiple stocks
- [ ] Historical performance analysis
- [ ] News integration for stock-related updates
- [ ] Export to PDF reports

### Technical Improvements
- [ ] Database integration for historical data storage
- [ ] Real-time WebSocket connections
- [ ] Mobile-responsive design enhancements
- [ ] User authentication and personalized dashboards

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Alpha Vantage** for providing free stock market API
- **Streamlit** team for the excellent web framework
- **Plotly** for interactive charting capabilities

## Contact

For questions, suggestions, or issues:
- Create an issue in the repository
- Email: [aditya.chandran@outlook.com]
- LinkedIn: [www.linkedin.com/in/aditya-chandran-202764172]

---

**⭐ If you found this project helpful, please consider giving it a star!**
