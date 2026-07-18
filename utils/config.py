"""
Application configuration — API keys, defaults, and constants.
"""
import os

# Finnhub API key (free tier: 50 calls/day)
# Get your free key at: https://finnhub.io/register
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

# Default settings
DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"

# Supported markets
MARKETS = ["INDIA"]

# Indian stock suffixes for Yahoo Finance
INDIAN_SUFFIXES = {
    "NSE": ".NS",
    "BSE": ".BO",
}
DEFAULT_RISK_PER_TRADE = 2.0  # %
DEFAULT_CAPITAL = 100000.0
DEFAULT_WIN_RATE = 0.55
DEFAULT_WIN_LOSS_RATIO = 1.5

# Strategy weights for combined signal
STRATEGY_WEIGHTS = {
    "rsi": 0.25,
    "macd": 0.25,
    "ma_crossover": 0.25,
    "bollinger": 0.25,
}

# Common Indian stocks for quick access
POPULAR_INDIAN = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "TATAMOTORS.NS", "WIPRO.NS",
    "ASIANPAINT.NS", "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS",
]

# Risk management defaults
