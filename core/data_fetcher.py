"""
Stock data fetching — yfinance (free, unlimited, handles Indian stocks).
"""
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# ── yfinance (primary data source) ─────────────────────────────

@st.cache_data(ttl=3600)
def get_stock_data(symbol, period="6mo", interval="1d"):
    """
    Fetch historical OHLCV data via yfinance.
    Works for both US and Indian stocks (.NS / .BO suffix).
    Returns DataFrame with Date index and Open/High/Low/Close/Volume columns.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        # Clean up — drop dividends/stock splits if present
        cols_to_keep = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in cols_to_keep if c in df.columns]]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    """
    Fetch company info and fundamentals via yfinance.
    Returns dict with name, sector, market_cap, etc.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("shortName") or info.get("longName", symbol),
            "symbol": symbol,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume", 0),
            "currency": info.get("currency", "INR"),
        }
    except Exception:
        return {"name": symbol, "symbol": symbol, "sector": "N/A", "currency": "INR"}


@st.cache_data(ttl=3600)
def get_multiple_stocks_data(symbols, period="1mo"):
    """Fetch close prices for multiple stocks at once (for watchlist)."""
    results = {}
    for symbol in symbols:
        df = get_stock_data(symbol, period=period, interval="1d")
        if not df.empty:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
            change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
            results[symbol] = {
                "price": latest["Close"],
                "change_pct": change_pct,
                "volume": latest.get("Volume", 0),
                "high": latest["High"],
                "low": latest["Low"],
            }
    return results


# ── Stock Search (yfinance) ────────────────────────────────────

def search_stocks(query):
    """
    Search for stocks by symbol or name.
    Uses yfinance search — returns list of matching tickers.
    """
    if not query or len(query) < 1:
        return []
    try:
        ticker = yf.Ticker(query.upper())
        info = ticker.info
        if info and info.get("symbol"):
            return [{
                "symbol": info["symbol"],
                "name": info.get("shortName") or info.get("longName", query.upper()),
                "exchange": info.get("exchange", ""),
                "type": info.get("quoteType", "Equity"),
            }]
        return []
    except Exception:
        return []
