"""
Stock screener — filter stocks by technical criteria.
"""
import pandas as pd
from core.data_fetcher import get_stock_data
from core.analyzer import calculate_all_indicators


def screen_stocks(symbols, filters=None):
    """
    Screen a list of stocks against technical filters.
    filters: dict with keys like 'rsi_min', 'rsi_max', 'macd_positive', etc.
    Returns: DataFrame of matching stocks with their indicator values.
    """
    if not filters:
        filters = {}

    results = []

    for symbol in symbols:
        try:
            df = get_stock_data(symbol, period="6mo")
            if df.empty or len(df) < 30:
                continue

            df = calculate_all_indicators(df)
            latest = df.iloc[-1]

            # Build stock data dict
            stock = {
                "Symbol": symbol,
                "Price": round(latest["Close"], 2),
                "RSI": round(latest.get("RSI", 0), 1) if pd.notna(latest.get("RSI")) else None,
                "MACD": round(latest.get("MACD", 0), 4) if pd.notna(latest.get("MACD")) else None,
                "SMA_20": round(latest.get("SMA_20", 0), 2) if pd.notna(latest.get("SMA_20")) else None,
                "SMA_50": round(latest.get("SMA_50", 0), 2) if pd.notna(latest.get("SMA_50")) else None,
                "ATR": round(latest.get("ATR", 0), 2) if pd.notna(latest.get("ATR")) else None,
                "Volume": int(latest.get("Volume", 0)),
                "BB_Width": round(latest.get("BB_Width", 0), 4) if pd.notna(latest.get("BB_Width")) else None,
            }

            # Apply filters
            passes = True

            if "rsi_min" in filters and stock["RSI"] is not None:
                if stock["RSI"] < filters["rsi_min"]:
                    passes = False
            if "rsi_max" in filters and stock["RSI"] is not None:
                if stock["RSI"] > filters["rsi_max"]:
                    passes = False

            if filters.get("macd_positive") and stock["MACD"] is not None:
                if stock["MACD"] <= 0:
                    passes = False

            if filters.get("macd_negative") and stock["MACD"] is not None:
                if stock["MACD"] >= 0:
                    passes = False

            if filters.get("price_above_sma20") and stock["SMA_20"] is not None:
                if stock["Price"] <= stock["SMA_20"]:
                    passes = False

            if filters.get("price_below_sma20") and stock["SMA_20"] is not None:
                if stock["Price"] >= stock["SMA_20"]:
                    passes = False

            if filters.get("price_above_sma50") and stock["SMA_50"] is not None:
                if stock["Price"] <= stock["SMA_50"]:
                    passes = False

            if "volume_min" in filters:
                if stock["Volume"] < filters["volume_min"]:
                    passes = False

            if passes:
                results.append(stock)

        except Exception:
            continue

    return pd.DataFrame(results) if results else pd.DataFrame()
