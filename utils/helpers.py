"""
Utility helper functions.
"""
import streamlit as st
from datetime import datetime


def format_currency(value, currency="USD"):
    """Format number as currency string."""
    if currency == "INR":
        return f"₹{value:,.2f}"
    return f"${value:,.2f}"


def format_percentage(value):
    """Format number as percentage with sign."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def format_large_number(value):
    """Format large numbers with K/M/B suffixes."""
    if abs(value) >= 1e9:
        return f"{value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.1f}K"
    return f"{value:.2f}"


def get_market_from_symbol(symbol):
    """Determine market from stock symbol suffix."""
    symbol = symbol.upper()
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return "INDIA"
    return "US"


def get_yfinance_symbol(symbol, market="US"):
    """Convert symbol to yfinance format."""
    symbol = symbol.upper()
    if market == "INDIA":
        if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
            return f"{symbol}.NS"
    return symbol


def signal_emoji(signal):
    """Return emoji for trading signal."""
    if signal == "BUY":
        return "🟢"
    elif signal == "SELL":
        return "🔴"
    return "🟡"


def pnl_color(value):
    """Return color for P&L value."""
    return "green" if value >= 0 else "red"


def get_date_range(period):
    """Map period string to start/end dates."""
    from datetime import timedelta
    end = datetime.now()
    period_map = {
        "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825,
    }
    days = period_map.get(period, 180)
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def check_alerts(alerts, current_data):
    """Check if any alerts should be triggered."""
    from utils.database import mark_alert_triggered
    triggered = []
    for alert in alerts:
        symbol = alert["symbol"]
        if symbol not in current_data.index:
            continue
        price = current_data.loc[symbol, "Close"] if "Close" in current_data.columns else None
        if price is None:
            continue

        should_trigger = False
        if alert["alert_type"] == "price_above" and price >= alert["target_value"]:
            should_trigger = True
        elif alert["alert_type"] == "price_below" and price <= alert["target_value"]:
            should_trigger = True

        if should_trigger:
            mark_alert_triggered(alert["id"])
            triggered.append(alert)
    return triggered
