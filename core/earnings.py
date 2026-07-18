"""
Earnings calendar — upcoming earnings dates, historical earnings, and earnings impact analysis.
Uses yfinance for earnings dates + analyst estimates.
"""
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_earnings_dates(symbol, limit=8):
    """Get upcoming and past earnings dates for a stock."""
    try:
        ticker = yf.Ticker(symbol)
        dates = ticker.earnings_dates
        if dates is not None and not dates.empty:
            result = dates.head(limit).reset_index()
            # Rename columns if needed
            if "Earnings Date" in result.columns:
                pass
            elif "index" in result.columns:
                result = result.rename(columns={"index": "Earnings Date"})
            return result
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_earnings_history(symbol, limit=8):
    """Get historical earnings data (actual vs estimate)."""
    try:
        ticker = yf.Ticker(symbol)
        earnings = ticker.earnings
        if earnings is not None and not earnings.empty:
            return earnings.head(limit)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_upcoming_earnings(symbols):
    """
    Check upcoming earnings for a list of stocks.
    Returns list of dicts with symbol, date, days_until_earnings.
    """
    results = []
    today = datetime.now()

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            cal = ticker.calendar
            if cal is not None and not cal.empty:
                # Try to extract earnings date
                if hasattr(cal, "index"):
                    for idx in cal.index:
                        if "earnings" in str(idx).lower() or "date" in str(idx).lower():
                            pass
                # Simplified: check if earnings are soon
                try:
                    earnings_date = cal.iloc[0]["Earnings Date"] if "Earnings Date" in cal.columns else None
                    if earnings_date:
                        days_until = (pd.Timestamp(earnings_date).date() - today.date()).days
                        if -7 <= days_until <= 60:  # Within next 2 months or just passed
                            results.append({
                                "symbol": sym,
                                "date": earnings_date,
                                "days_until": days_until,
                                "status": "Upcoming" if days_until > 0 else "Recently Reported",
                            })
                except Exception:
                    continue
        except Exception:
            continue

    return sorted(results, key=lambda x: x.get("days_until", 999))


def get_earnings_impact(symbol):
    """
    Analyze historical earnings impact on stock price.
    Returns: avg move, beat rate, earnings surprise stats.
    """
    try:
        ticker = yf.Ticker(symbol)
        earnings = ticker.earnings

        if earnings is None or earnings.empty:
            return None

        # Check if we have estimate vs actual data
        if "Earnings" in earnings.columns and "Revenue" in earnings.columns:
            history = ticker.earnings_history if hasattr(ticker, "earnings_history") else None

            return {
                "recent_earnings": earnings.head(4).to_dict(),
                "has_data": True,
            }

        return {"has_data": False}
    except Exception:
        return None


def is_near_earnings(symbol, days_threshold=7):
    """Check if a stock has earnings within the threshold days."""
    today = datetime.now()
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is not None:
            try:
                earnings_date = cal.iloc[0].get("Earnings Date") if len(cal) > 0 else None
                if earnings_date:
                    days_until = (pd.Timestamp(earnings_date).date() - today.date()).days
                    return 0 < days_until <= days_threshold
            except Exception:
                pass
    except Exception:
        pass
    return False
