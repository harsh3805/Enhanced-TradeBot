"""
FII/DII (Foreign/Domestic Institutional Investors) data for Indian markets.
Fetches from NSE India API — the primary source for institutional flow data.
"""
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta


@st.cache_data(ttl=3600)
def get_fii_dii_data():
    """
    Fetch FII/DII trading activity from NSE India.
    Returns DataFrame with date, FII buy/sell, DII buy/sell, net flow.
    """
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/market-data/fii-dii",
    }

    try:
        # NSE requires a session to get cookies first
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        resp = session.get(url, headers=headers, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            rows = data.get("data", [])
            if rows:
                df = pd.DataFrame(rows)
                return df
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_fii_dii_flow_summary(days=30):
    """
    Get FII/DII flow summary for last N days.
    Returns dict with total FII buy/sell, DII buy/sell, net flow.
    """
    df = get_fii_dii_data()
    if df.empty:
        return None

    try:
        # Parse the data based on NSE response format
        # Columns typically: date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net
        summary = {
            "fii_buy": 0, "fii_sell": 0, "fii_net": 0,
            "dii_buy": 0, "dii_sell": 0, "dii_net": 0,
            "days": min(days, len(df)),
        }

        for _, row in df.head(days).iterrows():
            for key in summary:
                if key in row and pd.notna(row[key]):
                    try:
                        val = float(str(row[key]).replace(",", "").replace("%", ""))
                        summary[key] += val
                    except (ValueError, TypeError):
                        pass

        return summary
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_fii_dii_daily():
    """
    Get daily FII/DII activity as a DataFrame.
    Clean format with Date, FII Net, DII Net columns.
    """
    df = get_fii_dii_data()
    if df.empty:
        return pd.DataFrame()

    try:
        # Try to parse common NSE formats
        result = pd.DataFrame()
        for col in df.columns:
            col_lower = col.lower().strip()
            if "date" in col_lower:
                result["Date"] = pd.to_datetime(df[col], errors="coerce")
            elif "fii" in col_lower and "buy" in col_lower:
                result["FII Buy"] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
            elif "fii" in col_lower and "sell" in col_lower:
                result["FII Sell"] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
            elif "dii" in col_lower and "buy" in col_lower:
                result["DII Buy"] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
            elif "dii" in col_lower and "sell" in col_lower:
                result["DII Sell"] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
            elif "fii" in col_lower and "net" in col_lower:
                result["FII Net"] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
            elif "dii" in col_lower and "net" in col_lower:
                result["DII Net"] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

        if "FII Net" not in result.columns and "FII Buy" in result.columns and "FII Sell" in result.columns:
            result["FII Net"] = result["FII Buy"] - result["FII Sell"]
        if "DII Net" not in result.columns and "DII Buy" in result.columns and "DII Sell" in result.columns:
            result["DII Net"] = result["DII Buy"] - result["DII Sell"]

        return result.head(60) if not result.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_fii_signal():
    """
    Get a simple BUY/SELL signal based on FII flow trend.
    If FIIs are net buyers over last 10 days → BULLISH
    If FIIs are net sellers → BEARISH
    """
    summary = get_fii_dii_flow_summary(days=10)
    if not summary:
        return "NEUTRAL", 0, "FII/DII data unavailable"

    fii_net = summary.get("fii_net", 0)
    dii_net = summary.get("dii_net", 0)

    if fii_net > 5000:  # >5000 Cr net buy
        return "BULLISH", 70, f"FIIs are strong net buyers (₹{fii_net:,.0f} Cr over 10 days)"
    elif fii_net > 0:
        return "BULLISH", 50, f"FIIs are net buyers (₹{fii_net:,.0f} Cr over 10 days)"
    elif fii_net < -5000:
        return "BEARISH", 70, f"FIIs are strong net sellers (₹{fii_net:,.0f} Cr over 10 days)"
    elif fii_net < 0:
        return "BEARISH", 50, f"FIIs are net sellers (₹{fii_net:,.0f} Cr over 10 days)"

    return "NEUTRAL", 30, "FII flow is mixed"
