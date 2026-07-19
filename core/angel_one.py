"""
Angel One SmartAPI — Real-time trading integration for Indian stock market.
Full API: market data, order placement, portfolio, WebSocket streaming.

Setup:
  1. Open Angel One account
  2. Go to https://smartapi.angelbroking.com/ → Create App
  3. Get Client ID + API Key + Password
  4. Generate TOTP seed (for 2FA login)
"""
import streamlit as st
import pandas as pd
import os
import re
import json
from datetime import datetime, timedelta

# API Credentials from environment
ANGEL_CLIENT_ID = os.environ.get("ANGEL_CLIENT_ID", "")
ANGEL_API_KEY = os.environ.get("ANGEL_API_KEY", "")
ANGEL_PASSWORD = os.environ.get("ANGEL_PASSWORD", "")
ANGEL_TOTP = os.environ.get("ANGEL_TOTP", "")


def generate_totp(secret):
    """
    Generate a 6-digit TOTP code from a base32 secret seed.
    If the input is already a 6-digit numeric string, return it as-is
    for backward compatibility.
    """
    if not secret:
        return "000000"
    # Backward compatibility: if it's already a 6-digit code, use it directly
    if re.fullmatch(r'\d{6}', secret.strip()):
        return secret.strip()
    # Otherwise, treat it as a base32 TOTP seed
    try:
        import pyotp
        totp = pyotp.TOTP(secret.strip())
        return totp.now()
    except Exception:
        return "000000"


# Session state
_session = None
_session_token = None
_refresh_token = None
_feed_token = None
_login_attempted = False
_last_login_result = None


def is_configured():
    """Check if API credentials are set."""
    return bool(ANGEL_CLIENT_ID and ANGEL_API_KEY and ANGEL_PASSWORD and ANGEL_TOTP)


def is_authenticated():
    """Check if logged in with valid session."""
    return _session is not None and _session_token is not None


def login():
    """Login to Angel One SmartAPI with auto-generated TOTP."""
    global _session, _session_token, _refresh_token, _feed_token, _last_login_result

    if not is_configured():
        _last_login_result = {"error": "API credentials not configured"}
        return _last_login_result

    try:
        from SmartApi import SmartConnect

        # Auto-generate TOTP code from seed
        totp_code = generate_totp(ANGEL_TOTP)

        session = SmartConnect(api_key=ANGEL_API_KEY)
        data = session.generateSession(
            clientCode=ANGEL_CLIENT_ID,
            password=ANGEL_PASSWORD,
            totp=totp_code,
        )
        if data.get("status"):
            _session = session
            _session_token = data.get("data", {}).get("jwtToken")
            _refresh_token = data.get("data", {}).get("refreshToken")
            _feed_token = data.get("data", {}).get("feedToken")

            # Register session expiry hook for auto-refresh
            session.setSessionExpiryHook(_on_session_expired)

            _last_login_result = {"success": True, "message": "Logged in to Angel One"}
            return _last_login_result
        else:
            _last_login_result = {"error": data.get("message", "Login failed")}
            return _last_login_result
    except Exception as e:
        _last_login_result = {"error": str(e)}
        return _last_login_result


def logout():
    """Logout from Angel One."""
    global _session, _session_token, _refresh_token, _feed_token, _login_attempted
    if _session:
        try:
            _session.terminateSession(ANGEL_CLIENT_ID)
        except:
            pass
    _session = None
    _session_token = None
    _refresh_token = None
    _feed_token = None
    _login_attempted = False


def get_session():
    """Get active session, attempting auto-login if needed."""
    global _session, _login_attempted

    if is_authenticated():
        return _session

    if not _login_attempted and is_configured():
        _login_attempted = True
        result = login()
        if "error" in result:
            return None
        return _session

    return None


# ── Session Management ───────────────────────────────────────

def _on_session_expired():
    """Callback for when SmartConnect detects an expired session."""
    global _session, _session_token, _refresh_token, _feed_token, _last_login_result

    # Try to refresh the token
    if _session and _refresh_token:
        try:
            result = refresh_session()
            if result.get("success"):
                return
        except Exception:
            pass

    # If refresh failed, attempt full re-login
    _session = None
    _session_token = None
    login()


def refresh_session():
    """Refresh the session using the stored refresh token."""
    global _session, _session_token, _refresh_token, _feed_token

    if not _session or not _refresh_token:
        return {"error": "No session to refresh"}

    try:
        response = _session.generateToken(_refresh_token)
        if response and response.get("status"):
            _session_token = response["data"]["jwtToken"]
            _refresh_token = response["data"]["refreshToken"]
            _feed_token = response["data"]["feedToken"]
            return {"success": True, "message": "Session refreshed"}
        return {"error": "Refresh failed"}
    except Exception as e:
        return {"error": str(e)}


def get_login_status():
    """Return human-readable login status for UI display."""
    if is_authenticated():
        return {"status": "connected", "message": "Connected to Angel One SmartAPI"}
    # Try auto-login if configured but not yet attempted
    if is_configured() and not _login_attempted:
        get_session()
        if is_authenticated():
            return {"status": "connected", "message": "Connected to Angel One SmartAPI"}
    if _last_login_result and "error" in _last_login_result:
        return {"status": "error", "message": _last_login_result["error"]}
    elif not is_configured():
        return {"status": "not_configured", "message": "Credentials not configured"}
    else:
        return {"status": "disconnected", "message": "Not connected"}


def get_feed_token():
    """Get the current feed token for WebSocket connections."""
    return _feed_token


# ── Market Quotes ─────────────────────────────────────────────

@st.cache_data(ttl=5)
def get_ltp(symbol, exchange="NSE"):
    """Get last traded price for a symbol."""
    sess = get_session()
    if not sess:
        return None
    try:
        params = {"symbol": symbol.upper(), "exchange": exchange.upper()}
        resp = sess.ltpData(exchange.upper(), symbol.upper(), "", symbol.upper())
        if resp and resp.get("status"):
            return resp["data"]
        return None
    except Exception:
        return None


@st.cache_data(ttl=3)
def get_quote(symbol, exchange="NSE"):
    """Get full quote data (bid, ask, LTP, volume, etc.)."""
    sess = get_session()
    if not sess:
        return None
    try:
        params = {
            "mode": "FULL",
            "exchangeTokens": {exchange.upper(): [symbol.upper()]},
        }
        resp = sess.getMarketData(params)
        if resp and resp.get("status"):
            data = resp["data"]
            if isinstance(data, dict) and "fetched" in data:
                fetched = data["fetched"]
                for item in fetched:
                    if item.get("tradingSymbol", "").upper() == symbol.upper():
                        return item
            return data
        return None
    except Exception:
        return None


def get_market_status():
    """Check if Indian markets are open."""
    sess = get_session()
    if not sess:
        now = datetime.now()
        h = now.hour
        wd = now.weekday()
        if wd >= 5: return "Weekend - Markets Closed"
        if 9 <= h <= 15 and not (h == 15 and now.minute > 30): return "🟢 Market Open (9:15 AM - 3:30 PM IST)"
        return "🔴 Market Closed"
    try:
        resp = sess.getMarketStatus()
        if resp and resp.get("status"):
            return f"🟢 {resp['data'].get('message', 'Open')}" if resp['data'].get('marketStatus') == 'OPEN' else f"🔴 Market Closed"
    except:
        pass
    return "🔴 Market Closed"


def get_historical(symbol, interval="ONE_DAY", days=365, exchange="NSE"):
    """
    Get historical OHLCV data.
    interval: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_DAY, ONE_WEEK, ONE_MONTH
    """
    sess = get_session()
    if not sess:
        return None
    try:
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        params = {
            "exchange": exchange.upper(),
            "symbol": symbol.upper(),
            "interval": interval,
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M"),
        }
        resp = sess.getCandleData(params)
        if resp and resp.get("status"):
            candles = resp["data"]
            df = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
            return df
        return None
    except Exception:
        return None


# ── Order Placement ──────────────────────────────────────────

def place_order(symbol, action, quantity, order_type="MARKET", product="INTRADAY",
                exchange="NSE", price=0, trigger_price=0):
    """
    Place an order via Angel One.
    action: "BUY" or "SELL"
    order_type: "MARKET", "LIMIT", "STOPLOSS_LIMIT", "STOPLOSS_MARKET"
    product: "INTRADAY", "DELIVERY", "STOPLOSS"
    Returns: order_id dict
    """
    sess = get_session()
    if not sess:
        return {"error": "Not logged in to Angel One"}

    try:
        order_params = {
            "symbol": symbol.upper(),
            "exchange": exchange.upper(),
            "tradingsymbol": symbol.upper(),
            "transactiontype": action.upper(),
            "quantity": quantity,
            "producttype": product.upper(),
            "ordertype": order_type.upper(),
        }
        if price > 0:
            order_params["price"] = price
        if trigger_price > 0:
            order_params["triggerprice"] = trigger_price
        if not price and order_type.upper() == "MARKET":
            order_params["price"] = 0

        resp = sess.placeOrder(order_params)
        if resp and resp.get("status"):
            return {"success": True, "order_id": resp.get("data", {}).get("orderid", resp["data"])}
        return {"error": resp.get("message", "Order failed")}
    except Exception as e:
        return {"error": str(e)}


def get_positions():
    """Get current positions."""
    sess = get_session()
    if not sess: return None
    try:
        resp = sess.position()
        if resp and resp.get("status"):
            return resp.get("data", [])
        return []
    except: return []


def get_holdings():
    """Get holdings (delivery)."""
    sess = get_session()
    if not sess: return None
    try:
        resp = sess.holding()
        if resp and resp.get("status"):
            return resp.get("data", [])
        return []
    except: return []


def get_order_history():
    """Get order history."""
    sess = get_session()
    if not sess: return None
    try:
        resp = sess.orderBook()
        if resp and resp.get("status"):
            return resp.get("data", [])
        return []
    except: return []


def get_account():
    """Get account info."""
    sess = get_session()
    if not sess: return None
    try:
        resp = sess.rmsLimit()
        if resp and resp.get("status"):
            return resp.get("data", {})
        return None
    except: return None


# ── Convenience: Get best price ──────────────────────────────

def get_best_price(symbol, exchange="NSE"):
    """Get best available price for a symbol."""
    ltp = get_ltp(symbol, exchange)
    if ltp and isinstance(ltp, dict):
        price = ltp.get("ltp") or ltp.get("lastprice", 0)
        return {
            "symbol": symbol,
            "price": float(price),
            "open": float(ltp.get("open", 0)),
            "high": float(ltp.get("high", 0)),
            "low": float(ltp.get("low", 0)),
            "change": float(ltp.get("change", 0)),
            "change_pct": float(ltp.get("changePct", 0) or ltp.get("percentageChange", 0)),
            "source": "angel_rt",
        }
    return None
