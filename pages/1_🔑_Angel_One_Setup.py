"""
Page 1: Angel One Setup — Authenticate for real trading + real-time data.
All you need: Angel One account -> SmartAPI keys -> Login -> Real Trading.
"""
import streamlit as st
import os
from datetime import datetime

st.set_page_config(page_title="Angel One Setup", page_icon="🔑", layout="wide")
st.title("🔑 Angel One SmartAPI — Real Trading Setup")

from core.angel_one import (
    is_configured, is_authenticated, login, logout, get_session,
    get_market_status, get_ltp, get_positions, get_holdings,
    get_order_history, get_account, get_quote,
    get_login_status, generate_totp, ANGEL_TOTP
)

# ── Connection Status ──
# Trigger auto-login on first visit
if is_configured() and not is_authenticated():
    get_session()
status = get_login_status()
if status["status"] == "connected":
    st.success(f"✅ **{status['message']} — Live Trading Mode**")
    try:
        market_status = get_market_status()
        st.info(f"📊 Market: {market_status}")
    except:
        pass
elif status["status"] == "error":
    st.error(f"❌ **Connection failed:** {status['message']}")
else:
    st.warning("🔴 **Not connected to Angel One** — Using paper trading + delayed data")

st.markdown("---")

# ── Setup Guide ──
st.markdown("## 📋 Setup Guide")

tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Get API Key", "2️⃣ Set Variables", "3️⃣ Login", "4️⃣ Dashboard"])

with tab1:
    st.markdown("""
    ### Step 1: Get your Angel One SmartAPI credentials

    1. Go to **[smartapi.angelbroking.com](https://smartapi.angelbroking.com/)**
    2. Login with your Angel One account
    3. Click **Create New App**
    4. Fill in:
       - **App Name:** `TradingAssistant`
    5. You'll receive:
       - **Client ID** (your trading account ID)
       - **API Key** (your app's API key)
       - **Password** (your trading password)

    ### Step 2: Generate TOTP

    - You can use an authenticator app (Google Authenticator, etc.)
    - Or use the **base32 TOTP secret seed** for automated login
    - The app will auto-generate 6-digit codes from your seed at login time

    ⚠️ **Keep these safe! Never share API credentials publicly.**
    """)
    st.info("💡 SmartAPI is **free** with an Angel One trading account. No extra charges.")

with tab2:
    st.markdown("""
    ### Step 3: Set Environment Variables

    Open PowerShell and run:

    ```powershell
    $env:ANGEL_CLIENT_ID = "your_client_id"
    $env:ANGEL_API_KEY = "your_api_key"
    $env:ANGEL_PASSWORD = "your_password"
    $env:ANGEL_TOTP = "your_totp_secret_seed"

    # Set permanently:
    [Environment]::SetEnvironmentVariable("ANGEL_CLIENT_ID", "your_client_id", "User")
    [Environment]::SetEnvironmentVariable("ANGEL_API_KEY", "your_api_key", "User")
    [Environment]::SetEnvironmentVariable("ANGEL_PASSWORD", "your_password", "User")
    [Environment]::SetEnvironmentVariable("ANGEL_TOTP", "your_totp_secret_seed", "User")
    ```

    **ANGEL_TOTP** should be your base32 TOTP secret seed (e.g., `IYG27JQI53K277ATA3UVR2NSYM`).
    The app will auto-generate 6-digit TOTP codes from this seed at login time.

    **Then restart the Streamlit app.**
    """)

    st.markdown("---")
    st.markdown("#### Or set TOTP seed via UI (current session only)")
    with st.form("totp_seed_form"):
        totp_seed = st.text_input(
            "TOTP Secret Seed",
            type="password",
            help="Base32 seed from Angel One TOTP setup (e.g., IYG27JQI53K277ATA3UVR2NSYM)",
            placeholder="Enter your TOTP seed...",
        )
        submitted = st.form_submit_button("Save TOTP Seed")
        if submitted and totp_seed:
            import core.angel_one as angel_module
            os.environ["ANGEL_TOTP"] = totp_seed.strip()
            angel_module.ANGEL_TOTP = totp_seed.strip()
            st.success("✅ TOTP seed saved for this session!")
            st.info("For permanent storage, use the PowerShell commands above and restart the app.")

with tab3:
    st.markdown("### 🔐 Login to Angel One")

    login_status = get_login_status()

    if login_status["status"] == "connected":
        st.success(f"✅ {login_status['message']}")
        st.caption("🔄 Auto-login active. Session will refresh automatically.")
        if st.button("🔴 Logout", type="secondary"):
            logout()
            st.rerun()
    elif login_status["status"] == "error":
        st.error(f"❌ Login failed: {login_status['message']}")
        if st.button("🔄 Retry Login", type="primary"):
            with st.spinner("Logging in..."):
                result = login()
                if "success" in result:
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.error(f"Login failed: {result.get('error')}")
    elif login_status["status"] == "not_configured":
        st.warning("⚠️ Set ANGEL_CLIENT_ID, ANGEL_API_KEY, ANGEL_PASSWORD, and ANGEL_TOTP first (Step 2)")
    else:
        st.info("Not connected. Click below to login.")
        if st.button("🔐 Login Now", type="primary"):
            with st.spinner("Logging in..."):
                result = login()
                if "success" in result:
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.error(f"Login failed: {result.get('error')}")
                    st.info("💡 Check your credentials in environment variables")

    if is_authenticated():
        st.markdown("---")
        st.markdown("#### Quick Price Test")
        test_sym = st.text_input("Symbol", value="RELIANCE", key="angel_test_sym")
        if st.button("Check Price"):
            ltp = get_ltp(test_sym.upper())
            if ltp:
                price = ltp.get("ltp") or ltp.get("lastprice", "?")
                st.success(f"{test_sym.upper()}: ₹{price}")
            else:
                st.error(f"Could not fetch price for {test_sym.upper()}")

with tab4:
    st.markdown("### 📊 Auto Logout Settings")

    if is_authenticated():
        st.markdown("#### Current Positions")
        positions = get_positions()
        if positions:
            pos_data = []
            for p in positions:
                if p.get("netQty", 0) != 0:
                    pos_data.append({
                        "Symbol": p.get("symbol", ""),
                        "Qty": p.get("netQty", 0),
                        "Avg": p.get("avgPrice", 0),
                        "LTP": p.get("ltp", 0),
                        "P&L": p.get("pnl", 0),
                    })
            if pos_data:
                import pandas as pd
                st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
            else:
                st.info("No open positions")
        else:
            st.info("No positions or login required")

        st.markdown("#### Recent Orders")
        orders = get_order_history()
        if orders:
            import pandas as pd
            od = [{
                "ID": o.get("orderid", ""),
                "Symbol": o.get("symbol", ""),
                "Side": o.get("transactiontype", ""),
                "Qty": o.get("quantity", 0),
                "Price": o.get("price", 0),
                "Status": o.get("status", ""),
            } for o in orders[-10:]]
            st.dataframe(pd.DataFrame(od), use_container_width=True, hide_index=True)

        st.markdown("#### Account Info")
        acc = get_account()
        if acc:
            st.json(acc)
