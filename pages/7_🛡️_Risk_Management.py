"""
Page 7: Risk Management — Position sizing, Kelly criterion, stop-loss calculator.
"""
import streamlit as st
from core.risk import kelly_criterion, calculate_position_size, risk_reward_ratio, portfolio_risk_metrics, var_calculation
from core.portfolio import get_portfolio_summary
from utils.helpers import format_currency
from utils.config import DEFAULT_RISK_PER_TRADE, DEFAULT_CAPITAL, DEFAULT_WIN_RATE, DEFAULT_WIN_LOSS_RATIO

st.set_page_config(page_title="Risk Management", page_icon="🛡️", layout="wide")
st.title("🛡️ Risk Management")

# ── Position Size Calculator ──
st.markdown("### 📐 Position Size Calculator")
st.markdown("Calculate how many shares to buy based on your risk tolerance.")

with st.form("position_calc"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        capital = st.number_input("Trading Capital", min_value=1000.0, value=DEFAULT_CAPITAL, step=1000.0)
    with col2:
        risk_pct = st.number_input("Risk Per Trade (%)", min_value=0.1, max_value=10.0, value=DEFAULT_RISK_PER_TRADE, step=0.1)
    with col3:
        entry_price = st.number_input("Entry Price", min_value=0.01, value=100.00, step=0.01)
    with col4:
        stop_loss_price = st.number_input("Stop Loss Price", min_value=0.01, value=95.00, step=0.01)

    if st.form_submit_button("📐 Calculate", type="primary"):
        result = calculate_position_size(capital, risk_pct, entry_price, stop_loss_price)

        if "error" in result:
            st.error(result["error"])
        else:
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("📊 Shares to Buy", f"{result['shares']}")
            with c2:
                st.metric("💰 Total Cost", format_currency(result["total_cost"]))
            with c3:
                st.metric("⚠️ Risk Amount", format_currency(result["risk_amount"]))
            with c4:
                st.metric("📏 Position Size", f"{result['position_pct']}%")

            st.info(f"💡 You are risking **{format_currency(result['risk_amount'])}** ({risk_pct}% of capital) "
                    f"to potentially gain **{format_currency(result['shares'] * (entry_price * 1.1 - entry_price))}** "
                    f"(10% target)")

st.markdown("---")

# ── Risk-Reward Calculator ──
st.markdown("### ⚖️ Risk-Reward Ratio Calculator")

with st.form("rr_calc"):
    col1, col2, col3 = st.columns(3)
    with col1:
        rr_entry = st.number_input("Entry Price", min_value=0.01, value=100.00, step=0.01, key="rr_entry")
    with col2:
        rr_sl = st.number_input("Stop Loss", min_value=0.01, value=95.00, step=0.01, key="rr_sl")
    with col3:
        rr_target = st.number_input("Target Price", min_value=0.01, value=115.00, step=0.01, key="rr_target")

    if st.form_submit_button("⚖️ Calculate R:R", type="primary"):
        rr = risk_reward_ratio(rr_entry, rr_sl, rr_target)
        risk_amount = abs(rr_entry - rr_sl)
        reward_amount = abs(rr_target - rr_entry)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Risk-Reward Ratio", f"1:{rr}")
        with c2:
            st.metric("Risk per Share", format_currency(risk_amount))
        with c3:
            st.metric("Reward per Share", format_currency(reward_amount))

        if rr >= 2:
            st.success(f"✅ **Excellent R:R ratio of 1:{rr}** — Risking ₹1 to make ₹{rr}")
        elif rr >= 1.5:
            st.info(f"👍 **Good R:R ratio of 1:{rr}** — Acceptable for most strategies")
        elif rr >= 1:
            st.warning(f"⚠️ **Marginal R:R ratio of 1:{rr}** — Consider adjusting target")
        else:
            st.error(f"❌ **Poor R:R ratio of 1:{rr}** — Risk exceeds reward")

st.markdown("---")

# ── Kelly Criterion ──
st.markdown("### 🎰 Kelly Criterion Calculator")
st.markdown("The Kelly Criterion determines the optimal position size to maximize long-term growth.")

with st.form("kelly_calc"):
    col1, col2 = st.columns(2)
    with col1:
        win_rate = st.slider("Win Rate (%)", 10, 90, int(DEFAULT_WIN_RATE * 100)) / 100
    with col2:
        wl_ratio = st.number_input("Win/Loss Ratio", min_value=0.1, max_value=10.0, value=DEFAULT_WIN_LOSS_RATIO, step=0.1)

    if st.form_submit_button("🎰 Calculate Kelly", type="primary"):
        kelly = kelly_criterion(win_rate, wl_ratio)

        st.metric("Optimal Position Size", f"{kelly}% of capital")
        st.markdown(f"""
        **Full Kelly:** {kelly * 2:.1f}% — aggressive, high variance
        **Half Kelly (recommended):** {kelly}% — smoother growth
        **Quarter Kelly:** {kelly / 2:.1f}% — conservative, slow growth
        """)

        if kelly > 20:
            st.warning("⚠️ High Kelly percentage — consider using half or quarter Kelly for safety")
        elif kelly > 0:
            st.success(f"✅ {kelly}% is a reasonable position size")

st.markdown("---")

# ── Portfolio Risk ──
st.markdown("### 🏦 Portfolio Risk Summary")

total_capital = st.number_input("Total Capital", min_value=1000.0, value=DEFAULT_CAPITAL, step=1000.0, key="port_cap")
summary = get_portfolio_summary()

if summary["holdings"]:
    risk = portfolio_risk_metrics(summary["holdings"], total_capital)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Capital Exposure", f"{risk['exposure_pct']}%")
    with c2:
        st.metric("Risk Level", risk["risk_level"])
    with c3:
        st.metric("Total Invested", format_currency(risk["total_exposure"]))
    with c4:
        st.metric("Cash Available", format_currency(risk["cash_available"]))

    if risk["exposure_pct"] > 80:
        st.warning("⚠️ High capital exposure! Consider taking some profits or adding more capital.")
    elif risk["exposure_pct"] > 50:
        st.info("📊 Moderate exposure. You have room for strategic additions.")
    else:
        st.success("✅ Low exposure. You have significant buying power available.")
else:
    st.info("Add holdings to your portfolio to see risk metrics.")

# ── Risk Guide ──
st.markdown("---")
with st.expander("📖 Risk Management Guide"):
    st.markdown("""
    **Golden Rules of Risk Management:**

    1. **Never risk more than 2% per trade** — Protect your capital from a single bad trade
    2. **Always use stop-losses** — Define your exit before you enter
    3. **Aim for 1:2 or better R:R** — Risk ₹1 to make at least ₹2
    4. **Position sizing** — Use Kelly Criterion or fixed-fractional sizing
    5. **Diversify** — Don't put all eggs in one basket
    6. **Track your trades** — Review regularly to improve your edge

    **Position Sizing Formula:**
    `Shares = (Capital × Risk%) / (Entry - StopLoss)`

    **Kelly Criterion:**
    `f* = (bp - q) / b` where b = win/loss ratio, p = win probability, q = 1-p
    """)
