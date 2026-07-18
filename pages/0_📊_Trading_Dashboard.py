"""
Page 0: AI Trading Dashboard v2 — Professional-grade with all Tiers.
Risk enforcement, market regime, MTF alignment, trade journal, checklist.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="AI Trading Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .regime-badge { padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; }
    .regime-trend { background: #00D4AA22; color: #00D4AA; border: 1px solid #00D4AA44; }
    .regime-range { background: #FFD93D22; color: #FFD93D; border: 1px solid #FFD93D44; }
    .regime-vol { background: #FF6B6B22; color: #FF6B6B; border: 1px solid #FF6B6B44; }
    .risk-alert { background: #FF475722; color: #FF4757; padding: 8px 16px; border-radius: 6px; }
    .risk-ok { background: #00D4AA22; color: #00D4AA; padding: 8px 16px; border-radius: 6px; }
    .checklist-item { padding: 6px 10px; margin: 2px 0; border-radius: 4px; }
    .checklist-pass { border-left: 3px solid #00D4AA; }
    .checklist-fail { border-left: 3px solid #FF4757; }
</style>
""", unsafe_allow_html=True)

from core.angel_one import is_configured as angel_configured, is_authenticated as angel_auth
from core.data_fetcher import get_stock_data
from core.analyzer import calculate_all_indicators, get_trend, get_latest_indicators
from core.ai_analyst import AIAnalyst
from core.market_regime import detect_regime, get_regime_bias
from core.trade_journal import (
    get_checklist, pre_trade_checklist_passed, add_journal_entry,
    complete_journal_entry, get_trade_journal, get_journal_stats,
    get_today_performance, update_daily_performance, CHECKLIST_ITEMS
)
from core.paper_trading import (
    get_account as paper_account, get_paper_performance,
    paper_buy, paper_sell, get_open_positions
)
from core.advanced_risk import check_daily_loss_limit, portfolio_correlation_check
from core.strategies import get_stop_loss_target
from utils.database import get_watchlist, add_to_watchlist, remove_from_watchlist
from utils.helpers import format_currency, format_percentage, signal_emoji
from utils.config import POPULAR_INDIAN

# ── Header ──
col_t, col_r, col_s = st.columns([2, 1, 1])
with col_t:
    st.markdown("# 📊 Professional Trading Dashboard")
    st.caption("AI analyzes 24 strategies + market regime + volume + MTF → You just execute")
with col_r:
    angel_ok = angel_configured() and angel_auth()
    st.markdown(f"🇮🇳 **Angel One:** {'✅ Live' if angel_ok else '⬜ Delayed'}")
with col_s:
    perf = get_paper_performance()
    st.markdown(f"💰 **Capital:** {format_currency(paper_account().get('current_capital', 0))}")

# ── Market Regime + Daily Risk Banner ──
regime_info = detect_regime()
bias, bias_text = get_regime_bias(regime_info)
daily = get_today_performance()
daily_pnl = daily.get("day_pnl", 0)
capital = paper_account().get("current_capital", 100000)
daily_limit_hit = daily_pnl < -capital * 0.03

regime_colors = {"TRENDING_BULL": "regime-trend", "TRENDING_BEAR": "regime-trend",
                 "RANGING": "regime-range", "VOLATILE": "regime-vol"}
rc = regime_colors.get(regime_info["regime"], "")

if daily_limit_hit:
    st.markdown(f"""
    <div class="risk-alert">🚨 DAILY LOSS LIMIT HIT: {format_currency(daily_pnl)} ({format_percentage((daily_pnl/capital)*100)})
    <br>Stop trading for the day. Review your trades. Resume tomorrow.</div>
    """, unsafe_allow_html=True)
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"<span class='regime-badge {rc}'>📊 {regime_info['regime']}</span>", unsafe_allow_html=True)
        st.caption(f"Confidence: {regime_info['confidence']}%")
    with c2: st.metric("Bias", bias, bias_text[:20])
    with c3: st.metric("Trades Today", daily.get("num_trades", 0))
    with c4: st.metric("Daily P&L", format_currency(daily_pnl),
                      f"{format_percentage((daily_pnl/capital)*100)}" if capital else "")
    with c5: st.metric("ADX", round(regime_info["details"].get("adx", 0), 1),
                      f"20d Vol: {regime_info['details'].get('volatility_ratio', 0):.1f}x")

st.markdown("---")

# ── Stock Selection ──
watchlist = get_watchlist()
watchlist_symbols = [w["symbol"].replace(".NS", "") for w in watchlist]
all_options = sorted(set([s.replace(".NS", "") for s in POPULAR_INDIAN] + watchlist_symbols))

col_sel1, col_sel2 = st.columns([3, 1])
with col_sel1:
    selected = st.selectbox("🔍 Stock", options=all_options if all_options else ["RELIANCE", "TCS", "INFY"],
                           index=all_options.index("RELIANCE") if "RELIANCE" in all_options else 0)
with col_sel2:
    st.markdown("####")
    full_sym = f"{selected}.NS"
    in_wl = any(w["symbol"] == full_sym for w in watchlist)
    if not in_wl:
        if st.button("➕ Add", use_container_width=True):
            add_to_watchlist(full_sym, "INDIA"); st.rerun()
    else:
        if st.button("➖ Remove", use_container_width=True):
            remove_from_watchlist(full_sym, "INDIA"); st.rerun()

# ── AI Analysis ──
with st.spinner(f"AI analyzing {selected} (24 strategies + regime + volume + MTF)..."):
    analyst = AIAnalyst(selected)
    result = analyst.analyze()

if "error" in result:
    st.error(result["error"]); st.stop()

decision = result["decision"]
confidence = result["confidence"]
price = result["price"]
regime = result.get("regime", {}).get("regime", "UNKNOWN")

# ── Pre-Trade Checklist (Tier 3: Psychology) ──
st.markdown("### ✅ Pre-Trade Checklist")
checklist = get_checklist()

# Auto-fill some checks from AI analysis
aligned_ok = result.get("mtf_aligned", (True, ""))
vol_ok = result.get("volume_ok", (True, ""))
near_earn = result.get("near_earnings", False)

cols_check = st.columns(4)
with cols_check[0]:
    trend = result.get("market_context", {}).get("trend", "NEUTRAL")
    is_bull = "BUY" in decision and trend in ("BULLISH", "TRENDING_BULL")
    is_bear = "SELL" in decision and trend in ("BEARISH", "TRENDING_BEAR")
    check_ok = is_bull or is_bear
    st.checkbox(f"📈 Trend ({trend})", value=check_ok, key="chk_trend", disabled=True)
with cols_check[1]:
    vol_state, vol_msg = vol_ok if isinstance(vol_ok, (list, tuple)) else (vol_ok, "")
    st.checkbox(f"📊 Volume confirmed", value=vol_state, key="chk_vol", disabled=True)
with cols_check[2]:
    rr = result.get("risk_reward", 0) or 0
    st.checkbox(f"⚖️ R:R >= 1:2 (1:{rr})", value=rr >= 2, key="chk_rr", disabled=True)
with cols_check[3]:
    st.checkbox(f"🔒 No earnings this week", value=not near_earn, key="chk_earn", disabled=True)

passed_all = (trend in ("BULLISH", "TRENDING_BULL", "BEARISH", "TRENDING_BEAR") and
              vol_state and rr >= 2 and not near_earn)
if passed_all and ("BUY" in decision or "SELL" in decision):
    st.success("✅ All checks pass — Ready to trade")
else:
    st.warning("⚠️ Not all criteria met — Exercise caution")

st.markdown("---")

# ── AI Decision Card ──
is_b = "BUY" in decision; is_s = "SELL" in decision
css = "ai-strong-buy" if decision == "STRONG BUY" else "ai-buy" if is_b else \
      "ai-strong-sell" if decision == "STRONG SELL" else "ai-sell" if is_s else "ai-hold"

st.markdown(f"""
<div class="ai-card">
    <div style="text-align:center;color:#888;font-size:0.9rem;">🤖 AI — {selected} ({regime} regime)</div>
    <div class="ai-decision {css}">{signal_emoji(decision)} {decision}<span style="font-size:1rem;opacity:0.7;"> | {confidence}% conf</span></div>
    <div style="text-align:center;color:#aaa;margin-top:0.5rem;">
        Price: <strong>{format_currency(price)}</strong>
        {' | SL: ' + format_currency(result['stop_loss']) if result['stop_loss'] else ''}
        {' | Target: ' + format_currency(result['target']) if result['target'] else ''}
        {' | R:R 1:' + str(result['risk_reward']) if result['risk_reward'] else ''}
    </div>
    <div style="text-align:center;color:#888;margin-top:0.5rem;font-size:0.85rem;">{result['reasoning']}</div>
</div>
""", unsafe_allow_html=True)

# ── Trade Buttons ──
cb1, cb2, cb3 = st.columns([1, 1, 2])
with cb1:
    if st.button(f"🟢 Paper BUY {selected}", type="primary", use_container_width=True):
        if not daily_limit_hit:
            # Enforce position sizing: risk% <= 2%
            max_risk = capital * 0.02
            risk_per_share = (price - (result.get("stop_loss") or price * 0.97))
            qty = max(1, int(max_risk / risk_per_share)) if risk_per_share > 0 else 10
            r = paper_buy(full_sym, qty, price,
                         stop_loss=result.get("stop_loss", 0) or 0,
                         target_price=result.get("target", 0) or 0,
                         strategy=f"AI {decision}", notes=result["reasoning"][:100])
            if "error" in r: st.error(r["error"])
            else:
                st.success(r["message"])
                update_daily_performance(0, 1, 0, regime=regime)
                add_journal_entry(full_sym, "BUY", qty, price, strategy=f"AI {decision}",
                                 reason=result["reasoning"][:100],
                                 emotion="confident" if confidence > 70 else "cautious",
                                 regime=regime, checklist_passed=passed_all, confidence=confidence)
        else: st.error("Daily loss limit hit — cannot trade. Stop for the day.")

with cb2:
    if st.button(f"🔴 Paper SELL {selected}", use_container_width=True):
        from core.paper_trading import get_open_positions
        positions = get_open_positions()
        for pos in positions:
            if pos["symbol"] == full_sym:
                r = paper_sell(pos["id"], price, reason="AI Signal",
                              notes=f"AI {decision} {confidence}%")
                if r.get("success"):
                    st.warning(r["message"])
                    complete_journal_entry(full_sym, price, "AI Signal", "executed")
with cb3:
    st.info(f"💰 Capital: {format_currency(capital)}")

# ── Indicator Snapshot ──
st.markdown("---")
st.markdown("### 📊 Strategy Vote Map")
votes = result.get("strategy_votes", {"BUY": 0, "SELL": 0, "HOLD": 0})
total = result.get("total_strategies", 24)
buy_pct = votes["BUY"]/total*100 if total else 0
sell_pct = votes["SELL"]/total*100 if total else 0
hold_pct = votes["HOLD"]/total*100 if total else 0

fig = go.Figure(data=[go.Bar(
    x=["BUY", "HOLD", "SELL"],
    y=[buy_pct, hold_pct, sell_pct],
    marker_color=["#00D4AA", "#FFD93D", "#FF4757"],
    text=[f"{v} ({p:.0f}%)" for v, p in zip([votes["BUY"], votes["HOLD"], votes["SELL"]],
                                              [buy_pct, hold_pct, sell_pct])],
    textposition="outside",
)])
fig.update_layout(height=300, template="plotly_dark", paper_bgcolor="#0E1117",
                  plot_bgcolor="#0E1117", showlegend=False, title=f"{total} Strategies Voting")
st.plotly_chart(fig, use_container_width=True)

# ── Chart ──
st.markdown("### 📈 Price")
df = get_stock_data(full_sym, period="6mo")
if not df.empty:
    df = calculate_all_indicators(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=selected,
        increasing_line_color="#00D4AA", decreasing_line_color="#FF4757"), row=1, col=1)
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#B388FF", width=1)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#FF4757", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00D4AA", opacity=0.5, row=2, col=1)
    fig.update_layout(height=450, template="plotly_dark", paper_bgcolor="#0E1117",
                      plot_bgcolor="#0E1117", xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Watchlist Signals ──
st.markdown("---")
st.markdown("### 👁️ Watchlist")
if watchlist:
    wl_data = []
    for w in watchlist:
        try:
            a = AIAnalyst(w["symbol"].replace(".NS", ""))
            r = a.analyze()
            if "error" not in r:
                wl_data.append({
                    "Symbol": w["symbol"].replace(".NS", ""),
                    "Price": format_currency(r["price"]),
                    "AI": r["decision"],
                    "Conf": f"{r['confidence']}%",
                    "RR": f"1:{r['risk_reward']}" if r.get("risk_reward") else "-",
                    "Regime": r.get("regime", {}).get("regime", ""),
                })
        except: pass
    if wl_data:
        st.dataframe(pd.DataFrame(wl_data), use_container_width=True, hide_index=True)
else:
    st.info("Add stocks to your watchlist to see AI signals for all.")

# ── Trade Journal (Tier 3) ──
st.markdown("---")
with st.expander("📜 Trade Journal & Psychology"):
    stats = get_journal_stats()
    if stats["total"] > 0:
        sj1, sj2, sj3, sj4 = st.columns(4)
        with sj1: st.metric("Total Trades", stats["total"])
        with sj2: st.metric("Win Rate", f"{stats['win_rate']}%", f"{stats['winners']}W/{stats['losers']}L")
        with sj3: st.metric("Avg P&L", format_currency(stats["avg_pnl"]))
        with sj4: st.metric("Best Emotion", stats["best_emotion"] or "N/A",
                           f"Worst: {stats['worst_emotion'] or 'N/A'}")

        if stats["emotion_pnl"]:
            st.markdown("#### P&L by Emotion")
            emo_df = pd.DataFrame([
                {"Emotion": e, "Trades": d["trades"], "P&L": format_currency(d["pnl"])}
                for e, d in stats["emotion_pnl"].items()
            ]).sort_values("P&L", ascending=False)
            st.dataframe(emo_df, use_container_width=True, hide_index=True)
            st.caption("💡 Green emotion = trade well. Red emotion = review your mindset.")
    else:
        st.info("Start trading to build your journal. Track emotions to find your edge.")

    journal = get_trade_journal(limit=20)
    if journal:
        st.markdown("#### Recent Trades")
        jf = pd.DataFrame([{
            "Symbol": j["symbol"].replace(".NS", ""),
            "Action": j["action"],
            "Entry": j["entry_price"],
            "Exit": j.get("exit_price", "-"),
            "P&L": format_currency(j["pnl"]) if j["pnl"] else "-",
            "Emotion": j.get("emotion_entry", ""),
            "Strategy": j.get("strategy", ""),
            "Rating": "⭐" * (j.get("rating") or 0),
        } for j in journal[:10]])
        st.dataframe(jf, use_container_width=True, hide_index=True)
