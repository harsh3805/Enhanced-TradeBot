"""
Advanced risk management — trailing stop-loss, daily loss limits, drawdown protection,
portfolio correlation analysis, and position-level risk scoring.
"""
import streamlit as st
import pandas as pd
import numpy as np
from core.data_fetcher import get_stock_data
from core.analyzer import calculate_all_indicators
from utils.database import get_connection


# ── Daily Loss Limit ───────────────────────────────────────────
DAILY_LOSS_LIMIT_PCT = 5.0     # Max 5% daily loss
MAX_DRAWDOWN_PCT = 15.0        # Max 15% total drawdown
MAX_POSITION_PCT = 20.0        # Max 20% in single position
MAX_CORRELATED_POSITIONS = 3   # Max 3 positions in same sector


def check_daily_loss_limit(trades_today, capital):
    """Check if daily loss limit has been breached."""
    daily_pnl = sum(t.get("pnl", 0) for t in trades_today)
    daily_pnl_pct = (daily_pnl / capital) * 100 if capital else 0

    if daily_pnl_pct < -DAILY_LOSS_LIMIT_PCT:
        return True, daily_pnl_pct, f"⛔ DAILY LOSS LIMIT BREACHED: {daily_pnl_pct:.2f}% (limit: -{DAILY_LOSS_LIMIT_PCT}%)"
    elif daily_pnl_pct < -DAILY_LOSS_LIMIT_PCT * 0.7:
        return False, daily_pnl_pct, f"⚠️ Approaching daily loss limit: {daily_pnl_pct:.2f}%"
    return False, daily_pnl_pct, f"✅ Daily P&L: {daily_pnl_pct:+.2f}%"


def check_drawdown(equity_curve, max_allowed=MAX_DRAWDOWN_PCT):
    """Check current drawdown against maximum allowed."""
    if not equity_curve or len(equity_curve) < 2:
        return False, 0

    peak = equity_curve[0]
    current = equity_curve[-1]
    for val in equity_curve:
        peak = max(peak, val)

    drawdown = ((peak - current) / peak) * 100

    if drawdown >= max_allowed:
        return True, drawdown
    return False, drawdown


# ── Trailing Stop-Loss ─────────────────────────────────────────

def calculate_trailing_stop(entry_price, current_price, highest_price, atr,
                            method="atr", trail_pct=2.0):
    """
    Calculate trailing stop-loss level.
    Methods:
    - 'atr': 2x ATR from highest price
    - 'percentage': trail_pct% below highest price
    - 'chandelier': highest price - ATR * multiplier
    """
    if method == "atr":
        trailing_sl = highest_price - 2 * atr
    elif method == "percentage":
        trailing_sl = highest_price * (1 - trail_pct / 100)
    elif "chandelier":
        # Chandelier Exit: Highest High - ATR * 3
        trailing_sl = highest_price - 3 * atr
    else:
        trailing_sl = highest_price * 0.98

    # Never let trailing stop go below entry
    trailing_sl = max(trailing_sl, entry_price * 0.95)

    return round(trailing_sl, 2)


# ── Position Risk Scoring ──────────────────────────────────────

def score_position_risk(symbol, entry_price, stop_loss, position_value, total_capital):
    """
    Score a position from 1 (low risk) to 10 (high risk).
    Based on: position size, risk/reward, distance to stop, market conditions.
    """
    score = 5  # Start neutral

    # Position size risk
    position_pct = (position_value / total_capital) * 100 if total_capital else 0
    if position_pct > MAX_POSITION_PCT:
        score += 3
    elif position_pct > 15:
        score += 2
    elif position_pct > 10:
        score += 1
    elif position_pct < 5:
        score -= 1

    # Stop-loss distance risk
    sl_distance_pct = ((entry_price - stop_loss) / entry_price) * 100 if entry_price and stop_loss else 0
    if sl_distance_pct > 5:
        score += 2  # Wide stop = more risk per trade
    elif sl_distance_pct < 1:
        score += 1  # Too tight = likely to get stopped out

    # Volatility risk (check ATR)
    try:
        df = get_stock_data(symbol, period="1mo")
        if not df.empty and len(df) > 14:
            df = calculate_all_indicators(df)
            atr = df["ATR"].iloc[-1]
            atr_pct = (atr / df["Close"].iloc[-1]) * 100
            if atr_pct > 3:
                score += 1  # High volatility
            elif atr_pct < 0.5:
                score -= 1  # Low volatility
    except Exception:
        pass

    return max(1, min(10, score))


def get_risk_label(score):
    """Convert risk score to human-readable label."""
    if score <= 3:
        return "🟢 LOW"
    elif score <= 5:
        return "🟡 MODERATE"
    elif score <= 7:
        return "🟠 HIGH"
    else:
        return "🔴 VERY HIGH"


# ── Portfolio-Level Risk ───────────────────────────────────────

def portfolio_correlation_check(symbols):
    """
    Check if portfolio positions are too correlated.
    Returns: correlation matrix and warnings.
    """
    if len(symbols) < 2:
        return None, []

    prices = {}
    for sym in symbols:
        try:
            df = get_stock_data(sym, period="3mo")
            if not df.empty:
                prices[sym] = df["Close"]
        except Exception:
            continue

    if len(prices) < 2:
        return None, []

    price_df = pd.DataFrame(prices)
    returns = price_df.pct_change().dropna()
    corr_matrix = returns.corr()

    warnings = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            if symbols[i] in corr_matrix.columns and symbols[j] in corr_matrix.columns:
                corr = corr_matrix.loc[symbols[i], symbols[j]]
                if corr > 0.8:
                    warnings.append(f"⚠️ {symbols[i]} and {symbols[j]} are highly correlated ({corr:.2f})")

    return corr_matrix, warnings


def calculate_portfolio_var(positions, confidence=0.95):
    """
    Calculate Value at Risk for the portfolio.
    Uses historical simulation method.
    """
    if not positions:
        return 0

    returns = []
    weights = []
    total_value = sum(p.get("value", 0) for p in positions)

    for pos in positions:
        sym = pos.get("symbol", "")
        value = pos.get("value", 0)
        weight = value / total_value if total_value else 0

        try:
            df = get_stock_data(sym, period="3mo")
            if not df.empty and len(df) > 30:
                daily_ret = df["Close"].pct_change().dropna()
                returns.append(daily_ret.values[-30:])  # Last 30 days
                weights.append(weight)
        except Exception:
            continue

    if not returns:
        return 0

    # Align lengths
    min_len = min(len(r) for r in returns)
    returns_arr = np.array([r[-min_len:] for r in returns])
    weights_arr = np.array(weights[:len(returns_arr)])

    # Portfolio returns
    portfolio_returns = np.sum(returns_arr * weights_arr[:, np.newaxis], axis=0)

    if len(portfolio_returns) < 5:
        return 0

    # Historical VaR
    var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
    return round(var * 100, 2)


# ── Smart Position Sizing ──────────────────────────────────────

def smart_position_size(capital, risk_pct, entry, stop_loss, symbol, max_positions=5, current_positions=0):
    """
    Enhanced position sizing that considers:
    - Current number of positions
    - Correlation with existing positions
    - Recent volatility
    """
    # Base position size
    risk_amount = capital * (risk_pct / 100)
    risk_per_share = abs(entry - stop_loss) if stop_loss and entry else entry * 0.02

    if risk_per_share <= 0:
        return 0

    base_shares = int(risk_amount / risk_per_share)

    # Adjust for number of positions (reduce size as we add more)
    position_factor = max(0.5, 1 - (current_positions / max_positions) * 0.3)
    adjusted_shares = int(base_shares * position_factor)

    # Adjust for volatility
    try:
        df = get_stock_data(symbol, period="1mo")
        if not df.empty and len(df) > 14:
            df = calculate_all_indicators(df)
            atr = df["ATR"].iloc[-1]
            atr_pct = (atr / entry) * 100
            if atr_pct > 3:
                adjusted_shares = int(adjusted_shares * 0.7)  # Reduce in high volatility
            elif atr_pct < 1:
                adjusted_shares = int(adjusted_shares * 1.1)  # Slightly more in low volatility
    except Exception:
        pass

    return max(1, adjusted_shares)
