"""
Risk management calculations — position sizing, stop-loss, Kelly criterion, VaR.
"""
import math
import numpy as np
from utils.config import DEFAULT_RISK_PER_TRADE, DEFAULT_CAPITAL, DEFAULT_WIN_RATE, DEFAULT_WIN_LOSS_RATIO


def kelly_criterion(win_rate, win_loss_ratio):
    """
    Calculate optimal position size using Kelly Criterion.
    f* = (bp - q) / b
    where b = win/loss ratio, p = win rate, q = 1 - p
    Returns: optimal fraction as percentage
    """
    if win_loss_ratio <= 0 or win_rate <= 0:
        return 0

    b = win_loss_ratio
    p = win_rate
    q = 1 - p

    kelly = (b * p - q) / b
    # Half-Kelly is more practical (less volatile)
    half_kelly = kelly / 2

    return max(0, min(50, round(half_kelly * 100, 2)))


def calculate_position_size(capital, risk_pct, entry_price, stop_loss):
    """
    Calculate how many shares to buy based on risk management.
    capital: total trading capital
    risk_pct: max % of capital to risk per trade
    entry_price: planned entry price
    stop_loss: planned stop-loss price
    Returns: dict with shares, risk_amount, risk_per_share, total_cost
    """
    if entry_price <= 0 or stop_loss <= 0 or capital <= 0:
        return {"shares": 0, "error": "Invalid inputs"}

    risk_amount = capital * (risk_pct / 100)
    risk_per_share = abs(entry_price - stop_loss)

    if risk_per_share == 0:
        return {"shares": 0, "error": "Entry and stop-loss are the same"}

    shares = int(risk_amount / risk_per_share)
    total_cost = shares * entry_price
    position_pct = (total_cost / capital) * 100

    return {
        "shares": shares,
        "risk_amount": round(risk_amount, 2),
        "risk_per_share": round(risk_per_share, 2),
        "total_cost": round(total_cost, 2),
        "position_pct": round(position_pct, 2),
    }


def risk_reward_ratio(entry, stop_loss, target):
    """Calculate risk-reward ratio."""
    risk = abs(entry - stop_loss)
    reward = abs(target - entry)

    if risk == 0:
        return 0

    return round(reward / risk, 2)


def max_drawdown(equity_curve):
    """
    Calculate maximum drawdown from an equity curve (list of portfolio values).
    Returns: max drawdown as percentage
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0

    peak = equity_curve[0]
    max_dd = 0

    for value in equity_curve:
        if value > peak:
            peak = value
        dd = ((peak - value) / peak) * 100
        max_dd = max(max_dd, dd)

    return round(max_dd, 2)


def var_calculation(returns, confidence=0.95):
    """
    Calculate Value at Risk (parametric method).
    returns: list of daily returns (as decimals, e.g., 0.01 for 1%)
    confidence: confidence level (0.95 = 95%)
    Returns: VaR as percentage
    """
    if not returns or len(returns) < 5:
        return 0

    returns_arr = np.array(returns)
    mean = np.mean(returns_arr)
    std = np.std(returns_arr)

    z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z = z_scores.get(confidence, 1.645)

    var = -(mean - z * std)
    return round(var * 100, 2)


def suggested_stop_loss(entry_price, atr, signal_type="BUY"):
    """Suggest stop-loss based on ATR multiplier."""
    multiplier = 2.0
    if signal_type == "BUY":
        return round(entry_price - multiplier * atr, 2)
    return round(entry_price + multiplier * atr, 2)


def portfolio_risk_metrics(holdings, total_capital):
    """Calculate portfolio-level risk metrics."""
    if not holdings:
        return {"exposure_pct": 0, "diversification": "N/A", "risk_level": "LOW"}

    total_invested = sum(h["quantity"] * h["buy_price"] for h in holdings)
    exposure = (total_invested / total_capital) * 100 if total_capital else 0

    unique_sectors = len(set(h.get("market", "") for h in holdings))

    if exposure > 90:
        risk_level = "HIGH"
    elif exposure > 60:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "exposure_pct": round(exposure, 1),
        "diversification_score": min(10, unique_sectors),
        "risk_level": risk_level,
        "total_exposure": round(total_invested, 2),
        "cash_available": round(total_capital - total_invested, 2),
    }
