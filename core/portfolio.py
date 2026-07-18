"""
Portfolio tracking — P&L calculations, allocation analysis, and performance metrics.
"""
import pandas as pd
from utils.database import get_portfolio, get_trade_history


def get_portfolio_summary(current_prices=None):
    """
    Calculate portfolio summary metrics.
    current_prices: dict mapping symbol → current price
    Returns dict with total_invested, current_value, pnl, roi_pct, etc.
    """
    holdings = get_portfolio()
    if not holdings:
        return {
            "total_invested": 0, "current_value": 0, "pnl": 0, "roi_pct": 0,
            "holdings_count": 0, "holdings": [], "allocation": {},
        }

    total_invested = 0
    total_current = 0
    holdings_detail = []

    for h in holdings:
        invested = h["quantity"] * h["buy_price"]
        symbol = h["symbol"]
        current_price = (current_prices or {}).get(symbol, h["buy_price"])
        current_value = h["quantity"] * current_price
        pnl = current_value - invested
        pnl_pct = (pnl / invested) * 100 if invested else 0

        total_invested += invested
        total_current += current_value

        holdings_detail.append({
            **h,
            "invested": round(invested, 2),
            "current_value": round(current_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "current_price": round(current_price, 2),
        })

    pnl = total_current - total_invested
    roi_pct = (pnl / total_invested) * 100 if total_invested else 0

    # Sector allocation
    allocation = {}
    for h in holdings_detail:
        alloc_key = h.get("market", "Unknown")
        allocation[alloc_key] = allocation.get(alloc_key, 0) + h["current_value"]

    return {
        "total_invested": round(total_invested, 2),
        "current_value": round(total_current, 2),
        "pnl": round(pnl, 2),
        "roi_pct": round(roi_pct, 2),
        "holdings_count": len(holdings),
        "holdings": holdings_detail,
        "allocation": allocation,
    }


def get_trade_stats():
    """Get historical trade statistics."""
    trades = get_trade_history()
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "avg_profit": 0, "total_pnl": 0}

    total = len(trades)
    winners = [t for t in trades if t["profit_loss"] > 0]
    win_rate = (len(winners) / total) * 100 if total else 0
    avg_profit = sum(t["profit_loss_pct"] for t in trades) / total if total else 0
    total_pnl = sum(t["profit_loss"] for t in trades)

    return {
        "total_trades": total,
        "win_rate": round(win_rate, 1),
        "avg_profit": round(avg_profit, 2),
        "total_pnl": round(total_pnl, 2),
        "best_trade": max(trades, key=lambda t: t["profit_loss_pct"]) if trades else None,
        "worst_trade": min(trades, key=lambda t: t["profit_loss_pct"]) if trades else None,
    }
