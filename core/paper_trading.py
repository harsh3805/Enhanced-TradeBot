"""
Paper Trading system — virtual capital, simulated trades, journal, performance tracking.
Uses real-time prices but fake money. Track performance before going live.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.data_fetcher import get_stock_data, get_stock_info
from core.analyzer import calculate_all_indicators, get_latest_indicators
from core.strategies import combined_signal, get_stop_loss_target
from utils.database import get_connection


# ── Paper Trading Database ─────────────────────────────────────
def init_paper_db():
    """Initialize paper trading tables."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS paper_portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                entry_date TIMESTAMP NOT NULL,
                stop_loss REAL DEFAULT 0,
                target_price REAL DEFAULT 0,
                strategy TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                status TEXT DEFAULT 'OPEN'
            );

            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                entry_date TIMESTAMP NOT NULL,
                exit_date TIMESTAMP NOT NULL,
                pnl REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,
                strategy TEXT DEFAULT '',
                exit_reason TEXT DEFAULT '',
                notes TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS paper_account (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                initial_capital REAL NOT NULL DEFAULT 100000,
                current_capital REAL NOT NULL DEFAULT 100000,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Initialize account if empty
        existing = conn.execute("SELECT COUNT(*) as cnt FROM paper_account").fetchone()["cnt"]
        if existing == 0:
            conn.execute("INSERT INTO paper_account (initial_capital, current_capital) VALUES (100000, 100000)")


init_paper_db()


def get_account():
    """Get paper trading account info."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
        return dict(row) if row else {"initial_capital": 100000, "current_capital": 100000}


def reset_account(initial_capital=100000):
    """Reset paper trading account to starting capital."""
    with get_connection() as conn:
        conn.execute("DELETE FROM paper_portfolio WHERE status = 'OPEN'")
        conn.execute("DELETE FROM paper_trades")
        conn.execute("UPDATE paper_account SET current_capital = ?, initial_capital = ? WHERE id = 1",
                     (initial_capital, initial_capital))


def get_open_positions():
    """Get all open paper trading positions."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM paper_portfolio WHERE status = 'OPEN' ORDER BY entry_date DESC").fetchall()
        return [dict(r) for r in rows]


def get_trade_journal():
    """Get completed paper trades."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM paper_trades ORDER BY exit_date DESC").fetchall()
        return [dict(r) for r in rows]


def paper_buy(symbol, quantity, price, stop_loss=0, target_price=0, strategy="", notes=""):
    """Execute a paper buy order."""
    account = get_account()
    cost = quantity * price
    commission = cost * 0.001  # 0.1% commission

    if cost + commission > account["current_capital"]:
        return {"error": f"Insufficient capital. Need ₹{cost + commission:,.2f}, have ₹{account['current_capital']:,.2f}"}

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO paper_portfolio (symbol, quantity, entry_price, entry_date, stop_loss, target_price, strategy, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol.upper(), quantity, price, datetime.now().isoformat(), stop_loss, target_price, strategy, notes)
        )
        new_capital = account["current_capital"] - cost - commission
        conn.execute("UPDATE paper_account SET current_capital = ? WHERE id = 1", (round(new_capital, 2),))

    return {"success": True, "message": f"Bought {quantity} shares of {symbol} at ₹{price:.2f}"}


def paper_sell(position_id, price, reason="Manual Close", notes=""):
    """Execute a paper sell order for an open position."""
    with get_connection() as conn:
        pos = conn.execute("SELECT * FROM paper_portfolio WHERE id = ?", (position_id,)).fetchone()
        if not pos:
            return {"error": "Position not found"}
        pos = dict(pos)

        gross_pnl = (price - pos["entry_price"]) * pos["quantity"]
        commission = (pos["entry_price"] * pos["quantity"] + price * pos["quantity"]) * 0.001
        net_pnl = gross_pnl - commission
        pnl_pct = ((price - pos["entry_price"]) / pos["entry_price"]) * 100

        # Record trade
        conn.execute(
            """INSERT INTO paper_trades (symbol, quantity, entry_price, exit_price, entry_date, exit_date, pnl, pnl_pct, strategy, exit_reason, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pos["symbol"], pos["quantity"], pos["entry_price"], price,
             pos["entry_date"], datetime.now().isoformat(), round(net_pnl, 2),
             round(pnl_pct, 2), pos["strategy"], reason, notes)
        )

        # Close position
        conn.execute("UPDATE paper_portfolio SET status = 'CLOSED' WHERE id = ?", (position_id,))

        # Update capital
        account = get_account()
        proceeds = pos["quantity"] * price - commission
        new_capital = account["current_capital"] + proceeds
        conn.execute("UPDATE paper_account SET current_capital = ? WHERE id = 1", (round(new_capital, 2),))

    emoji = "🟢" if net_pnl > 0 else "🔴"
    return {"success": True, "message": f"{emoji} Sold {pos['quantity']} shares at ₹{price:.2f} | P&L: ₹{net_pnl:+,.2f} ({pnl_pct:+.1f}%)"}


def check_paper_stops(current_prices):
    """Check if any open positions have hit their stop-loss or target."""
    positions = get_open_positions()
    actions = []

    for pos in positions:
        sym = pos["symbol"]
        if sym not in current_prices:
            continue

        current_price = current_prices[sym]

        # Stop-loss hit
        if pos["stop_loss"] > 0 and current_price <= pos["stop_loss"]:
            result = paper_sell(pos["id"], pos["stop_loss"], reason="Stop-Loss Hit")
            actions.append({"position": pos, "action": "SELL", "reason": "Stop-Loss", "result": result})

        # Target hit
        elif pos["target_price"] > 0 and current_price >= pos["target_price"]:
            result = paper_sell(pos["id"], pos["target_price"], reason="Target Hit")
            actions.append({"position": pos, "action": "SELL", "reason": "Target", "result": result})

    return actions


def get_paper_performance():
    """Calculate overall paper trading performance metrics."""
    account = get_account()
    trades = get_trade_journal()
    positions = get_open_positions()

    if not trades and not positions:
        return {
            "total_trades": 0, "win_rate": 0, "total_pnl": 0,
            "roi_pct": 0, "sharpe": 0, "max_drawdown": 0,
        }

    total_trades = len(trades)
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]
    win_rate = (len(winners) / total_trades) * 100 if total_trades else 0

    total_pnl = sum(t["pnl"] for t in trades)
    avg_win = np.mean([t["pnl"] for t in winners]) if winners else 0
    avg_loss = abs(np.mean([t["pnl"] for t in losers])) if losers else 0
    profit_factor = (sum(t["pnl"] for t in winners) / abs(sum(t["pnl"] for t in losers))) if losers and sum(t["pnl"] for t in losers) != 0 else 999

    roi_pct = ((account["current_capital"] - account["initial_capital"]) / account["initial_capital"]) * 100

    # Max drawdown from equity curve of closed trades
    equity = [account["initial_capital"]]
    for t in trades:
        equity.append(equity[-1] + t["pnl"])
    peak = equity[0]
    max_dd = 0
    for val in equity:
        peak = max(peak, val)
        dd = ((peak - val) / peak) * 100
        max_dd = max(max_dd, dd)

    return {
        "total_trades": total_trades,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(roi_pct, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_dd, 2),
        "current_capital": round(account["current_capital"], 2),
        "initial_capital": account["initial_capital"],
    }
