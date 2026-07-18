"""
Trade Journal — Track every trade with psychology, reasons, emotions.
Analyze patterns to find your edge.
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from utils.database import get_connection

# ── Database ──────────────────────────────────────────────────

def init_journal_db():
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_date TIMESTAMP NOT NULL,
                exit_date TIMESTAMP,
                pnl REAL,
                pnl_pct REAL,
                strategy TEXT DEFAULT '',
                entry_reason TEXT DEFAULT '',
                exit_reason TEXT DEFAULT '',
                emotion_entry TEXT DEFAULT '',
                emotion_exit TEXT DEFAULT '',
                market_regime TEXT DEFAULT '',
                pre_checklist_passed BOOLEAN DEFAULT 0,
                confidence_score INTEGER DEFAULT 0,
                lessons TEXT DEFAULT '',
                rating INTEGER DEFAULT 3,
                tags TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS daily_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                day_pnl REAL DEFAULT 0,
                num_trades INTEGER DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                emotion_rating TEXT DEFAULT '',
                market_regime TEXT DEFAULT '',
                notes TEXT DEFAULT ''
            );
        """)

init_journal_db()


# ── Pre-Trade Checklist ───────────────────────────────────────

CHECKLIST_ITEMS = [
    ("trend", "📈 Trend aligns with my trade direction?"),
    ("volume", "📊 Volume > 1.5x average?"),
    ("risk", "🛡️ Risk < 2% of capital on this trade?"),
    ("rr", "⚖️ Risk:Reward ratio >= 1:2?"),
    ("mtf", "🔄 Daily + Weekly timeframe agree?"),
    ("regime", "🌊 Strategy matches market regime?"),
    ("news", "📰 No major events/earnings this week?"),
    ("conviction", "🔥 I have a clear reason for this trade (not FOMO)?"),
]

def get_checklist():
    """Return checklist with default values."""
    return {item[0]: {"label": item[1], "checked": False} for item in CHECKLIST_ITEMS}


def pre_trade_checklist_passed(checklist):
    """Check if pre-trade essentials are met."""
    required = ["trend", "risk", "rr", "conviction"]
    for r in required:
        if r not in checklist or not checklist[r].get("checked", False):
            return False, f"Missing: {CHECKLIST_ITEMS[[i[0] for i in CHECKLIST_ITEMS].index(r)][1]}"
    return True, "All checks passed"


# ── Journal Entry ─────────────────────────────────────────────

def add_journal_entry(symbol, action, quantity, price, strategy="", reason="",
                      emotion="neutral", regime="", checklist_passed=False, confidence=50):
    """Record a new trade entry."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO trade_journal
               (symbol, action, quantity, entry_price, entry_date, strategy,
                entry_reason, emotion_entry, market_regime, pre_checklist_passed, confidence_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol.upper(), action.upper(), quantity, price, datetime.now().isoformat(),
             strategy, reason, emotion, regime, checklist_passed, confidence)
        )


def complete_journal_entry(symbol, exit_price, exit_reason="", emotion="", lessons="", rating=3):
    """Close an open journal entry."""
    with get_connection() as conn:
        entry = conn.execute(
            "SELECT * FROM trade_journal WHERE symbol = ? AND exit_date IS NULL ORDER BY entry_date DESC LIMIT 1",
            (symbol.upper(),)
        ).fetchone()

        if entry:
            pnl = (exit_price - entry["entry_price"]) * entry["quantity"] if entry["action"].upper() == "BUY" else \
                  (entry["entry_price"] - exit_price) * entry["quantity"]
            pnl_pct = ((exit_price - entry["entry_price"]) / entry["entry_price"]) * 100 if entry["action"].upper() == "BUY" else \
                      ((entry["entry_price"] - exit_price) / entry["entry_price"]) * 100

            conn.execute(
                """UPDATE trade_journal SET
                   exit_price = ?, exit_date = ?, pnl = ?, pnl_pct = ?,
                   exit_reason = ?, emotion_exit = ?, lessons = ?, rating = ?
                   WHERE id = ?""",
                (exit_price, datetime.now().isoformat(), round(pnl, 2), round(pnl_pct, 2),
                 exit_reason, emotion, lessons, rating, entry["id"])
            )
            return {"success": True, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2)}
    return {"error": "No open entry found"}


# ── Daily Performance ─────────────────────────────────────────

def update_daily_performance(pnl, num_trades, max_dd, emotion="", regime="", notes=""):
    """Update or create daily performance record."""
    today = datetime.now().strftime("%Y-%m-%d")
    with get_connection() as conn:
        existing = conn.execute("SELECT id FROM daily_performance WHERE date = ?", (today,)).fetchone()
        if existing:
            conn.execute(
                """UPDATE daily_performance SET
                   day_pnl = day_pnl + ?, num_trades = num_trades + ?,
                   max_drawdown = MAX(max_drawdown, ?),
                   emotion_rating = ?, market_regime = ?, notes = ?
                   WHERE id = ?""",
                (pnl, num_trades, max_dd, emotion, regime, notes, existing["id"])
            )
        else:
            conn.execute(
                "INSERT INTO daily_performance (date, day_pnl, num_trades, max_drawdown, emotion_rating, market_regime, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (today, pnl, num_trades, max_dd, emotion, regime, notes)
            )


def get_today_performance():
    """Get today's P&L and trade count."""
    today = datetime.now().strftime("%Y-%m-%d")
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM daily_performance WHERE date = ?", (today,)).fetchone()
        return dict(row) if row else {"day_pnl": 0, "num_trades": 0, "max_drawdown": 0, "emotion_rating": "", "market_regime": ""}


def get_trade_journal(limit=50):
    """Get journal entries."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM trade_journal ORDER BY entry_date DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_journal_stats():
    """Get statistics from the trade journal."""
    trades = get_trade_journal(limit=500)
    if not trades:
        return {"total": 0, "win_rate": 0, "avg_pnl": 0, "best_emotion": "", "worst_emotion": ""}

    completed = [t for t in trades if t["exit_price"] is not None]
    if not completed:
        return {"total": len(trades), "win_rate": 0, "avg_pnl": 0}

    winners = [t for t in completed if t["pnl"] and t["pnl"] > 0]
    losers = [t for t in completed if t["pnl"] and t["pnl"] <= 0]
    win_rate = (len(winners) / len(completed)) * 100

    # Analyze by emotion
    emotion_pnl = {}
    for t in completed:
        for emo in ["emotion_entry", "emotion_exit"]:
            e = t.get(emo, "")
            p = t.get("pnl", 0)
            if e:
                if e not in emotion_pnl:
                    emotion_pnl[e] = {"trades": 0, "pnl": 0}
                emotion_pnl[e]["trades"] += 1
                emotion_pnl[e]["pnl"] += p

    best_emotion = max(emotion_pnl, key=lambda k: emotion_pnl[k]["pnl"]) if emotion_pnl else ""
    worst_emotion = min(emotion_pnl, key=lambda k: emotion_pnl[k]["pnl"]) if emotion_pnl else ""

    avg_pnl = sum(t.get("pnl", 0) for t in completed) / len(completed)

    return {
        "total": len(trades),
        "completed": len(completed),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(win_rate, 1),
        "avg_pnl": round(avg_pnl, 2),
        "best_emotion": best_emotion,
        "worst_emotion": worst_emotion,
        "emotion_pnl": emotion_pnl,
    }
