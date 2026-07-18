"""
SQLite database operations for watchlist, portfolio, alerts, and trade history.
"""
import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trading.db")


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize database tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market TEXT NOT NULL,
                sector TEXT DEFAULT '',
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT DEFAULT '',
                UNIQUE(symbol, market)
            );

            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                buy_price REAL NOT NULL,
                buy_date TIMESTAMP NOT NULL,
                stop_loss REAL DEFAULT 0,
                target_price REAL DEFAULT 0,
                notes TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                target_value REAL NOT NULL,
                is_active INTEGER DEFAULT 1,
                triggered INTEGER DEFAULT 0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                buy_price REAL NOT NULL,
                sell_price REAL NOT NULL,
                buy_date TIMESTAMP NOT NULL,
                sell_date TIMESTAMP NOT NULL,
                profit_loss REAL DEFAULT 0,
                profit_loss_pct REAL DEFAULT 0
            );
        """)


# ── Watchlist CRUD ──────────────────────────────────────────────

def add_to_watchlist(symbol, market, sector="", notes=""):
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO watchlist (symbol, market, sector, notes) VALUES (?, ?, ?, ?)",
            (symbol.upper(), market.upper(), sector, notes),
        )


def remove_from_watchlist(symbol, market):
    with get_connection() as conn:
        conn.execute("DELETE FROM watchlist WHERE symbol = ? AND market = ?", (symbol.upper(), market.upper()))


def get_watchlist():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM watchlist ORDER BY added_date DESC").fetchall()
        return [dict(r) for r in rows]


def get_watchlist_symbols():
    wl = get_watchlist()
    return [f"{r['symbol']}" for r in wl]


# ── Portfolio CRUD ──────────────────────────────────────────────

def add_holding(symbol, market, quantity, buy_price, buy_date, stop_loss=0, target_price=0, notes=""):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO portfolio (symbol, market, quantity, buy_price, buy_date, stop_loss, target_price, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol.upper(), market.upper(), quantity, buy_price, buy_date, stop_loss, target_price, notes),
        )


def remove_holding(holding_id):
    with get_connection() as conn:
        conn.execute("DELETE FROM portfolio WHERE id = ?", (holding_id,))


def get_portfolio():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM portfolio ORDER BY buy_date DESC").fetchall()
        return [dict(r) for r in rows]


# ── Alerts CRUD ────────────────────────────────────────────────

def add_alert(symbol, market, alert_type, target_value):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO alerts (symbol, market, alert_type, target_value) VALUES (?, ?, ?, ?)",
            (symbol.upper(), market.upper(), alert_type, target_value),
        )


def remove_alert(alert_id):
    with get_connection() as conn:
        conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))


def get_active_alerts():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM alerts WHERE is_active = 1 ORDER BY created_date DESC").fetchall()
        return [dict(r) for r in rows]


def get_triggered_alerts():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM alerts WHERE triggered = 1 ORDER BY created_date DESC").fetchall()
        return [dict(r) for r in rows]


def mark_alert_triggered(alert_id):
    with get_connection() as conn:
        conn.execute("UPDATE alerts SET triggered = 1, is_active = 0 WHERE id = ?", (alert_id,))


# ── Trade History ──────────────────────────────────────────────

def add_trade(symbol, market, quantity, buy_price, sell_price, buy_date, sell_date):
    pnl = (sell_price - buy_price) * quantity
    pnl_pct = ((sell_price - buy_price) / buy_price) * 100 if buy_price else 0
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO trade_history
               (symbol, market, quantity, buy_price, sell_price, buy_date, sell_date, profit_loss, profit_loss_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol.upper(), market.upper(), quantity, buy_price, sell_price, buy_date, sell_date, pnl, pnl_pct),
        )


def get_trade_history():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM trade_history ORDER BY sell_date DESC").fetchall()
        return [dict(r) for r in rows]


# Initialize on import
init_db()
