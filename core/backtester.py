"""
Backtesting engine — test strategies on historical data with realistic simulation.
Calculates: Sharpe, Sortino, Max Drawdown, CAGR, Win Rate, Profit Factor, Calmar Ratio.
Simulates: slippage, commissions, position sizing, stop-losses.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.data_fetcher import get_stock_data
from core.analyzer import calculate_all_indicators
from core.strategies import combined_signal, get_stop_loss_target
from utils.config import STRATEGY_WEIGHTS


# ── Backtesting Configuration ──────────────────────────────────
SLIPPAGE_PCT = 0.05       # 0.05% slippage per trade
COMMISSION_PCT = 0.03     # 0.03% brokerage per trade (Indian broker avg)
STT_PCT = 0.1             # Securities Transaction Tax (India)
INITIAL_CAPITAL = 100000
RISK_PER_TRADE_PCT = 2.0
POSITION_SIZING = "fixed_risk"  # "fixed_risk" or "equal_weight"


def run_backtest(symbol, period="2y", interval="1d", strategy="combined",
                 initial_capital=INITIAL_CAPITAL, risk_per_trade=RISK_PER_TRADE_PCT,
                 custom_weights=None):
    """
    Run full backtest on a stock.
    Returns: dict with trades, equity_curve, metrics, and chart.
    """
    # Fetch historical data
    df = get_stock_data(symbol, period=period, interval=interval)
    if df.empty or len(df) < 60:
        return {"error": f"Insufficient data for {symbol}. Need at least 60 bars."}

    # Calculate indicators
    df = calculate_all_indicators(df)

    # Override strategy weights if custom provided
    weights = custom_weights or STRATEGY_WEIGHTS

    # ── Simulation State ──
    capital = initial_capital
    position = None        # {"shares": n, "entry": price, "entry_date": date, "sl": price, "tp": price}
    trades = []
    equity_curve = []
    daily_returns = []
    max_capital = initial_capital

    for i in range(60, len(df)):
        current_bar = df.iloc[i]
        current_date = df.index[i]
        price = current_bar["Close"]

        # Check stop-loss / target hit on current bar
        if position is not None:
            low = current_bar["Low"]
            high = current_bar["High"]

            exit_price = None
            exit_reason = None

            # Stop-loss hit
            if low <= position["sl"]:
                exit_price = position["sl"] * (1 - SLIPPAGE_PCT / 100)
                exit_reason = "Stop-Loss"

            # Target hit
            elif high >= position["tp"]:
                exit_price = position["tp"] * (1 - SLIPPAGE_PCT / 100)
                exit_reason = "Target"

            # Trailing stop-loss: if price moved up, move SL to entry (breakeven)
            if position and price > position["entry"] * 1.02:
                position["sl"] = max(position["sl"], position["entry"])

            if exit_price:
                # Calculate P&L
                gross_pnl = (exit_price - position["entry"]) * position["shares"]
                commission = (position["entry"] * position["shares"] * COMMISSION_PCT / 100) + \
                             (exit_price * position["shares"] * COMMISSION_PCT / 100)
                stt = (exit_price * position["shares"] * STT_PCT / 100)
                net_pnl = gross_pnl - commission - stt

                capital += position["shares"] * position["entry"] + net_pnl

                trades.append({
                    "entry_date": position["entry_date"],
                    "exit_date": current_date,
                    "entry_price": round(position["entry"], 2),
                    "exit_price": round(exit_price, 2),
                    "shares": position["shares"],
                    "gross_pnl": round(gross_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                    "commission": round(commission, 2),
                    "exit_reason": exit_reason,
                    "holding_days": (current_date - position["entry_date"]).days,
                })
                position = None

        # Generate signal for entry
        if position is None:
            lookback = df.iloc[max(0, i-100):i+1]
            if len(lookback) >= 30:
                sig, conf, reason = _generate_signal(lookback, weights)

                if sig == "BUY" and conf >= 25:
                    # Position sizing: risk X% of capital
                    risk_amount = capital * (risk_per_trade / 100)
                    sl, tp, rr = get_stop_loss_target(lookback, sig)

                    if sl and sl < price and tp and tp > price:
                        risk_per_share = price - sl
                        shares = int(risk_amount / risk_per_share)
                        cost = shares * price * (1 + COMMISSION_PCT / 100 + STT_PCT / 100)

                        if cost <= capital * 0.95 and shares > 0:
                            capital -= cost
                            position = {
                                "shares": shares,
                                "entry": price,
                                "entry_date": current_date,
                                "sl": sl,
                                "tp": tp,
                            }

        # Track equity
        position_value = 0
        if position:
            position_value = position["shares"] * price
        total_equity = capital + position_value
        equity_curve.append({
            "date": current_date,
            "equity": round(total_equity, 2),
            "capital": round(capital, 2),
            "position_value": round(position_value, 2),
        })

        if len(equity_curve) > 1:
            prev_eq = equity_curve[-2]["equity"]
            daily_returns.append((total_equity - prev_eq) / prev_eq)
        max_capital = max(max_capital, total_equity)

    # Close any open position at end
    if position:
        final_price = df["Close"].iloc[-1]
        gross_pnl = (final_price - position["entry"]) * position["shares"]
        commission = (position["entry"] * position["shares"] * COMMISSION_PCT / 100) + \
                     (final_price * position["shares"] * COMMISSION_PCT / 100)
        net_pnl = gross_pnl - commission
        capital += position["shares"] * position["entry"] + net_pnl
        trades.append({
            "entry_date": position["entry_date"],
            "exit_date": df.index[-1],
            "entry_price": round(position["entry"], 2),
            "exit_price": round(final_price, 2),
            "shares": position["shares"],
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "commission": round(commission, 2),
            "exit_reason": "End of Backtest",
            "holding_days": (df.index[-1] - position["entry_date"]).days,
        })

    # ── Calculate Metrics ──
    metrics = _calculate_metrics(trades, equity_curve, daily_returns, initial_capital, symbol)

    return {
        "trades": trades,
        "equity_curve": pd.DataFrame(equity_curve),
        "metrics": metrics,
        "symbol": symbol,
        "period": period,
        "initial_capital": initial_capital,
    }


def _generate_signal(df, weights):
    """Generate combined signal with custom weights."""
    try:
        from core.strategies import rsi_strategy, macd_strategy, ma_crossover_strategy, bollinger_strategy
        strategies = {
            "rsi": rsi_strategy(df),
            "macd": macd_strategy(df),
            "ma_crossover": ma_crossover_strategy(df),
            "bollinger": bollinger_strategy(df),
        }
        scores = {"BUY": 0, "SELL": 0, "HOLD": 0}
        total_weight = sum(weights.values())
        for name, (signal, confidence, _) in strategies.items():
            w = weights.get(name, 0.25)
            scores[signal] += (confidence / 100) * w

        if scores["BUY"] > scores["SELL"] and scores["BUY"] > scores["HOLD"] and scores["BUY"] > 0.15:
            return "BUY", min(95, int((scores["BUY"] / total_weight) * 100)), ""
        elif scores["SELL"] > scores["BUY"] and scores["SELL"] > scores["HOLD"] and scores["SELL"] > 0.15:
            return "SELL", min(95, int((scores["SELL"] / total_weight) * 100)), ""
        return "HOLD", min(60, int((scores["HOLD"] / total_weight) * 100)), ""
    except Exception:
        return "HOLD", 0, ""


def _calculate_metrics(trades, equity_curve, daily_returns, initial_capital, symbol):
    """Calculate comprehensive performance metrics."""
    if not trades or not equity_curve:
        return {"error": "No trades generated"}

    equity_df = pd.DataFrame(equity_curve)
    final_equity = equity_df["equity"].iloc[-1]

    # Basic metrics
    total_trades = len(trades)
    winners = [t for t in trades if t["net_pnl"] > 0]
    losers = [t for t in trades if t["net_pnl"] <= 0]
    win_rate = (len(winners) / total_trades) * 100 if total_trades else 0

    avg_win = np.mean([t["net_pnl"] for t in winners]) if winners else 0
    avg_loss = abs(np.mean([t["net_pnl"] for t in losers])) if losers else 0.01
    profit_factor = (sum(t["net_pnl"] for t in winners) / abs(sum(t["net_pnl"] for t in losers))) if losers and sum(t["net_pnl"] for t in losers) != 0 else 999

    # Returns
    total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
    days = (equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days
    years = max(days / 365.25, 0.01)
    cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100

    # Risk metrics
    daily_returns_arr = np.array(daily_returns) if daily_returns else np.array([0])
    sharpe = (np.mean(daily_returns_arr) / np.std(daily_returns_arr)) * np.sqrt(252) if np.std(daily_returns_arr) > 0 else 0
    downside_returns = daily_returns_arr[daily_returns_arr < 0]
    sortino = (np.mean(daily_returns_arr) / np.std(downside_returns)) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0

    # Max drawdown
    equity_values = equity_df["equity"].values
    peak = equity_values[0]
    max_dd = 0
    max_dd_duration = 0
    dd_start = 0
    for i, val in enumerate(equity_values):
        if val > peak:
            peak = val
            dd_start = i
        dd = ((peak - val) / peak) * 100
        if dd > max_dd:
            max_dd = dd
            max_dd_duration = i - dd_start

    calmar = cagr / max_dd if max_dd > 0 else 999

    # Average holding period
    avg_holding = np.mean([t["holding_days"] for t in trades]) if trades else 0

    # Consecutive wins/losses
    max_consec_wins = _max_consecutive(trades, True)
    max_consec_losses = _max_consecutive(trades, False)

    return {
        "symbol": symbol,
        "total_return_pct": round(total_return_pct, 2),
        "cagr": round(cagr, 2),
        "total_trades": total_trades,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown": round(max_dd, 2),
        "max_dd_duration_days": max_dd_duration,
        "calmar_ratio": round(calmar, 2),
        "avg_holding_days": round(avg_holding, 1),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "final_equity": round(final_equity, 2),
        "total_profit": round(final_equity - initial_capital, 2),
        "total_commission": round(sum(t["commission"] for t in trades), 2),
    }


def _max_consecutive(trades, winners=True):
    """Count max consecutive wins or losses."""
    max_count = 0
    count = 0
    for t in trades:
        if (winners and t["net_pnl"] > 0) or (not winners and t["net_pnl"] <= 0):
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count


def create_backtest_chart(backtest_result):
    """Create comprehensive backtest visualization."""
    if "error" in backtest_result:
        return None

    equity_df = backtest_result["equity_curve"]
    trades = backtest_result["trades"]
    metrics = backtest_result["metrics"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=["Equity Curve", "Drawdown", "Trade P&L"],
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(x=equity_df["date"], y=equity_df["equity"], name="Equity",
                   line=dict(color="#00D4AA", width=2), fill="tozeroy",
                   fillcolor="rgba(0,212,170,0.1)"),
        row=1, col=1,
    )

    # Initial capital reference line
    fig.add_hline(y=backtest_result["initial_capital"], line_dash="dash",
                  line_color="white", opacity=0.3, row=1, col=1)

    # Drawdown
    equity_vals = equity_df["equity"].values
    peak = equity_vals[0]
    drawdowns = []
    for val in equity_vals:
        peak = max(peak, val)
        drawdowns.append(((peak - val) / peak) * -100)

    fig.add_trace(
        go.Scatter(x=equity_df["date"], y=drawdowns, name="Drawdown",
                   fill="tozeroy", line=dict(color="#FF4757", width=1),
                   fillcolor="rgba(255,71,87,0.2)"),
        row=2, col=1,
    )

    # Trade P&L bars
    trade_dates = [t["exit_date"] for t in trades]
    trade_pnls = [t["net_pnl"] for t in trades]
    colors = ["#00D4AA" if p > 0 else "#FF4757" for p in trade_pnls]

    fig.add_trace(
        go.Bar(x=trade_dates, y=trade_pnls, name="Trade P&L",
               marker_color=colors, opacity=0.7),
        row=3, col=1,
    )

    # Add buy/sell markers on equity curve
    for t in trades:
        fig.add_trace(go.Scatter(
            x=[t["entry_date"]], y=[equity_df[equity_df["date"] == t["entry_date"]]["equity"].values[0] if len(equity_df[equity_df["date"] == t["entry_date"]]) > 0 else 0],
            mode="markers", marker=dict(symbol="triangle-up", size=10, color="#00D4AA"),
            showlegend=False, hovertext=f"BUY @ {t['entry_price']}"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[t["exit_date"]], y=[equity_df[equity_df["date"] == t["exit_date"]]["equity"].values[0] if len(equity_df[equity_df["date"] == t["exit_date"]]) > 0 else 0],
            mode="markers", marker=dict(symbol="triangle-down", size=10, color="#FF4757"),
            showlegend=False, hovertext=f"SELL @ {t['exit_price']} ({t['exit_reason']})"
        ), row=1, col=1)

    fig.update_layout(
        height=800,
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def run_multi_stock_backtest(symbols, period="2y", initial_capital=100000):
    """Run backtest across multiple stocks to find the best performer."""
    results = []
    for sym in symbols:
        try:
            result = run_backtest(sym, period=period, initial_capital=initial_capital)
            if "error" not in result:
                results.append(result["metrics"])
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe_ratio", ascending=False)
    return df
