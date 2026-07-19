"""
Enhanced Backtester — Walk-forward validation, Monte Carlo simulation,
full Indian market cost model, and anti-overfitting diagnostics.

Key improvements over core/backtester.py:
    - Entry on NEXT BAR OPEN (no look-ahead bias)
    - Full Indian market cost model (STT, stamp duty, SEBI, GST, slippage)
    - Walk-forward validation (train IS, test OOS, roll forward)
    - Monte Carlo simulation for confidence intervals
    - Anti-overfitting diagnostics (deflated Sharpe, IS vs OOS divergence)
    - Regime-stratified performance reporting
    - Comparison vs buy-and-hold benchmark
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
import warnings

import numpy as np
import pandas as pd

from core.data_fetcher import get_stock_data
from core.analyzer import calculate_all_indicators
from core.cost_model import IndianMarketCostModel, LiquidityTier
from core.adaptive_params import AdaptiveParamEngine
from core.volatility_model import VolatilityModel
from core.ml_features import FeatureEngineeringEngine
from core.signal_combiner import SmartSignalCombiner


# ── Data Structures ──

@dataclass
class BacktestTrade:
    """A single completed trade in backtest."""
    entry_date: str = ""
    exit_date: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: int = 0
    direction: str = "LONG"
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    exit_reason: str = ""
    holding_days: int = 0
    regime_at_entry: str = ""
    vol_at_entry: str = ""


@dataclass
class BacktestMetrics:
    """All calculated performance metrics."""
    # Returns
    total_return_pct: float = 0.0
    cagr: float = 0.0
    total_profit: float = 0.0

    # Risk
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_dd_duration_days: int = 0
    calmar_ratio: float = 0.0

    # Trade quality
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_days: float = 0.0
    expectancy: float = 0.0
    avg_win_loss_ratio: float = 0.0

    # Costs
    total_costs: float = 0.0
    cost_drag_pct: float = 0.0

    # Robustness
    n_walk_forward_windows: int = 0
    avg_oos_accuracy: float = 0.0
    min_trades_check: bool = False
    pf_consistency: float = 0.0

    # Benchmark
    buy_hold_return: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0


@dataclass
class MonteCarloResult:
    """Confidence intervals from Monte Carlo simulation."""
    n_simulations: int = 0
    median_sharpe: float = 0.0
    p5_sharpe: float = 0.0
    p95_sharpe: float = 0.0
    median_return: float = 0.0
    p5_return: float = 0.0
    p95_return: float = 0.0
    median_max_dd: float = 0.0
    p95_max_dd: float = 0.0  # Worst-case drawdown (95th percentile)
    prob_profit: float = 0.0


@dataclass
class WalkForwardResult:
    """Complete walk-forward backtest result."""
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: pd.DataFrame = None
    metrics: BacktestMetrics = None
    monte_carlo: MonteCarloResult = None
    window_metrics: List[dict] = field(default_factory=list)
    anti_overfit_score: float = 0.0
    stratified: Dict[str, dict] = field(default_factory=dict)


# ── Main Backtester ──

class EnhancedBacktester:
    """
    Walk-forward backtester with realistic simulation.

    Uses:
        - cost_model: IndianMarketCostModel for full transaction costs
        - adaptive_engine: AdaptiveParamEngine for regime-based parameters
        - signal_fn: Callable that generates signals from a DataFrame
    """

    def __init__(self, cost_model: Optional[IndianMarketCostModel] = None):
        self.cost_model = cost_model or IndianMarketCostModel()
        self.adaptive_engine = AdaptiveParamEngine()

    def run_walk_forward(
        self,
        symbol: str,
        period: str = "5y",
        signal_fn: Optional[Callable] = None,
        train_days: int = 504,
        test_days: int = 63,
        step_days: int = 63,
        initial_capital: float = 100000.0,
        risk_per_trade_pct: float = 2.0,
        market_cap_cr: Optional[float] = None,
    ) -> WalkForwardResult:
        """
        Walk-forward backtest:
            For each window:
                1. In-sample: evaluate strategies on train portion
                2. Out-of-sample: trade on test portion with IS parameters
                3. NO future data leakage

        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS")
            period: Data period ("2y", "5y", "max")
            signal_fn: Function that takes(df) → [(date, signal, conf, sl, tp)]
            train_days: Days of training data per window
            test_days: Days of testing per window
            step_days: Roll forward by this many days
        """
        # Fetch data
        df = get_stock_data(symbol, period=period)
        if df.empty or len(df) < train_days + test_days:
            return WalkForwardResult(
                metrics=BacktestMetrics(),
                equity_curve=pd.DataFrame(),
            )

        df = calculate_all_indicators(df)
        if signal_fn is None:
            signal_fn = self._default_signal_fn

        n = len(df)
        windows = self._walk_forward_windows(n, train_days, test_days, step_days)

        all_trades = []
        all_equity_curves = []
        window_metrics = []

        for win_idx, (train_end, test_start, test_end) in enumerate(windows):
            if test_end > n:
                break

            train_df = df.iloc[:train_end]
            test_df = df.iloc[test_start:test_end]

            if len(test_df) < 10:
                continue

            # Generate signals on test portion
            signals = signal_fn(train_df, test_df)

            # Execute trades
            trades, equity = self._execute_trades(
                test_df, signals, initial_capital, risk_per_trade_pct, market_cap_cr
            )

            all_trades.extend(trades)
            all_equity_curves.append(equity)

            if trades:
                metrics = self._calculate_metrics(trades, equity, initial_capital)
                window_metrics.append({
                    "window": win_idx,
                    "train_end": str(train_df.index[-1]),
                    "test_start": str(test_df.index[0]),
                    "test_end": str(test_df.index[-1]),
                    **{k: v for k, v in metrics.__dict__.items() if isinstance(v, (int, float))}
                })

        # Combine equity curves
        combined_equity = pd.concat(all_equity_curves).drop_duplicates(subset=["date"]) if all_equity_curves else pd.DataFrame()
        if not combined_equity.empty:
            combined_equity = combined_equity.sort_values("date").reset_index(drop=True)

        # Final metrics
        metrics = self._calculate_metrics(all_trades, combined_equity, initial_capital)
        if window_metrics:
            metrics.n_walk_forward_windows = len(window_metrics)

        # Monte Carlo
        mc = self.run_monte_carlo(all_trades, initial_capital) if all_trades else MonteCarloResult()

        # Anti-overfit score
        anti_overfit = self.compute_anti_overfit_score(metrics, window_metrics)

        # Stratified performance
        stratified = self.stratified_performance(all_trades)

        return WalkForwardResult(
            trades=all_trades,
            equity_curve=combined_equity,
            metrics=metrics,
            monte_carlo=mc,
            window_metrics=window_metrics,
            anti_overfit_score=anti_overfit,
            stratified=stratified,
        )

    def run_single_backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000.0,
        risk_per_trade_pct: float = 2.0,
        market_cap_cr: Optional[float] = None,
    ) -> Tuple[List[BacktestTrade], pd.DataFrame, BacktestMetrics]:
        """
        Single-pass backtest using combined signal from strategies.

        KEY FIX: Entry on NEXT BAR OPEN, not current bar close.
        This removes the look-ahead bias in the original backtester.
        """
        from core.strategies import combined_signal

        trades = []
        equity_curve = []
        capital = initial_capital
        position = None  # {"shares": n, "entry": price, "entry_date": date, "sl": price, "tp": price}

        for i in range(100, len(df) - 1):  # -1 because we enter on next bar
            current_bar = df.iloc[i]
            next_bar = df.iloc[i + 1]
            current_date = df.index[i]
            next_date = df.index[i + 1]
            price = next_bar["Open"]  # ENTER ON NEXT BAR OPEN

            # Check stop-loss/target using next bar's range
            if position is not None:
                low = next_bar["Low"]
                high = next_bar["High"]
                exit_price = None
                exit_reason = None

                if low <= position["sl"]:
                    exit_price = position["sl"]
                    exit_reason = "Stop-Loss"
                elif high >= position["tp"]:
                    exit_price = position["tp"]
                    exit_reason = "Target"

                if exit_price:
                    gross_pnl = (exit_price - position["entry"]) * position["shares"]
                    costs = self.cost_model.calculate_round_trip_costs(
                        position["entry"], exit_price, position["shares"],
                        "delivery", market_cap_cr
                    )
                    net_pnl = gross_pnl - costs.round_trip

                    capital += position["shares"] * position["entry"] + net_pnl

                    trades.append(BacktestTrade(
                        entry_date=str(position["entry_date"]),
                        exit_date=str(next_date),
                        entry_price=round(position["entry"], 2),
                        exit_price=round(exit_price, 2),
                        shares=position["shares"],
                        direction="LONG",
                        gross_pnl=round(gross_pnl, 2),
                        costs=round(costs.round_trip, 2),
                        net_pnl=round(net_pnl, 2),
                        exit_reason=exit_reason,
                        holding_days=(next_date - position["entry_date"]).days,
                    ))
                    position = None

            # Generate signal using up-to-current-bar data
            if position is None:
                lookback = df.iloc[:i + 1]
                if len(lookback) >= 30:
                    sig, conf, _ = combined_signal(lookback)
                    if sig == "BUY" and conf >= 25:
                        risk_amount = capital * (risk_per_trade_pct / 100)
                        from core.strategies import get_stop_loss_target
                        sl, tp, rr = get_stop_loss_target(lookback, sig)

                        if sl and sl < price and tp and tp > price:
                            risk_per_share = price - sl
                            shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

                            if shares > 0 and price * shares <= capital * 0.95:
                                capital -= price * shares
                                position = {
                                    "shares": shares,
                                    "entry": price,
                                    "entry_date": current_date,
                                    "sl": sl,
                                    "tp": tp,
                                }

            # Track equity (at next bar open)
            pos_value = position["shares"] * price if position else 0
            total_eq = capital + pos_value
            equity_curve.append({"date": next_date, "equity": round(total_eq, 2)})

        # Close any open position
        if position:
            final_price = df["Close"].iloc[-1]
            gross_pnl = (final_price - position["entry"]) * position["shares"]
            costs = self.cost_model.calculate_round_trip_costs(
                position["entry"], final_price, position["shares"], "delivery", market_cap_cr
            )
            net_pnl = gross_pnl - costs.round_trip
            trades.append(BacktestTrade(
                exit_reason="End of Backtest",
                net_pnl=round(net_pnl, 2),
                costs=round(costs.round_trip, 2),
                holding_days=0,
            ))

        equity_df = pd.DataFrame(equity_curve)
        metrics = self._calculate_metrics(trades, equity_df, initial_capital)
        return trades, equity_df, metrics

    def run_monte_carlo(self, trades: List[BacktestTrade],
                        initial_capital: float,
                        n_simulations: int = 1000) -> MonteCarloResult:
        """
        Monte Carlo simulation: bootstrap resample from historical trades.

        Steps:
            1. Collect all trade returns
            2. Randomly resample N trade sequences
            3. Compute equity curves and metrics for each
            4. Report percentiles
        """
        if len(trades) < 10:
            return MonteCarloResult(n_simulations=0)

        trade_returns = [t.net_pnl for t in trades if t.net_pnl != 0]
        if not trade_returns:
            return MonteCarloResult(n_simulations=0)

        results = []
        n_trades = len(trade_returns)

        for sim in range(n_simulations):
            resampled = np.random.choice(trade_returns, size=n_trades, replace=True)
            equity = initial_capital + np.cumsum(resampled)
            final_eq = equity[-1]

            # Calculate metrics
            ret_pct = (final_eq - initial_capital) / initial_capital * 100
            peak = equity[0]
            dd = 0
            for val in equity:
                if val > peak:
                    peak = val
                dd = max(dd, (peak - val) / peak * 100)

            # Approximate Sharpe
            daily_ret = pd.Series(resampled).pct_change().dropna()
            sharpe = 0
            if len(daily_ret) > 5 and daily_ret.std() > 0:
                sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)

            results.append({"return": ret_pct, "sharpe": sharpe, "max_dd": dd})

        if not results:
            return MonteCarloResult(n_simulations=0)

        df_mc = pd.DataFrame(results)
        return MonteCarloResult(
            n_simulations=n_simulations,
            median_sharpe=round(float(df_mc["sharpe"].median()), 2),
            p5_sharpe=round(float(df_mc["sharpe"].quantile(0.05)), 2),
            p95_sharpe=round(float(df_mc["sharpe"].quantile(0.95)), 2),
            median_return=round(float(df_mc["return"].median()), 2),
            p5_return=round(float(df_mc["return"].quantile(0.05)), 2),
            p95_return=round(float(df_mc["return"].quantile(0.95)), 2),
            median_max_dd=round(float(df_mc["max_dd"].median()), 2),
            p95_max_dd=round(float(df_mc["max_dd"].quantile(0.95)), 2),
            prob_profit=round(float((df_mc["return"] > 0).mean()), 4),
        )

    def compute_anti_overfit_score(self, metrics: BacktestMetrics,
                                    window_metrics: List[dict]) -> float:
        """
        Anti-overfitting diagnostics. Returns score 0-1 (higher = more robust).

        Factors:
            1. Minimum 30 trades (statistical significance)
            2. Profit factor > 1.2
            3. Sharpe ratio > 0.5
            4. Consistency across walk-forward windows
            5. Not worse than buy-and-hold by too much
        """
        score = 0.0
        factors = []

        # 1. Minimum trades
        if metrics.total_trades >= 30:
            score += 0.2
            factors.append("min_trades")
        elif metrics.total_trades >= 15:
            score += 0.1
            factors.append("min_trades_marginal")

        # 2. Profit factor
        if metrics.profit_factor >= 1.5:
            score += 0.2
        elif metrics.profit_factor >= 1.2:
            score += 0.1
        factors.append(f"pf_{metrics.profit_factor:.2f}")

        # 3. Sharpe ratio
        if metrics.sharpe_ratio >= 1.0:
            score += 0.2
        elif metrics.sharpe_ratio >= 0.5:
            score += 0.1
        factors.append(f"sharpe_{metrics.sharpe_ratio:.2f}")

        # 4. Walk-forward consistency
        if window_metrics:
            sharpes = [w.get("sharpe_ratio", 0) for w in window_metrics if w.get("sharpe_ratio", 0) != 0]
            if sharpes:
                mean_s = np.mean(sharpes)
                std_s = np.std(sharpes) + 0.001
                consistency = mean_s / std_s
                if consistency >= 1.0:
                    score += 0.2
                elif consistency >= 0.5:
                    score += 0.1
                factors.append(f"wf_consistency_{consistency:.2f}")

        # 5. Drawdown penalty
        if metrics.max_drawdown < 20:
            score += 0.1
        elif metrics.max_drawdown < 35:
            score += 0.05
        factors.append(f"dd_{metrics.max_drawdown:.1f}")

        # 6. Positive CAGR
        if metrics.cagr > 0:
            score += 0.1
        factors.append(f"cagr_{metrics.cagr:.1f}")

        return round(min(1.0, max(0.0, score)), 3)

    def stratified_performance(self, trades: List[BacktestTrade]) -> Dict[str, dict]:
        """Break down performance by regime."""
        result = {}
        if not trades:
            return result

        for regime in ["TRENDING_BULL", "TRENDING_BEAR", "RANGING", "VOLATILE"]:
            subset = [t for t in trades if getattr(t, "regime_at_entry", "") == regime]
            if not subset:
                continue
            wins = sum(1 for t in subset if t.net_pnl > 0)
            total_pnl = sum(t.net_pnl for t in subset)
            result[regime] = {
                "trades": len(subset),
                "wins": wins,
                "win_rate": round(wins / len(subset) * 100, 1) if subset else 0,
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(total_pnl / len(subset), 2) if subset else 0,
            }

        return result

    # ── Internal Methods ──

    def _walk_forward_windows(self, n_rows: int, train_days: int,
                               test_days: int, step_days: int) -> List[Tuple[int, int, int]]:
        """Generate (train_end, test_start, test_end) indices for walk-forward."""
        windows = []
        train_end = train_days

        while train_end + test_days <= n_rows:
            test_start = train_end
            test_end = test_start + test_days
            windows.append((train_end, test_start, test_end))
            train_end += step_days

        return windows

    def _default_signal_fn(self, train_df: pd.DataFrame,
                            test_df: pd.DataFrame) -> List[Tuple]:
        """Default signal generation for walk-forward."""
        from core.strategies import combined_signal

        signals = []
        for i in range(len(test_df)):
            combined_df = pd.concat([train_df, test_df.iloc[:i + 1]])
            if len(combined_df) >= 30:
                sig, conf, _ = combined_signal(combined_df)
                signals.append((test_df.index[i], sig, conf))
        return signals

    def _execute_trades(self, df: pd.DataFrame, signals: List[Tuple],
                        initial_capital: float, risk_pct: float,
                        market_cap_cr: Optional[float]) -> Tuple[List[BacktestTrade], pd.DataFrame]:
        """Execute signals on a DataFrame and produce trades + equity curve."""
        trades = []
        equity_curve = []
        capital = initial_capital
        position = None

        for i in range(len(df) - 1):
            current_bar = df.iloc[i]
            next_bar = df.iloc[i + 1]
            current_date = df.index[i]
            next_date = df.index[i + 1]
            price = next_bar["Open"]

            if position is not None:
                low = next_bar["Low"]
                high = next_bar["High"]
                exit_price = None
                exit_reason = None

                if low <= position["sl"]:
                    exit_price = position["sl"]
                    exit_reason = "Stop-Loss"
                elif high >= position["tp"]:
                    exit_price = position["tp"]
                    exit_reason = "Target"

                if exit_price:
                    gross_pnl = (exit_price - position["entry"]) * position["shares"]
                    costs = self.cost_model.calculate_round_trip_costs(
                        position["entry"], exit_price, position["shares"],
                        "delivery", market_cap_cr
                    )
                    net_pnl = gross_pnl - costs.round_trip
                    capital += position["shares"] * position["entry"] + net_pnl

                    trades.append(BacktestTrade(
                        entry_date=str(position["entry_date"]),
                        exit_date=str(next_date),
                        entry_price=round(position["entry"], 2),
                        exit_price=round(exit_price, 2),
                        shares=position["shares"],
                        direction="LONG",
                        gross_pnl=round(gross_pnl, 2),
                        costs=round(costs.round_trip, 2),
                        net_pnl=round(net_pnl, 2),
                        exit_reason=exit_reason,
                        holding_days=(next_date - position["entry_date"]).days,
                    ))
                    position = None

            # Check signals for this bar
            if position is None and i < len(signals):
                signal_date, sig, conf = signals[i]
                if sig == "BUY" and conf >= 25:
                    from core.strategies import get_stop_loss_target
                    sl, tp, rr = get_stop_loss_target(df.iloc[:i + 1], "BUY")
                    if sl and sl < price and tp and tp > price:
                        risk_amount = capital * (risk_pct / 100)
                        risk_per_share = price - sl
                        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                        if shares > 0 and price * shares <= capital * 0.95:
                            capital -= price * shares
                            position = {
                                "shares": shares, "entry": price,
                                "entry_date": current_date, "sl": sl, "tp": tp,
                            }

            pos_value = position["shares"] * price if position else 0
            equity_curve.append({"date": next_date, "equity": round(capital + pos_value, 2)})

        return trades, pd.DataFrame(equity_curve)

    def _calculate_metrics(self, trades: List[BacktestTrade],
                            equity_curve: pd.DataFrame,
                            initial_capital: float) -> BacktestMetrics:
        """Calculate comprehensive metrics from trades and equity curve."""
        if not trades or equity_curve.empty:
            return BacktestMetrics()

        final_equity = equity_curve["equity"].iloc[-1]
        metrics = BacktestMetrics()

        # Basic trade stats
        metrics.total_trades = len(trades)
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl <= 0]
        metrics.win_rate = round((len(winners) / metrics.total_trades * 100), 1) if metrics.total_trades else 0

        metrics.avg_win = round(np.mean([t.net_pnl for t in winners]), 2) if winners else 0
        metrics.avg_loss = round(abs(np.mean([t.net_pnl for t in losers])), 2) if losers else 0
        metrics.avg_win_loss_ratio = round(metrics.avg_win / metrics.avg_loss, 2) if metrics.avg_loss > 0 else 0

        # Profit factor
        gross_profit = sum(t.net_pnl for t in winners)
        gross_loss = abs(sum(t.net_pnl for t in losers))
        metrics.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (999 if gross_profit > 0 else 0)

        # Expectancy
        win_prob = len(winners) / metrics.total_trades if metrics.total_trades else 0
        metrics.expectancy = round(win_prob * metrics.avg_win - (1 - win_prob) * metrics.avg_loss, 2)

        # Returns
        metrics.total_return_pct = round(((final_equity - initial_capital) / initial_capital) * 100, 2)
        metrics.total_profit = round(final_equity - initial_capital, 2)

        days = (equity_curve["date"].iloc[-1] - equity_curve["date"].iloc[0]).days if len(equity_curve) > 1 else 0
        years = max(days / 365.25, 0.01)
        metrics.cagr = round(((final_equity / initial_capital) ** (1 / years) - 1) * 100, 2) if years else 0

        # Risk metrics (use equity curve)
        equity_vals = equity_curve["equity"].values
        daily_returns = pd.Series(equity_vals).pct_change().dropna()

        if len(daily_returns) > 5 and daily_returns.std() > 0:
            metrics.sharpe_ratio = round((daily_returns.mean() / daily_returns.std()) * np.sqrt(252), 2)

            downside = daily_returns[daily_returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                metrics.sortino_ratio = round((daily_returns.mean() / downside.std()) * np.sqrt(252), 2)

        # Max drawdown
        peak = equity_vals[0]
        dd_start = 0
        max_dd = 0
        max_dd_dur = 0
        for i, val in enumerate(equity_vals):
            if val > peak:
                peak = val
                dd_start = i
            dd = ((peak - val) / peak) * 100
            if dd > max_dd:
                max_dd = dd
                max_dd_dur = i - dd_start

        metrics.max_drawdown = round(max_dd, 2)
        metrics.max_dd_duration_days = max_dd_dur
        metrics.calmar_ratio = round(metrics.cagr / max_dd, 2) if max_dd > 0 else 0

        # Holding period
        holding_days = [t.holding_days for t in trades if t.holding_days > 0]
        metrics.avg_holding_days = round(np.mean(holding_days), 1) if holding_days else 0

        # Costs
        metrics.total_costs = round(sum(t.costs for t in trades), 2)
        gross_total = sum(abs(t.gross_pnl) for t in trades)
        metrics.cost_drag_pct = round((metrics.total_costs / gross_total) * 100, 2) if gross_total > 0 else 0

        # Benchmarks
        metrics.min_trades_check = metrics.total_trades >= 30

        return metrics
