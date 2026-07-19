"""
Smart Signal Combiner — Performance-aware meta strategy combiner.

Replaces the flat weighted average in strategies.py combined_signal().
Tracks each strategy's recent performance, adjusts weights dynamically,
integrates ML predictions as a 'super strategy', and enforces
signal cooldown after losses to prevent revenge trading.

Key features:
    - Dynamic weighting by recent hit rate × profit factor
    - Regime-specific multipliers (e.g., trend strategies get 1.5× in trending markets)
    - ML prediction integration (30% base weight)
    - Signal cooldown after losses (configurable, default 3 days)
    - 5-level output: STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL
"""
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class StrategyPerformance:
    """Rolling performance stats for a single strategy."""
    name: str = ""
    hit_rate: float = 0.5           # % correct signals over lookback
    profit_factor: float = 1.0      # gross profit / gross loss
    avg_return: float = 0.0         # average return per signal
    total_signals: int = 0
    recent_signals: int = 0         # signals in current lookback window
    last_signal_date: str = ""
    last_loss_date: str = ""
    is_in_cooldown: bool = False


@dataclass
class CombinedSignal:
    """Complete combined signal output."""
    signal: str = "NEUTRAL"                         # STRONG BUY/BUY/NEUTRAL/SELL/STRONG SELL
    confidence: float = 50.0                        # 0-100 calibrated
    net_score: float = 0.0                          # Raw weighted score (-1 to +1)
    reasoning: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    ml_contribution: float = 0.0                    # How much ML influenced (0-1)
    regime_filter_applied: bool = False
    cooldown_active: bool = False
    individual_signals: Dict[str, Tuple[str, float]] = field(default_factory=dict)


# Regime-specific weight multipliers for each strategy category
REGIME_MULTIPLIERS = {
    "TRENDING_BULL": {
        "SuperTrend": 1.5, "Ichimoku": 1.4, "PSAR": 1.3,
        "MA Crossover": 1.3, "Vortex": 1.2, "KST": 1.2,
        "Stochastic": 0.5, "Williams %R": 0.6, "Keltner": 0.7,
        "Bollinger": 0.6, "CCI": 0.5,
    },
    "TRENDING_BEAR": {
        "SuperTrend": 1.4, "Ichimoku": 1.3, "PSAR": 1.3,
        "Stochastic": 0.5, "Williams %R": 0.6,
    },
    "RANGING": {
        "Stochastic": 1.5, "Williams %R": 1.4, "Bollinger": 1.3,
        "RSI": 1.2, "CCI": 1.2, "Ultimate Osc": 1.2,
        "SuperTrend": 0.5, "Ichimoku": 0.4, "PSAR": 0.5,
        "Donchian": 0.6,
    },
    "VOLATILE": {
        "Bollinger": 1.3, "Keltner": 1.4, "Donchian": 1.3,
        "ATR": 1.3, "Chandelier": 1.3,
        "Ichimoku": 0.3, "KST": 0.5,
    },
}

# Default multipliers for strategies not in the above dict
DEFAULT_REGIME_MULTIPLIER = 1.0

# Signal level thresholds
SIGNAL_THRESHOLDS = [
    ("STRONG BUY", 0.60),
    ("BUY", 0.25),
    ("NEUTRAL", -0.25),
    ("SELL", -0.60),
    ("STRONG SELL", -1.0),
]


class SmartSignalCombiner:
    """
    Performance-aware meta-learning signal combiner.

    Tracks each strategy's rolling hit rate and profit factor to dynamically
    adjust weights. Integrates ML predictions as a weighted 'super strategy'.
    """

    def __init__(
        self,
        performance_lookback: int = 50,
        cooldown_days: int = 3,
        min_strategy_signals: int = 5,
        ml_base_weight: float = 0.3,
    ):
        """
        Args:
            performance_lookback: Number of past signals to track per strategy
            cooldown_days: Days to wait before re-entering after a loss
            min_strategy_signals: Minimum signals before a strategy gets dynamic weight
            ml_base_weight: Base weight for ML prediction in the ensemble
        """
        self.lookback = performance_lookback
        self.cooldown_days = cooldown_days
        self.min_signals = min_strategy_signals
        self.ml_weight = ml_base_weight

        # Rolling performance storage: strategy_name → deque of (was_correct, return_pct, date)
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=performance_lookback))

    def combine_signals(
        self,
        strategy_results: List[Tuple[str, str, float, str]],
        ml_prediction: Optional[object] = None,
        regime: str = "RANGING",
        vol_regime: str = "NORMAL",
    ) -> CombinedSignal:
        """
        Combine all strategy signals + ML into a single actionable signal.

        Args:
            strategy_results: List of (name, signal, confidence, reason) from run_all_strategies()
            ml_prediction: Optional PredictionResult from ml_engine.MLEngine.predict()
            regime: Market regime string
            vol_regime: Volatility regime string

        Returns:
            CombinedSignal with all aggregation details
        """
        # 1. Calculate dynamic weights
        strategy_names = [s[0] for s in strategy_results]
        weights = self._calculate_dynamic_weights(strategy_names, regime, vol_regime)

        # 2. Build individual signal map
        individual = {}
        for name, signal, conf, reason in strategy_results:
            # Map signal to numeric
            sig_num = self._signal_to_numeric(signal, conf)
            individual[name] = (signal, sig_num)

        # 3. Weighted vote (exclude HOLDs from calculation)
        weighted_sum = 0.0
        total_weight = 0.0
        strategy_details = {}
        ml_contrib = 0.0

        for name, (signal, sig_num) in individual.items():
            w = weights.get(name, 0.0)
            if signal != "HOLD":
                weighted_sum += sig_num * w
                total_weight += w
            strategy_details[name] = sig_num * w

        # 4. Integrate ML prediction
        if ml_prediction and ml_prediction.is_trained:
            ml_signal = self._signal_to_numeric(
                ml_prediction.signal, ml_prediction.confidence
            )
            ml_w = self.ml_weight * (0.5 + 0.5 * ml_prediction.oos_accuracy / 0.5)
            ml_w = min(ml_w, 0.4)  # Cap ML weight at 40%

            weighted_sum += ml_signal * ml_w
            total_weight += ml_w
            ml_contrib = ml_w / max(total_weight, 0.001)

            # Boost/bias based on model agreement (high agreement → higher weight)
            if ml_prediction.model_agreement > 80:
                weighted_sum *= 1.1

        # 5. Check cooldown
        cooldown_active = False
        for name in strategy_names:
            perf = self.get_strategy_performance(name)
            if perf and perf.is_in_cooldown:
                cooldown_active = True
                break

        # 6. Final score
        net_score = weighted_sum / max(total_weight, 0.001) if total_weight > 0 else 0

        # Apply regime filter check
        regime_filtered = regime in ("TRENDING_BULL", "TRENDING_BEAR", "RANGING", "VOLATILE")

        # 7. Map to signal level
        signal, confidence = self._map_to_signal_level(net_score, ml_prediction, regime)

        # 8. Build reasoning
        reasoning = []
        warnings = []

        if signal in ("STRONG BUY", "BUY"):
            buy_strategies = [(name, w) for name, w in strategy_details.items() if w > 0]
            top_buy = sorted(buy_strategies, key=lambda x: -x[1])[:3]
            reasoning.append(f"Top bullish: {', '.join(s[0] for s in top_buy)}" if top_buy else "Bullish signal")
        elif signal in ("STRONG SELL", "SELL"):
            sell_strategies = [(name, w) for name, w in strategy_details.items() if w < 0]
            top_sell = sorted(sell_strategies, key=lambda x: x[1])[:3]
            reasoning.append(f"Top bearish: {', '.join(s[0] for s in top_sell)}" if top_sell else "Bearish signal")

        if ml_prediction and ml_prediction.is_trained:
            reasoning.append(f"ML: {ml_prediction.signal} (acc: {ml_prediction.oos_accuracy:.0%})")

        reasoning.append(f"Regime: {regime}, Vol: {vol_regime}")

        if cooldown_active:
            warnings.append("Signal cooldown active — recent losses detected")

        if regime == "VOLATILE":
            warnings.append("High volatility — reduce position sizes")

        # Cap confidence
        confidence = min(95, max(5, confidence))

        return CombinedSignal(
            signal=signal,
            confidence=confidence,
            net_score=round(net_score, 3),
            reasoning=reasoning,
            warnings=warnings,
            strategy_weights={k: round(v, 3) for k, v in weights.items()},
            ml_contribution=round(ml_contrib, 3),
            regime_filter_applied=regime_filtered,
            cooldown_active=cooldown_active,
            individual_signals=individual,
        )

    def record_outcome(self, strategy_name: str, signal: str,
                       was_correct: bool, actual_return: float, date: str):
        """
        Record the outcome of a past signal for performance tracking.

        Called after each trade closes to update rolling hit rates.

        Args:
            strategy_name: Name of the strategy
            signal: "BUY", "SELL", or "HOLD"
            was_correct: Whether the trade was profitable
            actual_return: Actual return percentage of the trade
            date: Date string (YYYY-MM-DD)
        """
        self._history[strategy_name].append({
            "signal": signal,
            "correct": was_correct,
            "return": actual_return,
            "date": date,
        })

    def get_strategy_performance(self, name: str) -> Optional[StrategyPerformance]:
        """Return current performance stats for a strategy."""
        history = self._history.get(name)
        if not history or len(history) == 0:
            return None

        recent = list(history)
        n = len(recent)
        wins = sum(1 for r in recent if r["correct"])
        losses = n - wins

        gross_profit = sum(r["return"] for r in recent if r["correct"] and r["return"] > 0)
        gross_loss = abs(sum(r["return"] for r in recent if not r["correct"] and r["return"] < 0))

        hit_rate = wins / n if n > 0 else 0.5
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
        avg_ret = np.mean([r["return"] for r in recent]) if n > 0 else 0.0

        # Cooldown check
        last_loss_date = None
        for r in reversed(recent):
            if not r["correct"]:
                last_loss_date = r["date"]
                break

        is_cooling = False
        if last_loss_date:
            try:
                last_dt = datetime.strptime(last_loss_date, "%Y-%m-%d")
                days_since = (datetime.now() - last_dt).days
                is_cooling = days_since < self.cooldown_days
            except ValueError:
                is_cooling = False

        return StrategyPerformance(
            name=name,
            hit_rate=round(hit_rate, 3),
            profit_factor=round(profit_factor, 2),
            avg_return=round(avg_ret, 3),
            total_signals=n,
            recent_signals=n,
            last_loss_date=last_loss_date or "",
            is_in_cooldown=is_cooling,
        )

    def reset_performance_history(self):
        """Clear all tracked performance."""
        self._history.clear()

    # ── Private Methods ──

    def _calculate_dynamic_weights(self, strategy_names: List[str],
                                   regime: str, vol_regime: str) -> Dict[str, float]:
        """Calculate weight for each strategy based on recent performance and regime."""
        n = len(strategy_names)
        base_weight = 1.0 / max(n, 1)

        # Get regime multipliers
        regime_mult = REGIME_MULTIPLIERS.get(regime, {})

        weights = {}
        for name in strategy_names:
            w = base_weight

            # Adjust by recent performance
            perf = self.get_strategy_performance(name)
            if perf and perf.total_signals >= self.min_signals:
                # Hit rate adjustment: higher hit rate → higher weight
                w *= perf.hit_rate / 0.5

                # Profit factor adjustment: PF > 1.5 → bonus
                w *= min(perf.profit_factor, 3.0) / 1.5

                # Cooldown penalty
                if perf.is_in_cooldown:
                    w *= 0.3

            # Regime multiplier
            mult = regime_mult.get(name, DEFAULT_REGIME_MULTIPLIER)
            w *= mult

            # Volatility adjustment: in high vol, reduce noise-sensitive strategies
            if vol_regime in ("HIGH", "EXTREME"):
                if name in ("Stochastic", "Williams %R", "RSI"):
                    w *= 0.7

            weights[name] = max(w, 0.01)  # Never go to zero

        # Normalize so weights sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _signal_to_numeric(self, signal: str, confidence: float) -> float:
        """Convert signal + confidence to a numeric value between -1 and +1."""
        sig = signal.upper()
        if sig in ("BUY", "STRONG BUY"):
            return min(confidence / 100, 0.95)
        elif sig in ("SELL", "STRONG SELL"):
            return -min(confidence / 100, 0.95)
        return 0.0  # HOLD

    def _map_to_signal_level(self, net_score: float,
                             ml_prediction: Optional[object] = None,
                             regime: str = "") -> Tuple[str, float]:
        """
        Map continuous score to discrete signal level with confidence.

        Thresholds:
            > +0.60 → STRONG BUY  (conf 80-95)
            > +0.25 → BUY         (conf 60-80)
            > -0.25 → NEUTRAL     (conf 40-60)
            > -0.60 → SELL       (conf 60-80)
            else    → STRONG SELL (conf 80-95)
        """
        abs_score = abs(net_score)

        if net_score > 0.60:
            sig = "STRONG BUY"
            conf = 75 + int(abs_score * 33)
        elif net_score > 0.25:
            sig = "BUY"
            conf = 55 + int(abs_score * 80)
        elif net_score > -0.25:
            sig = "NEUTRAL"
            conf = 40 + int((1 - abs_score * 4) * 20)
        elif net_score > -0.60:
            sig = "SELL"
            conf = 55 + int(abs_score * 80)
        else:
            sig = "STRONG SELL"
            conf = 75 + int(abs_score * 33)

        # Boost confidence if ML agrees
        if ml_prediction and ml_prediction.is_trained:
            if (sig.startswith("BUY") and ml_prediction.signal == "BUY") or \
               (sig.startswith("SELL") and ml_prediction.signal == "SELL"):
                conf = min(95, conf + 10)
                if ml_prediction.oos_accuracy > 0.6:
                    conf = min(95, conf + 5)

        return sig, conf
