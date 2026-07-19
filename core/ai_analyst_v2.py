"""
Enhanced AI Analyst v2 — ML-first trading support engine for Indian markets.

Combines:
    - ML ensemble predictions (30% weight, adjusts by OOS accuracy)
    - 24-strategy vote aggregation via SmartSignalCombiner (25%)
    - Technical health assessment (15%)
    - Market regime alignment (10%)
    - FII/DII institutional flow (10%, drops to 0 if stale)
    - Multi-timeframe confirmation (5%)
    - Volume + liquidity confirmation (5%)

Outputs a complete actionable TradeRecommendation with entry, stop-loss,
target, position size, cost analysis, and human-readable reasoning.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from core.data_fetcher import get_stock_data, get_stock_info
from core.analyzer import calculate_all_indicators, get_trend, get_latest_indicators
from core.strategies_all import run_all_strategies
from core.strategies import get_stop_loss_target as old_stop_loss
from core.institutional import get_fii_signal
from core.market_regime import detect_regime, get_regime_bias
from core.cost_model import IndianMarketCostModel
from core.volatility_model import VolatilityModel
from core.adaptive_params import AdaptiveParamEngine
from core.signal_combiner import SmartSignalCombiner
from core.ml_features import FeatureEngineeringEngine
from core.ml_engine import MLEngine, PredictionResult


@dataclass
class TradeRecommendation:
    """Complete actionable trading recommendation."""
    symbol: str = ""
    action: str = "HOLD"            # STRONG BUY / BUY / HOLD / SELL / STRONG SELL
    confidence: float = 50.0        # 0-100 calibrated confidence

    # Entry
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    risk_reward: float = 0.0

    # Position sizing
    position_size_shares: int = 0
    position_size_pct: float = 0.0  # % of capital

    # Cost analysis
    estimated_costs: float = 0.0
    break_even_move_pct: float = 0.0
    net_expected_return: float = 0.0

    # Signal sources
    ml_signal: Optional[str] = None
    ml_confidence: Optional[float] = None
    ml_oos_accuracy: Optional[float] = None
    strategy_counts: Dict[str, int] = field(default_factory=lambda: {"BUY": 0, "SELL": 0, "HOLD": 0})

    # Context
    market_regime: str = "RANGING"
    vol_regime: str = "NORMAL"
    adaptive_params: Dict[str, Any] = field(default_factory=dict)
    fii_dii_signal: str = "NEUTRAL"
    mtf_aligned: bool = True
    volume_confirmed: bool = True
    near_earnings: bool = False

    # Reasoning
    reasoning: str = ""
    warnings: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)

    # Metadata
    timestamp: str = ""
    analysis_version: str = "v2.0"


class AIAnalystV2:
    """
    Enhanced AI Analyst — ML-first, regime-aware, cost-conscious.

    Usage:
        analyst = AIAnalystV2("RELIANCE", capital=100000)
        rec = analyst.analyze()
        print(rec.action, rec.confidence, rec.reasoning)
    """

    def __init__(self, symbol: str, market: str = "INDIA", capital: float = 100000.0):
        self.symbol = symbol.upper().replace(".NS", "")
        self.full_symbol = f"{self.symbol}.NS"
        self.market = market
        self.capital = capital

        # Lazy-init components
        self._cost_model = IndianMarketCostModel()
        self._vol_model = VolatilityModel()
        self._adaptive_engine = AdaptiveParamEngine()
        self._signal_combiner = SmartSignalCombiner()
        self._feature_engine = FeatureEngineeringEngine(forward_horizon=5)
        self._ml_engine = MLEngine()

    def analyze(self) -> TradeRecommendation:
        """
        Full analysis pipeline — runs all components and assembles recommendation.

        Steps:
            1. Fetch daily (1y) + weekly (3mo) data
            2. Detect market regime
            3. Estimate volatility regime
            4. Get adaptive parameter set
            5. Run all 24 strategies
            6. Generate ML prediction (train if needed)
            7. Combine signals via SmartSignalCombiner
            8. Assess technical health
            9. Check FII/DII flow
            10. Check MTF alignment, volume, earnings proximity
            11. Calculate position size with costs
            12. Assemble TradeRecommendation
        """
        # ── Step 1: Data ──
        df = get_stock_data(self.full_symbol, period="1y")
        if df.empty or len(df) < 60:
            return TradeRecommendation(
                symbol=self.symbol,
                action="HOLD",
                confidence=0,
                reasoning=f"Insufficient data for {self.full_symbol}",
                timestamp=datetime.now().isoformat(),
            )

        df = calculate_all_indicators(df)

        # Weekly data for MTF
        df_weekly = get_stock_data(self.full_symbol, period="3mo", interval="1wk")
        if not df_weekly.empty and len(df_weekly) > 2:
            df_weekly = calculate_all_indicators(df_weekly)

        price = float(df["Close"].iloc[-1])

        # ── Step 2: Market Regime ──
        regime_info = detect_regime(df)
        market_regime = regime_info["regime"]
        bias, bias_text = get_regime_bias(regime_info)

        # ── Step 3: Volatility Regime ──
        vol_est = self._vol_model.forecast_volatility(df)
        vol_regime_str = vol_est.vol_regime.value if hasattr(vol_est.vol_regime, "value") else str(vol_est.vol_regime.name)

        # ── Step 4: Adaptive Parameters ──
        adaptive_params = self._adaptive_engine.get_params(vol_regime_str, market_regime)
        params_dict = adaptive_params.to_dict()

        # ── Step 5: Run all strategies ──
        strategy_results = run_all_strategies(df)

        # Filter out strategies blocked by regime
        allowed = regime_info.get("allowed_strategies", [])
        blocked = regime_info.get("blocked_strategies", [])
        strategy_results_filtered = [(n, s, c, r) for n, s, c, r in strategy_results if n not in blocked]

        # Count votes
        buy_votes = sum(1 for _, s, _, _ in strategy_results_filtered if s == "BUY")
        sell_votes = sum(1 for _, s, _, _ in strategy_results_filtered if s == "SELL")
        hold_votes = sum(1 for _, s, _, _ in strategy_results_filtered if s == "HOLD")
        total_votes = len(strategy_results_filtered)

        # ── Step 6: ML Prediction ──
        ml_pred = self._get_ml_prediction(df)

        # ── Step 7: Combine Signals ──
        combined = self._signal_combiner.combine_signals(
            strategy_results_filtered,
            ml_prediction=ml_pred,
            regime=market_regime,
            vol_regime=vol_regime_str,
        )

        # ── Step 8: Technical Health ──
        tech_score, tech_detail = self._assess_technical_health(df)

        # ── Step 9: Context Checks ──
        # FII/DII
        try:
            fii_sig, fii_conf, fii_reason = get_fii_signal()
        except Exception:
            fii_sig, fii_reason = "NEUTRAL", "Data unavailable"

        # MTF alignment
        mtf_ok = self._check_mtf_alignment(df, df_weekly)

        # Volume confirmation
        vol_ok, vol_detail = self._check_volume(df)

        # Earnings proximity
        near_earnings = self._check_earnings()

        # ── Step 10: Stop-Loss & Target ──
        atr_val = float(df.get("ATR", df["Close"] * 0.02).iloc[-1])
        sl, tp, rr = self._adaptive_engine.get_adaptive_stop_loss(
            price, atr_val, combined.signal, adaptive_params
        )

        if sl is None or tp is None or rr is None or rr < 0.5:
            # Fall back to fixed ATR method
            sl_old, tp_old, rr_old = old_stop_loss(df, combined.signal, risk_pct=2.0)
            if sl_old:
                sl, tp, rr = sl_old, tp_old, rr_old

        # ── Step 11: Position Size ──
        shares, pos_pct, costs = self._compute_position_size(
            price, sl, combined.confidence, vol_regime_str
        )

        # ── Step 12: Cost Analysis ──
        try:
            info = get_stock_info(self.full_symbol)
            mcap = info.get("market_cap", None)
            mcap_cr = mcap / 10000000 if mcap else None

            if tp > 0:
                cost_breakdown = self._cost_model.calculate_round_trip_costs(
                    price, tp, max(shares, 1), "delivery", mcap_cr
                )
                be_move = cost_breakdown.break_even_pct
                est_costs = cost_breakdown.round_trip
            else:
                be_move = self._cost_model.break_even_move(price, max(shares, 1), "delivery", mcap_cr)
                est_costs = 0

            # Net expected return (expected_move - costs)
            if tp > price:
                expected_return = ((tp - price) / price) * 100
                net_return = round(expected_return - be_move, 2)
            else:
                net_return = 0
        except Exception:
            be_move = 0.3
            est_costs = 0
            net_return = 0

        # ── Assemble Recommendation ──
        reasoning = self._build_reasoning(combined, ml_pred, market_regime, vol_regime_str,
                                          tech_score, fii_sig, bias_text, mtf_ok)
        warnings = self._build_warnings(combined, market_regime, vol_regime_str, near_earnings, rr)

        strengths = []
        if tech_score >= 65:
            strengths.append(f"Technical health: {tech_detail.get('rating', 'Good')} ({tech_score})")
        if ml_pred and ml_pred.is_trained and ml_pred.oos_accuracy > 0.55:
            strengths.append(f"ML model OOS accuracy: {ml_pred.oos_accuracy:.1%}")
        if rr and rr >= 2:
            strengths.append(f"Strong risk/reward: 1:{rr}")
        if fii_sig in ("BULLISH",) and combined.signal.startswith("BUY"):
            strengths.append("FII buying confirms institutional support")

        return TradeRecommendation(
            symbol=self.symbol,
            action=combined.signal,
            confidence=combined.confidence,
            entry_price=round(price, 2),
            stop_loss=round(sl, 2) if sl else 0,
            target_price=round(tp, 2) if tp else 0,
            risk_reward=round(rr, 2) if rr else 0,
            position_size_shares=shares,
            position_size_pct=pos_pct,
            estimated_costs=round(est_costs, 2) if est_costs else 0,
            break_even_move_pct=be_move,
            net_expected_return=net_return,
            ml_signal=ml_pred.signal if ml_pred else None,
            ml_confidence=ml_pred.confidence if ml_pred else None,
            ml_oos_accuracy=round(ml_pred.oos_accuracy, 3) if ml_pred else None,
            strategy_counts={"BUY": buy_votes, "SELL": sell_votes, "HOLD": hold_votes},
            market_regime=market_regime,
            vol_regime=vol_regime_str,
            adaptive_params=params_dict,
            fii_dii_signal=fii_sig,
            mtf_aligned=mtf_ok,
            volume_confirmed=vol_ok,
            near_earnings=near_earnings,
            reasoning=reasoning,
            warnings=warnings,
            strengths=strengths,
            timestamp=datetime.now().isoformat(),
        )

    # ── Internal Methods ──

    def _get_ml_prediction(self, df: pd.DataFrame) -> Optional[PredictionResult]:
        """Get or train ML model, then predict."""
        try:
            if self._ml_engine.needs_retrain(self.symbol):
                perf = self._ml_engine.train(df, self._feature_engine)
                if isinstance(perf, dict) and "error" not in perf:
                    self._ml_engine.save_model(self.symbol)
            else:
                self._ml_engine.load_model(self.symbol)

            result = self._ml_engine.predict(df, self._feature_engine)
            return result if result.is_trained else None
        except Exception:
            return None

    def _assess_technical_health(self, df: pd.DataFrame) -> Tuple[float, dict]:
        """Score technical health 0-100 using adaptive thresholds."""
        if df is None or df.empty:
            return 50, {"score": 50, "rating": "Neutral", "details": {}}

        latest = df.iloc[-1]
        score = 50
        details = {}
        price = float(latest["Close"])

        # Price vs SMAs (30% weight equivalent)
        for ma_name, period in [("SMA_20", 20), ("SMA_50", 50), ("SMA_200", 200)]:
            ma = latest.get(ma_name)
            if ma and not pd.isna(ma):
                if price > ma:
                    score += 6
                    details[f"price_vs_{ma_name}"] = "bullish"
                else:
                    score -= 4
                    details[f"price_vs_{ma_name}"] = "bearish"

        # MA alignment
        sma20 = latest.get("SMA_20")
        sma50 = latest.get("SMA_50")
        if sma20 and sma50 and not pd.isna(sma20) and not pd.isna(sma50):
            if sma20 > sma50:
                score += 10
                details["ma_alignment"] = "golden_cross"
            else:
                score -= 8
                details["ma_alignment"] = "death_cross"

        # RSI
        rsi = latest.get("RSI", 50)
        if pd.isna(rsi): rsi = 50
        if rsi < 35:
            score += 12; details["rsi"] = "oversold_bounce_setup"
        elif rsi < 45:
            score += 5; details["rsi"] = "near_oversold"
        elif rsi > 70:
            score -= 8; details["rsi"] = "overbought"
        elif rsi > 60:
            score -= 3; details["rsi"] = "near_overbought"
        else:
            details["rsi"] = "neutral"

        # MACD
        macd = latest.get("MACD")
        macd_sig = latest.get("MACD_Signal")
        if macd and macd_sig and not pd.isna(macd) and not pd.isna(macd_sig):
            if macd > macd_sig:
                score += 8; details["macd"] = "bullish"
            else:
                score -= 5; details["macd"] = "bearish"

        # Volume
        vol = latest.get("Volume", 0)
        vol_avg = df["Volume"].tail(20).mean() if "Volume" in df.columns else 0
        if vol_avg > 0 and vol > vol_avg * 1.5:
            score += 5; details["volume"] = "high_volume_confirms"
        elif vol_avg > 0 and vol < vol_avg * 0.5:
            score -= 3; details["volume"] = "low_volume_risk"

        score = int(max(0, min(100, score)))
        if score >= 70:
            rating = "Strong"
        elif score >= 55:
            rating = "Good"
        elif score >= 35:
            rating = "Weak"
        else:
            rating = "Poor"

        return score, {"score": score, "rating": rating, "details": details}

    def _check_mtf_alignment(self, df_daily: pd.DataFrame, df_weekly: pd.DataFrame) -> bool:
        """Check if daily and weekly trends align."""
        if df_weekly is None or df_weekly.empty or len(df_weekly) < 2:
            return True
        try:
            from core.analyzer import get_trend
            w_trend, w_conf, _ = get_trend(df_weekly)
            d_trend, d_conf, _ = get_trend(df_daily)
            return (w_trend == d_trend) or \
                   (w_trend in ("BULLISH", "BEARISH") and d_trend in ("BULLISH", "BEARISH"))
        except Exception:
            return True

    def _check_volume(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check volume confirmation. Returns (ok, detail)."""
        try:
            if "Volume" not in df.columns:
                return True, "Volume data unavailable"
            vol_cur = float(df["Volume"].iloc[-1])
            vol_avg = float(df["Volume"].tail(20).mean())
            if vol_avg <= 0:
                return True, "N/A"
            ratio = vol_cur / vol_avg
            if ratio >= 1.5:
                return True, f"{ratio:.1f}x avg"
            elif ratio >= 1.0:
                return True, f"{ratio:.1f}x avg (adequate)"
            return False, f"Low volume {ratio:.1f}x avg"
        except Exception:
            return True, "Check unavailable"

    def _check_earnings(self) -> bool:
        """Check if stock is near earnings."""
        try:
            from core.earnings import is_near_earnings
            return is_near_earnings(self.full_symbol, days_threshold=7)
        except Exception:
            return False

    def _compute_position_size(self, price: float, stop_loss: float,
                               confidence: float, vol_regime: str) -> Tuple[int, float, float]:
        """Calculate position size adjusted by confidence and volatility."""
        if price <= 0 or stop_loss is None or stop_loss >= price:
            return 0, 0, 0

        risk_per_trade = 0.02  # 2% base risk
        # Adjust for confidence
        conf_factor = confidence / 100.0
        # Adjust for volatility
        vol_factor = {
            "EXTREME": 0.35,
            "HIGH": 0.6,
            "NORMAL": 1.0,
            "LOW": 1.25,
        }.get(vol_regime, 1.0)

        risk_amount = self.capital * risk_per_trade * conf_factor * vol_factor
        risk_per_share = abs(price - stop_loss)

        if risk_per_share <= 0:
            return 0, 0, 0

        shares = int(risk_amount / risk_per_share)
        total_cost = shares * price
        pos_pct = (total_cost / self.capital * 100) if self.capital > 0 else 0

        return shares, round(pos_pct, 2), total_cost

    def _build_reasoning(self, combined, ml_pred, regime, vol_regime,
                         tech_score, fii_sig, bias_text, mtf_ok) -> str:
        """Build concise, human-readable reasoning."""
        parts = []

        action = combined.signal
        if action in ("STRONG BUY", "BUY"):
            parts.append(f"Bullish signal from ML + strategies")
        elif action in ("STRONG SELL", "SELL"):
            parts.append(f"Bearish signal from ML + strategies")

        if ml_pred and ml_pred.is_trained and ml_pred.signal != "HOLD":
            parts.append(f"ML {ml_pred.signal} ({ml_pred.confidence}%, acc: {ml_pred.oos_accuracy:.0%})")

        if combined.reasoning:
            for r in combined.reasoning[:3]:
                parts.append(r)

        parts.append(f"Tech health: {tech_score}/100")
        parts.append(f"Regime: {regime} | Vol: {vol_regime}")
        if fii_sig in ("BULLISH", "BEARISH") and fii_sig:
            parts.append(f"FII: {fii_sig}")

        if not mtf_ok:
            parts.append("MTF conflict: daily vs weekly disagree")
        if combined.cooldown_active:
            parts.append("Cooldown: recent losses")

        return " | ".join(parts[:6]) if parts else "No clear signal"

    def _build_warnings(self, combined, regime, vol_regime,
                        near_earnings, rr) -> List[str]:
        """Build actionable warnings."""
        w = list(combined.warnings) if combined.warnings else []

        if near_earnings:
            w.append("Earnings this week — avoid new positions")
        if rr and rr < 1.5:
            w.append(f"Poor risk/reward (1:{rr})")
        if vol_regime in ("HIGH", "EXTREME"):
            w.append(f"High volatility — reduce position size")
        if combined.cooldown_active:
            w.append("Cooldown active — wait 3 days after loss")

        return w[:4]  # Max 4 warnings
