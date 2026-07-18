"""
AI Analyst — Multi-factor decision engine for Indian stock market.
Analyzes ALL 24+ strategies together with market context, then outputs clear BUY/SELL.
User only needs to: select stock -> see AI recommendation -> click Buy or Sell.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.data_fetcher import get_stock_data
from core.analyzer import calculate_all_indicators, get_trend, get_latest_indicators, get_support_resistance
from core.strategies_all import run_all_strategies, get_strategy_votes, ALL_STRATEGIES
from core.strategies import get_stop_loss_target
from core.institutional import get_fii_signal, get_fii_dii_flow_summary
from core.market_regime import detect_regime, get_regime_bias


class AIAnalyst:
    """
    AI-powered market analyst that considers:
    - ALL 24+ technical strategies (trend, momentum, volume, volatility, breakout, mean reversion)
    - Market context (FII/DII flow)
    - Volatility regime
    - Risk-adjusted positioning
    Outputs: clear BUY/SELL/HOLD with actionable reasoning.
    """

    def __init__(self, symbol, market="INDIA"):
        self.symbol = symbol.upper().replace(".NS", "")
        self.full_symbol = f"{self.symbol}.NS"
        self.market = market
        self.df = None
        self.indicators = None
        self.analysis = {}

    def analyze(self):
        """Run full analysis: ALL strategies, market context, volatility, trend."""
        self.df = get_stock_data(self.full_symbol, period="1y")
        if self.df.empty or len(self.df) < 30:
            return {"error": f"Insufficient data for {self.full_symbol}"}

        self.df = calculate_all_indicators(self.df)
        self.indicators = get_latest_indicators(self.df)
        price = self.df["Close"].iloc[-1]

        # 1. Market regime detection
        regime_info = detect_regime(self.df)
        regime = regime_info["regime"]
        bias, bias_text = get_regime_bias(regime_info)

        # 2. Run ALL 24+ strategies
        strategy_results = run_all_strategies(self.df)

        # 3. Filter strategies by market regime
        allowed = regime_info["allowed_strategies"]
        blocked = regime_info["blocked_strategies"]
        filtered_results = [(n, s, c, r) for n, s, c, r in strategy_results
                           if n not in blocked]
        # Add blocked strategies as HOLD so they don't vote
        strategy_results_filtered = filtered_results + \
            [(n, "HOLD", 0, f"Blocked in {regime} regime") for n in blocked]

        votes, weighted_buy, weighted_sell, total_conf = get_strategy_votes(strategy_results_filtered)

        # 4. Multi-timeframe alignment (weekly check)
        mtf_aligned = self._check_mtf_alignment()

        # 5. Volume confirmation
        volume_ok, vol_detail = self._check_volume()

        # 6. Earnings proximity check
        near_earnings = self._check_earnings()

        # Original combined signal for reference
        from core.strategies import combined_signal as orig_signal
        combined, conf, _ = orig_signal(self.df)
        trend, trend_conf, trend_det = get_trend(self.df)

        # Market context
        fii_sig, fii_conf, fii_reason = get_fii_signal()

        # Technical assessment
        tech_score, tech_details = self._score_technicals()
        vol_score, vol_detail = self._assess_volatility()
        trend_str, trend_detail = self._assess_trend_strength()
        supports, resistances = get_support_resistance(self.df)

        # Entry/SL/Target
        sl, tp, rr = get_stop_loss_target(self.df, combined)

        # Final AI decision using ALL strategies + regime + MTF + volume
        decision, confidence, reasoning = self._make_ai_decision(
            votes=votes, total=len(strategy_results_filtered),
            weighted_buy=weighted_buy, weighted_sell=weighted_sell,
            fii_sig=fii_sig, tech_score=tech_score,
            trend_str=trend_str, price=price, sl=sl, tp=tp, rr=rr,
            regime=regime, bias=bias, mtf_aligned=mtf_aligned,
            volume_ok=volume_ok, near_earnings=near_earnings,
        )
        # Round confidence to integer
        confidence = min(95, max(0, int(confidence)))

        # Build per-strategy breakdown
        breakdown = []
        for name, sig, c, reason in strategy_results_filtered:
            breakdown.append({"name": name, "signal": sig, "confidence": c, "reason": reason})

        self.analysis = {
            "symbol": self.symbol, "price": round(price, 2),
            "decision": decision, "confidence": confidence, "reasoning": reasoning,
            "entry": round(price, 2), "stop_loss": sl, "target": tp, "risk_reward": rr,
            "strategy_votes": votes, "total_strategies": len(strategy_results_filtered),
            "regime": regime_info,
            "mtf_aligned": mtf_aligned,
            "volume_ok": volume_ok,
            "near_earnings": near_earnings,
            "bias": bias,
            "strategy_details": breakdown,
            "indicators": {
                "RSI": self.indicators.get("RSI", "-"),
                "MACD": self.indicators.get("MACD", "-"),
                "ADX": self.indicators.get("ADX", "-"),
                "ATR": self.indicators.get("ATR", "-"),
            },
            "market_context": {
                "fii_signal": fii_sig, "fii_reason": fii_reason,
                "trend": trend, "trend_confidence": trend_conf,
            },
            "technical_health": tech_score,
            "volatility": vol_detail,
            "trend_strength": trend_detail,
            "support_levels": supports, "resistance_levels": resistances,
            "timestamp": datetime.now().isoformat(),
        }
        return self.analysis

    def _score_technicals(self):
        """Score technical health 0-100."""
        if self.df is None:
            return 50, {"score": 50, "rating": "Neutral", "details": {}}
        latest = self.df.iloc[-1]
        score = 50
        details = {}
        price = latest["Close"]

        if latest.get("SMA_20") and price > latest["SMA_20"]:
            score += 8; details["price_vs_sma20"] = "bullish"
        elif latest.get("SMA_20"):
            score -= 5; details["price_vs_sma20"] = "bearish"

        if latest.get("SMA_50") and price > latest["SMA_50"]:
            score += 6; details["price_vs_sma50"] = "bullish"
        elif latest.get("SMA_50"):
            score -= 4; details["price_vs_sma50"] = "bearish"

        if latest.get("SMA_20") and latest.get("SMA_50") and latest["SMA_20"] > latest["SMA_50"]:
            score += 10; details["ma_alignment"] = "golden_cross"
        elif latest.get("SMA_20") and latest.get("SMA_50"):
            score -= 8; details["ma_alignment"] = "death_cross"

        rsi = latest.get("RSI", 50)
        if pd.isna(rsi): rsi = 50
        if rsi < 35: score += 12; details["rsi_zone"] = "oversold"
        elif rsi < 45: score += 5; details["rsi_zone"] = "near_oversold"
        elif rsi > 70: score -= 8; details["rsi_zone"] = "overbought"
        elif rsi > 60: score -= 3; details["rsi_zone"] = "near_overbought"
        else: details["rsi_zone"] = "neutral"

        if latest.get("MACD") and latest.get("MACD_Signal"):
            if latest["MACD"] > latest["MACD_Signal"]: score += 8; details["macd"] = "bullish"
            else: score -= 5; details["macd"] = "bearish"

        if latest.get("BB_Lower") and price <= latest["BB_Lower"] * 1.02: score += 8; details["bb"] = "near_lower"
        elif latest.get("BB_Upper") and price >= latest["BB_Upper"] * 0.98: score -= 6; details["bb"] = "near_upper"

        vol = latest.get("Volume", 0)
        vol_avg = self.df["Volume"].tail(20).mean() if "Volume" in self.df.columns else 0
        if vol_avg > 0 and vol > vol_avg * 1.5: score += 5; details["volume"] = "high_volume"
        elif vol_avg > 0 and vol < vol_avg * 0.5: score -= 3; details["volume"] = "low_volume"

        score = max(0, min(100, score))
        rating = "Strong" if score >= 70 else "Good" if score >= 55 else "Weak" if score >= 35 else "Poor"
        return score, {"score": score, "rating": rating, "details": details}

    def _assess_volatility(self):
        if self.df is None or len(self.df) < 15: return "NORMAL", "Insufficient data"
        latest = self.df.iloc[-1]
        atr = latest.get("ATR", 0)
        if pd.isna(atr) or atr == 0: return "NORMAL", "ATR N/A"
        atr_pct = (atr / latest["Close"]) * 100
        if atr_pct > 3: return "HIGH", f"ATR {atr_pct:.1f}% - High vol, wide stops"
        elif atr_pct > 1.5: return "NORMAL", f"ATR {atr_pct:.1f}% - Normal vol"
        return "LOW", f"ATR {atr_pct:.1f}% - Low vol, tight stops"

    def _assess_trend_strength(self):
        if self.df is None or len(self.df) < 20: return "WEAK", "Insufficient data"
        adx = self.df.iloc[-1].get("ADX", 0)
        if pd.isna(adx): return "WEAK", "ADX N/A"
        adx = float(adx)
        if adx >= 40: return "STRONG", f"ADX {adx:.0f} - Strong trend"
        elif adx >= 25: return "MODERATE", f"ADX {adx:.0f} - Moderate trend"
        return "WEAK", f"ADX {adx:.0f} - Weak trend (ranging)"

    def _check_mtf_alignment(self):
        """Check if Daily and Weekly timeframes agree."""
        try:
            weekly = get_stock_data(self.full_symbol, period="3mo", interval="1wk")
            if weekly.empty or len(weekly) < 2:
                return True, "Weekly data unavailable"
            weekly = calculate_all_indicators(weekly)
            w_trend, w_conf, _ = get_trend(weekly)
            d_trend, d_conf, _ = get_trend(self.df)
            aligned = (w_trend == d_trend) or (w_trend in ("BULLISH", "BEARISH") and d_trend in ("BULLISH", "BEARISH"))
            if aligned:
                return True, f"Both timeframes {d_trend}"
            return False, f"Daily {d_trend} vs Weekly {w_trend}"
        except:
            return True, "MTF check unavailable"

    def _check_volume(self):
        """Check if volume confirms the signal."""
        try:
            if self.df.empty or "Volume" not in self.df.columns:
                return True, "Volume data unavailable"
            vol_cur = self.df["Volume"].iloc[-1]
            vol_avg = self.df["Volume"].tail(20).mean()
            if vol_avg <= 0:
                return True, "N/A"
            ratio = vol_cur / vol_avg
            if ratio >= 1.5:
                return True, f"Volume {ratio:.1f}x avg (confirmed)"
            elif ratio >= 1.0:
                return True, f"Volume {ratio:.1f}x avg (adequate)"
            else:
                return False, f"Low volume {ratio:.1f}x avg"
        except:
            return True, "Volume check unavailable"

    def _check_earnings(self):
        """Check if stock is near earnings."""
        try:
            from core.earnings import is_near_earnings
            return is_near_earnings(self.full_symbol, days_threshold=7)
        except:
            return False

    def _make_ai_decision(self, votes, total, weighted_buy, weighted_sell,
                           fii_sig, tech_score, trend_str, price, sl, tp, rr,
                           regime="UNKNOWN", bias="NEUTRAL", mtf_aligned=(True, ""),
                           volume_ok=True, near_earnings=False):
        """Final AI decision using ALL strategy votes + market context."""
        buy_score = 0
        sell_score = 0
        reasons = []
        warnings = []

        # 1. Strategy vote count (40%)
        total_votes = votes["BUY"] + votes["SELL"] + votes["HOLD"]
        if total_votes > 0:
            buy_pct = votes["BUY"] / total_votes
            sell_pct = votes["SELL"] / total_votes
            if buy_pct > 0.5:
                buy_score += 30 * buy_pct
                reasons.append(f"{votes['BUY']}/{total} strategies bullish")
            elif sell_pct > 0.5:
                sell_score += 30 * sell_pct
                reasons.append(f"{votes['SELL']}/{total} strategies bearish")

        # 2. Weighted confidence (10%)
        avg_conf_buy = weighted_buy / max(votes["BUY"], 1)
        avg_conf_sell = weighted_sell / max(votes["SELL"], 1)
        if votes["BUY"] > votes["SELL"]:
            buy_score += 10 * min(avg_conf_buy / 100, 1)
        elif votes["SELL"] > votes["BUY"]:
            sell_score += 10 * min(avg_conf_sell / 100, 1)

        # 3. Technical health (20%)
        if tech_score >= 60:
            buy_score += 20 * (tech_score / 100)
        elif tech_score <= 40:
            sell_score += 20 * ((100 - tech_score) / 100)

        # 4. Trend (10%)
        if trend_str == "STRONG" and votes["BUY"] > votes["SELL"]:
            buy_score += 10
            reasons.append("Strong uptrend + bullish strategy alignment")
        elif trend_str == "STRONG" and votes["SELL"] > votes["BUY"]:
            sell_score += 10
            reasons.append("Strong downtrend + bearish strategy alignment")
        elif trend_str == "WEAK":
            warnings.append("Weak trend - strategies may give false signals")

        # 5. FII/DII context (10%)
        if fii_sig == "BULLISH":
            buy_score += 10
            reasons.append("FIIs buying - institutional support")
        elif fii_sig == "BEARISH":
            sell_score += 10
            reasons.append("FIIs selling - institutional pressure")

        # 6. Risk/Reward (10%)
        if rr and rr >= 2:
            buy_score += 10
            reasons.append(f"Good R:R (1:{rr})")
        elif rr and rr < 1.5:
            sell_score += 3
            warnings.append(f"Poor R:R (1:{rr})")

        # 7. Market regime alignment (10%)
        regime_ok = regime not in ("UNKNOWN",)
        if bias == "BULLISH" and votes["BUY"] > votes["SELL"]:
            buy_score += 10
            reasons.append(f"Regime {regime}: bias matches bullish signals")
        elif bias == "BEARISH" and votes["SELL"] > votes["BUY"]:
            sell_score += 10
            reasons.append(f"Regime {regime}: bias matches bearish signals")
        elif regime == "RANGING":
            buy_score += 3  # Neutral regime slightly favors mean reversion

        # 8. Multi-timeframe alignment (10%)
        mtf_ok, mtf_detail = mtf_aligned if isinstance(mtf_aligned, (list, tuple)) else (mtf_aligned, "")
        if mtf_ok:
            buy_score += 5 if votes["BUY"] > votes["SELL"] else 0
            sell_score += 5 if votes["SELL"] > votes["BUY"] else 0
            reasons.append(f"MTF aligned: {mtf_detail[:30] if mtf_detail else 'Daily+Weekly agree'}")
        else:
            sell_score += 8  # Heavy penalty for MTF disagreement
            warnings.append(f"MTF conflict: {mtf_detail[:30] if mtf_detail else 'timeframes disagree'}")

        # 9. Volume filter (10%)
        vol_ok, vol_detail = volume_ok if isinstance(volume_ok, (list, tuple)) else (volume_ok, "")
        if vol_ok and votes["BUY"] > votes["SELL"]:
            buy_score += 5
        elif not vol_ok and votes["BUY"] > votes["SELL"]:
            sell_score += 8
            warnings.append(f"Low volume: {vol_detail[:30] if vol_detail else 'no volume confirmation'}")
        elif not vol_ok and votes["SELL"] > votes["BUY"]:
            sell_score += 3

        # 10. Earnings penalty (hard block)
        if near_earnings:
            sell_score += 10  # Very heavy penalty near earnings
            warnings.append("⚠️ Earnings this week - avoid this stock")

        # Minor report - standout strategies
        extra = ""
        if votes["BUY"] >= total * 0.6:
            extra = f" ({votes['BUY']}/{total} strategies agree)"

        net = buy_score - sell_score

        if net >= 20:
            decision = "STRONG BUY"
            confidence = min(95, int(55 + net))
        elif net >= 7:
            decision = "BUY"
            confidence = min(85, int(45 + net))
        elif net <= -20:
            decision = "STRONG SELL"
            confidence = min(95, int(55 + abs(net)))
        elif net <= -7:
            decision = "SELL"
            confidence = min(85, int(45 + abs(net)))
        else:
            decision = "HOLD"
            confidence = max(40, 50 - abs(net))
            reasons = ["Mixed signals - wait for clearer setup"]
            warnings = []

        if warnings:
            reasons.append(f"Caution: {'; '.join(warnings[:2])}")

        reasoning = ". ".join(reasons[:4]) if reasons else "No clear signal"
        return decision, confidence, reasoning
