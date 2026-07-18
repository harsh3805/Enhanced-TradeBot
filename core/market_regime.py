"""
Market Regime Detector — Classifies market as TRENDING, RANGING, or VOLATILE.
Strategies must match the regime or they get filtered out.
"""
import pandas as pd
import numpy as np
from core.data_fetcher import get_stock_data
from core.analyzer import calculate_all_indicators


def detect_regime(df=None, period="3mo"):
    """
    Detect current market regime.
    Returns: {
        "regime": "TRENDING_BULL" | "TRENDING_BEAR" | "RANGING" | "VOLATILE",
        "confidence": 0-100,
        "details": {...},
        "allowed_strategies": [...],
        "blocked_strategies": [...],
    }
    """
    if df is None:
        df = get_stock_data("NIFTY.NS" if False else "^NSEI", period=period)
        if df.empty:
            # Try Nifty 50 index
            df = get_stock_data("^NSEI", period=period)
            if df.empty:
                # Fall back to a major stock
                df = get_stock_data("RELIANCE.NS", period=period)
                if df.empty:
                    return _fallback_regime()

    if df.empty or len(df) < 30:
        return _fallback_regime()

    df = calculate_all_indicators(df)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # 1. ADX for trend strength
    adx = df["ADX"].iloc[-1] if "ADX" in df.columns else 20
    adx_avg = df["ADX"].tail(20).mean() if "ADX" in df.columns else 20
    if pd.isna(adx): adx = 20
    if pd.isna(adx_avg): adx_avg = 20

    # 2. Bollinger Band width for volatility
    bb_width = df["BB_Width"].tail(20) if "BB_Width" in df.columns else None
    current_width = bb_width.iloc[-1] if bb_width is not None else 0.05
    avg_width = bb_width.mean() if bb_width is not None else 0.05
    if pd.isna(current_width) or not bb_width is not None:
        current_width = 0.05
        avg_width = 0.05
    vol_ratio = current_width / avg_width if avg_width > 0 else 1.0

    # 3. Directional movement
    sma20 = df["SMA_20"].iloc[-1] if "SMA_20" in df.columns else close.iloc[-1]
    sma50 = df["SMA_50"].iloc[-1] if "SMA_50" in df.columns else close.iloc[-1]
    sma200 = df["SMA_200"].iloc[-1] if "SMA_200" in df.columns else None

    price = close.iloc[-1]
    price_20d_ago = close.iloc[-20] if len(close) >= 20 else price
    change_20d = ((price - price_20d_ago) / price_20d_ago) * 100

    price_50d_ago = close.iloc[-50] if len(close) >= 50 else price
    change_50d = ((price - price_50d_ago) / price_50d_ago) * 100

    # 4. Count consecutive up/down days
    returns = close.pct_change()
    up_days = (returns.tail(20) > 0).sum()
    down_days = (returns.tail(20) < 0).sum()

    # ── Regime Classification ──
    regime = "RANGING"
    confidence = 50
    details = {}

    # High volatility
    if vol_ratio > 1.3:
        if change_20d > 3:
            regime = "TRENDING_BULL"
            confidence = min(90, 60 + int(adx / 5))
        elif change_20d < -3:
            regime = "TRENDING_BEAR"
            confidence = min(90, 60 + int(adx / 5))
        else:
            regime = "VOLATILE"
            confidence = 70
    # High ADX = trending
    elif adx > 30:
        if sma20 > sma50 and price > sma50:
            regime = "TRENDING_BULL"
            confidence = min(85, 50 + int(adx / 4))
        elif sma20 < sma50 and price < sma50:
            regime = "TRENDING_BEAR"
            confidence = min(85, 50 + int(adx / 4))
        else:
            regime = "RANGING"
            confidence = 40
    # Low ADX = ranging
    elif adx < 20:
        regime = "RANGING"
        confidence = 60
    else:
        # Mixed — check price action
        if up_days >= 14:
            regime = "TRENDING_BULL"
            confidence = 55
        elif down_days >= 14:
            regime = "TRENDING_BEAR"
            confidence = 55
        else:
            regime = "RANGING"
            confidence = 45

    details = {
        "adx": round(adx, 1),
        "volatility_ratio": round(vol_ratio, 2),
        "change_20d": round(change_20d, 1),
        "up_days_20": up_days,
        "down_days_20": down_days,
        "price_vs_sma50": "above" if price > sma50 else "below",
        "price_vs_sma200": "above" if sma200 and price > sma200 else "below" if sma200 else "unknown",
    }

    # Which strategies to filter
    allowed = _strategies_for_regime(regime)
    blocked = _blocked_for_regime(regime)

    return {
        "regime": regime,
        "confidence": confidence,
        "details": details,
        "allowed_strategies": allowed,
        "blocked_strategies": blocked,
        "summary": f"Market is {regime} (ADX {adx:.0f}, Vol {vol_ratio:.1f}x, {change_20d:+.1f}% 20d)",
    }


def _strategies_for_regime(regime):
    """Return list of strategy names that work in this regime."""
    base = ["RSI", "MACD", "MFI", "OBV", "VPT", "Volume Breakout"]
    if regime == "TRENDING_BULL":
        return base + ["SuperTrend", "Ichimoku", "PSAR", "Vortex", "TRIX", "KST", "KAMA",
                        "Donchian", "Awesome Osc", "ROC", "TSI", "WMA Cross"]
    elif regime == "TRENDING_BEAR":
        return base + ["SuperTrend", "Ichimoku", "PSAR", "Donchian", "Chandelier"]
    elif regime == "RANGING":
        return base + ["Stochastic", "Williams %R", "CCI", "Ultimate Osc",
                       "Keltner", "Bollinger", "S/R Breakout"]
    elif regime == "VOLATILE":
        return base + ["Keltner", "Donchian", "Chandelier", "Mass Index"]
    return base


def _blocked_for_regime(regime):
    """Return list of strategy names that should be skipped in this regime."""
    if regime == "TRENDING_BULL":
        return ["Stochastic", "Williams %R", "Keltner"]
    elif regime == "TRENDING_BEAR":
        return ["Stochastic", "Williams %R", "CCI", "Ultimate Osc"]
    elif regime == "RANGING":
        return ["SuperTrend", "Ichimoku", "PSAR", "Donchian", "TRIX"]
    elif regime == "VOLATILE":
        return ["Ichimoku", "KST", "KAMA", "TSI", "WMA Cross"]
    return []


def _fallback_regime():
    return {
        "regime": "UNKNOWN",
        "confidence": 30,
        "details": {"error": "Could not fetch market data"},
        "allowed_strategies": [],
        "blocked_strategies": [],
        "summary": "Unknown regime (data unavailable)",
    }


def get_regime_bias(regime_info):
    """Return a trading bias based on regime."""
    regime = regime_info["regime"]
    if regime == "TRENDING_BULL":
        return "BULLISH", "Prefer BUY signals. Let profits run with trailing stops."
    elif regime == "TRENDING_BEAR":
        return "BEARISH", "Prefer SELL signals or stay in cash. Tight stops on any long."
    elif regime == "RANGING":
        return "NEUTRAL", "Trade bounces at support/resistance. Take quick profits."
    elif regime == "VOLATILE":
        return "CAUTIOUS", "Reduce position sizes 50%. Wide stops. Day trade only."
    return "NEUTRAL", "No clear bias. Trade confirmed setups only."
