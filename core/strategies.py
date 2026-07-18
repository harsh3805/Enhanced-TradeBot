"""
Trading strategy signals — each returns a signal (BUY/SELL/HOLD) with confidence.
Combined signal aggregates all strategies with configurable weights.
"""
import pandas as pd
import numpy as np
from utils.config import STRATEGY_WEIGHTS


def rsi_strategy(df):
    """
    RSI-based signals:
    - BUY when RSI crosses above 30 (oversold)
    - SELL when RSI crosses below 70 (overbought)
    - HOLD otherwise
    """
    if df.empty or "RSI" not in df.columns:
        return "HOLD", 0, "Insufficient data"

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    rsi = latest["RSI"]
    prev_rsi = prev["RSI"]

    if pd.isna(rsi):
        return "HOLD", 0, "RSI not available"

    if rsi < 30:
        return "BUY", min(90, int((30 - rsi) * 3)), f"RSI oversold at {rsi:.1f}"
    elif prev_rsi < 30 and rsi >= 30:
        return "BUY", 70, f"RSI crossed above 30 (now {rsi:.1f})"
    elif rsi > 70:
        return "SELL", min(90, int((rsi - 70) * 3)), f"RSI overbought at {rsi:.1f}"
    elif prev_rsi > 70 and rsi <= 70:
        return "SELL", 70, f"RSI crossed below 70 (now {rsi:.1f})"
    elif rsi < 40:
        return "BUY", 40, f"RSI approaching oversold at {rsi:.1f}"
    elif rsi > 60:
        return "SELL", 40, f"RSI approaching overbought at {rsi:.1f}"
    else:
        return "HOLD", 20, f"RSI neutral at {rsi:.1f}"


def macd_strategy(df):
    """
    MACD crossover signals:
    - BUY when MACD crosses above signal line
    - SELL when MACD crosses below signal line
    """
    if df.empty or "MACD" not in df.columns or "MACD_Signal" not in df.columns:
        return "HOLD", 0, "Insufficient data"

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    macd = latest["MACD"]
    signal = latest["MACD_Signal"]
    prev_macd = prev["MACD"]
    prev_signal = prev["MACD_Signal"]

    if pd.isna(macd) or pd.isna(signal):
        return "HOLD", 0, "MACD not available"

    hist = macd - signal
    prev_hist = prev_macd - prev_signal

    # Bullish crossover
    if prev_hist <= 0 and hist > 0:
        # Strength based on histogram magnitude
        strength = min(85, int(abs(hist) * 500 + 50))
        return "BUY", strength, f"MACD bullish crossover (histogram: {hist:.4f})"

    # Bearish crossover
    if prev_hist >= 0 and hist < 0:
        strength = min(85, int(abs(hist) * 500 + 50))
        return "SELL", strength, f"MACD bearish crossover (histogram: {hist:.4f})"

    # Strong trend continuation
    if hist > 0 and hist > prev_hist:
        return "BUY", 40, f"MACD bullish momentum increasing ({hist:.4f})"
    elif hist < 0 and hist < prev_hist:
        return "SELL", 40, f"MACD bearish momentum increasing ({hist:.4f})"
    elif hist > 0:
        return "BUY", 25, f"MACD above signal (bullish: {hist:.4f})"
    elif hist < 0:
        return "SELL", 25, f"MACD below signal (bearish: {hist:.4f})"

    return "HOLD", 15, "MACD neutral"


def ma_crossover_strategy(df):
    """
    Moving average crossover signals:
    - Golden Cross: SMA20 crosses above SMA50 → BUY
    - Death Cross: SMA20 crosses below SMA50 → SELL
    - Price position relative to SMAs for additional context
    """
    if df.empty or "SMA_20" not in df.columns or "SMA_50" not in df.columns:
        return "HOLD", 0, "Insufficient data"

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    price = latest["Close"]
    sma20 = latest["SMA_20"]
    sma50 = latest["SMA_50"]
    sma200 = latest.get("SMA_200")
    prev_sma20 = prev["SMA_20"]
    prev_sma50 = prev["SMA_50"]

    if pd.isna(sma20) or pd.isna(sma50):
        return "HOLD", 0, "MA data not available"

    # Golden / Death cross
    if prev_sma20 <= prev_sma50 and sma20 > sma50:
        return "BUY", 80, "Golden Cross: SMA20 crossed above SMA50"
    if prev_sma20 >= prev_sma50 and sma20 < sma50:
        return "SELL", 80, "Death Cross: SMA20 crossed below SMA50"

    # Price position
    bullish_count = 0
    bearish_count = 0

    if price > sma20:
        bullish_count += 1
    else:
        bearish_count += 1

    if sma20 > sma50:
        bullish_count += 1
    else:
        bearish_count += 1

    if sma200 and sma50 > sma200:
        bullish_count += 1
    elif sma200:
        bearish_count += 1

    if bullish_count >= 2:
        conf = 30 + bullish_count * 10
        return "BUY", min(75, conf), f"Price above MAs (bullish: {bullish_count}/3)"
    elif bearish_count >= 2:
        conf = 30 + bearish_count * 10
        return "SELL", min(75, conf), f"Price below MAs (bearish: {bearish_count}/3)"

    return "HOLD", 20, "MA signals mixed"


def bollinger_strategy(df):
    """
    Bollinger Band signals:
    - BUY when price touches/crosses lower band (mean reversion)
    - SELL when price touches/crosses upper band
    - BUY on squeeze breakout upward
    """
    if df.empty or "BB_Upper" not in df.columns:
        return "HOLD", 0, "Insufficient data"

    latest = df.iloc[-1]
    price = latest["Close"]
    bb_upper = latest["BB_Upper"]
    bb_lower = latest["BB_Lower"]
    bb_mid = latest["BB_Mid"]
    bb_width = latest.get("BB_Width", 0)

    if pd.isna(bb_upper) or pd.isna(bb_lower):
        return "HOLD", 0, "Bollinger data not available"

    # Price at/below lower band
    if price <= bb_lower * 1.01:
        return "BUY", 75, f"Price near lower Bollinger Band ({price:.2f} vs {bb_lower:.2f})"

    # Price at/above upper band
    if price >= bb_upper * 0.99:
        return "SELL", 75, f"Price near upper Bollinger Band ({price:.2f} vs {bb_upper:.2f})"

    # Mean reversion — moving toward middle
    if price < bb_mid and price > bb_lower:
        return "BUY", 35, f"Price between lower band and middle ({price:.2f})"
    elif price > bb_mid and price < bb_upper:
        return "SELL", 35, f"Price between middle and upper band ({price:.2f})"

    return "HOLD", 20, "Price within Bollinger Bands"


def combined_signal(df):
    """
    Combine all strategies with configurable weights.
    Returns: (signal, confidence, explanation)
    """
    strategies = {
        "rsi": rsi_strategy(df),
        "macd": macd_strategy(df),
        "ma_crossover": ma_crossover_strategy(df),
        "bollinger": bollinger_strategy(df),
    }

    # Weighted scoring
    scores = {"BUY": 0, "SELL": 0, "HOLD": 0}
    total_weight = sum(STRATEGY_WEIGHTS.values())
    explanations = []

    for name, (signal, confidence, reason) in strategies.items():
        weight = STRATEGY_WEIGHTS.get(name, 0.25)
        weighted_score = (confidence / 100) * weight
        scores[signal] += weighted_score
        explanations.append(f"**{name.upper()}**: {signal} ({confidence}%) — {reason}")

    # Determine overall signal
    buy_score = scores["BUY"]
    sell_score = scores["SELL"]
    hold_score = scores["HOLD"]

    if buy_score > sell_score and buy_score > hold_score and buy_score > 0.25:
        overall = "BUY"
        conf = min(95, int((buy_score / total_weight) * 100))
    elif sell_score > buy_score and sell_score > hold_score and sell_score > 0.25:
        overall = "SELL"
        conf = min(95, int((sell_score / total_weight) * 100))
    else:
        overall = "HOLD"
        conf = min(60, int((hold_score / total_weight) * 100))

    explanation = "\n".join(explanations)
    return overall, conf, explanation


def get_stop_loss_target(df, signal, risk_pct=2.0):
    """
    Suggest stop-loss and target based on ATR.
    Returns: (stop_loss, target_price, risk_reward)
    """
    if df.empty or "ATR" not in df.columns:
        return None, None, None

    latest = df.iloc[-1]
    price = latest["Close"]
    atr = latest.get("ATR", 0)

    if pd.isna(atr) or atr == 0:
        return None, None, None

    if signal == "BUY":
        stop_loss = round(price - 2 * atr, 2)
        target = round(price + 3 * atr, 2)
    elif signal == "SELL":
        stop_loss = round(price + 2 * atr, 2)
        target = round(price - 3 * atr, 2)
    else:
        return None, None, None

    risk = abs(price - stop_loss)
    reward = abs(target - price)
    rr = round(reward / risk, 2) if risk > 0 else 0

    return stop_loss, target, rr
