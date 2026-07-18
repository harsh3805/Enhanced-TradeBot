"""
Technical indicator calculations using the `ta` library.
All functions take a DataFrame with OHLCV data and return the DataFrame with indicators added.
"""
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice


def calculate_all_indicators(df):
    """
    Calculate all technical indicators and add them as columns.
    Input: DataFrame with Open, High, Low, Close, Volume.
    Returns: DataFrame with indicator columns added.
    """
    if df.empty or len(df) < 20:
        return df

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── Moving Averages ──
    df["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close, window=50).sma_indicator()
    df["SMA_200"] = SMAIndicator(close, window=200).sma_indicator()
    df["EMA_12"] = EMAIndicator(close, window=12).ema_indicator()
    df["EMA_26"] = EMAIndicator(close, window=26).ema_indicator()

    # ── RSI ──
    df["RSI"] = RSIIndicator(close, window=14).rsi()

    # ── MACD ──
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # ── Bollinger Bands ──
    bb = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Mid"] = bb.bollinger_mavg()
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]

    # ── ATR ──
    df["ATR"] = AverageTrueRange(high, low, close, window=14).average_true_range()

    # ── Stochastic RSI ──
    stoch_rsi = StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    df["Stoch_RSI_K"] = stoch_rsi.stochrsi_k()
    df["Stoch_RSI_D"] = stoch_rsi.stochrsi_d()

    # ── ADX (trend strength) ──
    adx = ADXIndicator(high, low, close, window=14)
    df["ADX"] = adx.adx()
    df["ADX_Pos"] = adx.adx_pos()
    df["ADX_Neg"] = adx.adx_neg()

    # ── VWAP (intraday approximation) ──
    try:
        df["VWAP"] = VolumeWeightedAveragePrice(
            high=high, low=low, close=close, volume=volume, window=14
        ).volume_weighted_average_price()
    except Exception:
        df["VWAP"] = df["Close"].rolling(14).mean()

    return df


def get_latest_indicators(df):
    """Get the most recent values of key indicators."""
    if df.empty:
        return {}
    latest = df.iloc[-1]
    return {
        "Price": round(latest.get("Close", 0), 2),
        "RSI": round(latest.get("RSI", 0), 1),
        "MACD": round(latest.get("MACD", 0), 3),
        "MACD Signal": round(latest.get("MACD_Signal", 0), 3),
        "MACD Histogram": round(latest.get("MACD_Hist", 0), 3),
        "SMA 20": round(latest.get("SMA_20", 0), 2),
        "SMA 50": round(latest.get("SMA_50", 0), 2),
        "SMA 200": round(latest.get("SMA_200", 0), 2),
        "EMA 12": round(latest.get("EMA_12", 0), 2),
        "EMA 26": round(latest.get("EMA_26", 0), 2),
        "BB Upper": round(latest.get("BB_Upper", 0), 2),
        "BB Lower": round(latest.get("BB_Lower", 0), 2),
        "ATR": round(latest.get("ATR", 0), 2),
        "ADX": round(latest.get("ADX", 0), 1),
        "Stoch RSI K": round(latest.get("Stoch_RSI_K", 0), 3),
        "Stoch RSI D": round(latest.get("Stoch_RSI_D", 0), 3),
    }


def get_trend(df):
    """
    Determine the overall trend based on multiple indicators.
    Returns: (trend_string, confidence_pct, details_dict)
    """
    if df.empty or len(df) < 50:
        return "INSUFFICIENT DATA", 0, {}

    latest = df.iloc[-1]
    signals = {"bullish": 0, "bearish": 0}

    # Price vs MAs
    price = latest["Close"]
    if latest.get("SMA_20") and price > latest["SMA_20"]:
        signals["bullish"] += 1
    elif latest.get("SMA_20"):
        signals["bearish"] += 1

    if latest.get("SMA_50") and price > latest["SMA_50"]:
        signals["bullish"] += 1
    elif latest.get("SMA_50"):
        signals["bearish"] += 1

    # MA alignment
    if latest.get("SMA_20") and latest.get("SMA_50") and latest["SMA_20"] > latest["SMA_50"]:
        signals["bullish"] += 1
    elif latest.get("SMA_20") and latest.get("SMA_50"):
        signals["bearish"] += 1

    # RSI
    rsi = latest.get("RSI", 50)
    if rsi > 55:
        signals["bullish"] += 1
    elif rsi < 45:
        signals["bearish"] += 1

    # MACD
    if latest.get("MACD") and latest.get("MACD_Signal") and latest["MACD"] > latest["MACD_Signal"]:
        signals["bullish"] += 1
    elif latest.get("MACD") and latest.get("MACD_Signal"):
        signals["bearish"] += 1

    total = signals["bullish"] + signals["bearish"]
    if total == 0:
        return "SIDEWAYS", 50, signals

    if signals["bullish"] > signals["bearish"]:
        confidence = int((signals["bullish"] / total) * 100)
        return "BULLISH", confidence, signals
    elif signals["bearish"] > signals["bullish"]:
        confidence = int((signals["bearish"] / total) * 100)
        return "BEARISH", confidence, signals
    else:
        return "SIDEWAYS", 50, signals


def get_support_resistance(df, window=20):
    """Calculate approximate support and resistance levels."""
    if df.empty or len(df) < window:
        return [], []

    recent = df.tail(window)
    price = df["Close"].iloc[-1]

    # Simple pivot point method
    support_levels = []
    resistance_levels = []

    for i in range(2, len(recent) - 2):
        if (recent["Low"].iloc[i] < recent["Low"].iloc[i-1] and
            recent["Low"].iloc[i] < recent["Low"].iloc[i-2] and
            recent["Low"].iloc[i] < recent["Low"].iloc[i+1] and
            recent["Low"].iloc[i] < recent["Low"].iloc[i+2]):
            support_levels.append(recent["Low"].iloc[i])

        if (recent["High"].iloc[i] > recent["High"].iloc[i-1] and
            recent["High"].iloc[i] > recent["High"].iloc[i-2] and
            recent["High"].iloc[i] > recent["High"].iloc[i+1] and
            recent["High"].iloc[i] > recent["High"].iloc[i+2]):
            resistance_levels.append(recent["High"].iloc[i])

    # Add Bollinger Band levels as S/R
    latest = df.iloc[-1]
    if latest.get("BB_Upper"):
        resistance_levels.append(latest["BB_Upper"])
    if latest.get("BB_Lower"):
        support_levels.append(latest["BB_Lower"])

    support_levels = sorted(set([round(s, 2) for s in support_levels if s < price]), reverse=True)[:3]
    resistance_levels = sorted(set([round(r, 2) for r in resistance_levels if r > price]))[:3]

    return support_levels, resistance_levels
