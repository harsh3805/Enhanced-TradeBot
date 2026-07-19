"""
Adaptive Parameters — dynamically adjusts indicator periods and stop-loss/target
multipliers based on the current volatility regime.

The problem with fixed parameters (RSI=14, MACD=12/26/9, BB=20/2):
- In high volatility, they're too slow to react
- In low volatility, they whipsaw too much

This module maps volatility regime → optimal parameter set and recalculates
indicators with those parameters.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

from core.volatility_model import VolRegime


@dataclass
class AdaptiveIndicatorParams:
    """All adjustable indicator parameters for a given regime."""
    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR
    atr_period: int = 14
    atr_multiplier_sl: float = 2.0     # for stop-loss
    atr_multiplier_tp: float = 3.0     # for target

    # Stochastic
    stoch_k: int = 14
    stoch_d: int = 3

    # ADX
    adx_period: int = 14

    # Signal confidence thresholds
    min_confidence: float = 25.0        # minimum confidence to act
    strong_signal_threshold: float = 70.0

    # Position sizing adjustment factor (1.0 = normal, < 1 = reduce, > 1 = increase)
    size_multiplier: float = 1.0

    # Source annotations
    regime_source: str = "DEFAULT"

    def to_dict(self) -> dict:
        """Return parameters as flat dict (for display in UI)."""
        return {
            "RSI Period": self.rsi_period,
            "RSI Overbought": self.rsi_overbought,
            "RSI Oversold": self.rsi_oversold,
            "MACD (slow/fast/signal)": f"{self.macd_fast}/{self.macd_slow}/{self.macd_signal}",
            "BB Period/Std": f"{self.bb_period}/{self.bb_std}",
            "ATR Multiplier (SL/TP)": f"{self.atr_multiplier_sl}/{self.atr_multiplier_tp}",
            "Min Confidence": self.min_confidence,
            "Strong Threshold": self.strong_signal_threshold,
            "Size Multiplier": self.size_multiplier,
            "Regime": self.regime_source,
        }


class AdaptiveParamEngine:
    """
    Maps volatility and market regime to optimal indicator parameter sets.

    Three profiles:
    - HIGH_VOL:  Faster indicators, tighter stops, higher confidence thresholds
    - NORMAL:    Standard parameters (same as current defaults)
    - LOW_VOL:   Slower indicators, wider stops, lower thresholds

    Usage:
        engine = AdaptiveParamEngine()
        params = engine.get_params(vol_regime="HIGH", market_regime="TRENDING_BULL")
    """

    # ── Regime-specific parameter profiles ──
    PROFILES = {
        "HIGH_VOL": AdaptiveIndicatorParams(
            rsi_period=10,
            rsi_overbought=75.0, rsi_oversold=25.0,
            macd_fast=8, macd_slow=17, macd_signal=7,
            bb_period=15, bb_std=1.8,
            atr_period=14, atr_multiplier_sl=1.5, atr_multiplier_tp=2.5,
            stoch_k=10, stoch_d=3,
            adx_period=10,
            min_confidence=35.0, strong_signal_threshold=80.0,
            size_multiplier=0.6,
            regime_source="HIGH_VOL",
        ),
        "EXTREME_VOL": AdaptiveIndicatorParams(
            rsi_period=8,
            rsi_overbought=80.0, rsi_oversold=20.0,
            macd_fast=5, macd_slow=13, macd_signal=5,
            bb_period=10, bb_std=1.5,
            atr_period=10, atr_multiplier_sl=1.2, atr_multiplier_tp=2.0,
            stoch_k=8, stoch_d=3,
            adx_period=8,
            min_confidence=45.0, strong_signal_threshold=85.0,
            size_multiplier=0.35,
            regime_source="EXTREME_VOL",
        ),
        "NORMAL": AdaptiveIndicatorParams(
            regime_source="NORMAL",
        ),
        "LOW_VOL": AdaptiveIndicatorParams(
            rsi_period=21,
            rsi_overbought=65.0, rsi_oversold=35.0,
            macd_fast=19, macd_slow=39, macd_signal=14,
            bb_period=30, bb_std=2.2,
            atr_period=14, atr_multiplier_sl=2.5, atr_multiplier_tp=3.5,
            stoch_k=21, stoch_d=7,
            adx_period=21,
            min_confidence=20.0, strong_signal_threshold=60.0,
            size_multiplier=1.25,
            regime_source="LOW_VOL",
        ),
    }

    def __init__(self):
        pass

    def get_params(self, vol_regime: str, market_regime: Optional[str] = None) -> AdaptiveIndicatorParams:
        """
        Map volatility regime + optional market regime to parameter set.

        Args:
            vol_regime: "LOW", "NORMAL", "HIGH", or "EXTREME"
            market_regime: Optional "TRENDING_BULL", "TRENDING_BEAR", "RANGING", "VOLATILE"

        Returns:
            AdaptiveIndicatorParams with appropriate profile
        """
        vol_regime = vol_regime.upper() if vol_regime else "NORMAL"

        # Map vol regime to profile
        if vol_regime in ("EXTREME",):
            params = self.PROFILES["EXTREME_VOL"]
        elif vol_regime in ("HIGH",):
            params = self.PROFILES["HIGH_VOL"]
        elif vol_regime in ("LOW",):
            params = self.PROFILES["LOW_VOL"]
        else:
            params = self.PROFILES["NORMAL"]

        # Market regime fine-tuning
        if market_regime and market_regime in ("TRENDING_BULL", "TRENDING_BEAR"):
            # Trending: slightly faster to catch moves early
            if vol_regime == "NORMAL":
                params = AdaptiveIndicatorParams(
                    rsi_period=12, rsi_overbought=72, rsi_oversold=28,
                    macd_fast=10, macd_slow=22, macd_signal=8,
                    bb_period=18, bb_std=2.0,
                    atr_period=14, atr_multiplier_sl=1.8, atr_multiplier_tp=3.0,
                    min_confidence=22.0,
                    size_multiplier=1.15,
                    regime_source=f"NORMAL_TRENDING",
                )

        return params

    def get_adaptive_stop_loss(self, entry_price: float, atr: float,
                               signal: str, params: AdaptiveIndicatorParams) -> tuple:
        """
        Calculate stop-loss and target using adaptive ATR multipliers.

        Args:
            entry_price: Entry price
            atr: Current ATR value
            signal: "BUY" or "SELL"
            params: Adaptive parameters with multipliers

        Returns:
            (stop_loss, target_price, risk_reward)
        """
        if atr <= 0 or entry_price <= 0:
            return None, None, None

        if signal.upper() == "BUY":
            stop_loss = round(entry_price - params.atr_multiplier_sl * atr, 2)
            target = round(entry_price + params.atr_multiplier_tp * atr, 2)
        else:
            stop_loss = round(entry_price + params.atr_multiplier_sl * atr, 2)
            target = round(entry_price - params.atr_multiplier_tp * atr, 2)

        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)
        rr = round(reward / risk, 2) if risk > 0 else 0

        return stop_loss, target, rr

    def apply_to_dataframe(self, df: pd.DataFrame, params: AdaptiveIndicatorParams) -> pd.DataFrame:
        """
        Calculate indicators on df using adaptive parameters.
        Adds columns with suffix _adaptive (does NOT overwrite originals).

        Args:
            df: DataFrame with Open, High, Low, Close, Volume
            params: Adaptive parameters to use

        Returns:
            DataFrame with additional adaptive indicator columns
        """
        if df.empty or len(df) < max(params.rsi_period, params.macd_slow, params.bb_period) + 5:
            return df

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # Adaptive RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(params.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(params.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI_adaptive"] = 100 - (100 / (1 + rs))

        # Adaptive MACD
        ema_fast = close.ewm(span=params.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=params.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=params.macd_signal, adjust=False).mean()
        df["MACD_adaptive"] = macd_line
        df["MACD_Signal_adaptive"] = signal_line
        df["MACD_Hist_adaptive"] = macd_line - signal_line

        # Adaptive Bollinger Bands
        sma = close.rolling(params.bb_period).mean()
        std = close.rolling(params.bb_period).std()
        df["BB_Upper_adaptive"] = sma + params.bb_std * std
        df["BB_Lower_adaptive"] = sma - params.bb_std * std
        df["BB_Mid_adaptive"] = sma

        # Adaptive ADX
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr_adx = tr.rolling(params.atr_period).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        pos_di = 100 * (pos_dm.rolling(params.atr_period).mean() / atr_adx)
        neg_di = 100 * (neg_dm.rolling(params.atr_period).mean() / atr_adx)
        dx = 100 * ((pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan))
        df["ADX_adaptive"] = dx.rolling(params.atr_period).mean()

        return df

    def describe(self, params: AdaptiveIndicatorParams) -> str:
        """Return human-readable description of current parameter set."""
        regime = params.regime_source
        desc = f"[{regime}] RSI({params.rsi_period}) MACD({params.macd_fast},{params.macd_slow},{params.macd_signal}) "
        desc += f"BB({params.bb_period},{params.bb_std}) SL={params.atr_multiplier_sl}x TP={params.atr_multiplier_tp}x"
        return desc
