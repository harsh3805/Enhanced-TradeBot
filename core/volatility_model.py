"""
Volatility Model — GARCH(1,1) conditional volatility, realized volatility estimation,
volatility regime classification, and volatility cone analysis.

Provides forward-looking volatility estimates used by:
- Adaptive parameter engine (selects indicator periods)
- Position sizing (adjusts risk per volatility regime)
- Signal combiner (regime-specific weighting)
"""
import warnings
from dataclasses import dataclass
from typing import Optional
from enum import Enum

import numpy as np
import pandas as pd


class VolRegime(Enum):
    """Volatility regime classifications."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class VolatilityEstimate:
    """Complete volatility estimate combining multiple methods."""
    current_realized_vol: float = 0.0       # Annualized 20-day realized vol (%)
    garch_forecast_1d: float = 0.0          # GARCH(1,1) 1-step ahead conditional vol (%)
    garch_forecast_5d: float = 0.0          # 5-day ahead forecast (%)
    garch_forecast_21d: float = 0.0         # 21-day (1 month) ahead forecast (%)
    vol_regime: VolRegime = VolRegime.NORMAL
    percentile_rank: float = 50.0           # Current vol in historical distribution (0-100)
    garman_klass: float = 0.0               # Garman-Klass efficient estimator (%)
    parkinson: float = 0.0                  # Parkinson HL range estimator (%)
    atr_ratio: float = 0.0                  # ATR / Close (%)
    atr: float = 0.0                        # Raw ATR value
    vol_of_vol: float = 0.0                 # Volatility of volatility (stability)
    cone_percentiles: dict = None            # {window: {min, p25, p50, p75, max}}
    convergence_ok: bool = True             # Whether GARCH converged

    def __post_init__(self):
        if self.cone_percentiles is None:
            self.cone_percentiles = {}


class VolatilityModel:
    """
    Multi-method volatility estimator with GARCH forecasting.

    Combines:
    1. Realized volatility (close-to-close)
    2. Garman-Klass estimator (efficient, uses OHLC)
    3. Parkinson estimator (uses HL range)
    4. GARCH(1,1) conditional volatility forecast
    5. Volatility cone (multi-window percentile analysis)
    """

    def __init__(self, lookback_days: int = 756):
        """
        Args:
            lookback_days: Historical lookback for GARCH estimation (3 years default)
        """
        self.lookback_days = lookback_days

    def forecast_volatility(self, df: pd.DataFrame) -> VolatilityEstimate:
        """
        Full volatility estimation combining all methods.

        Args:
            df: DataFrame with columns Open, High, Low, Close, Volume

        Returns:
            VolatilityEstimate with all fields populated
        """
        if df.empty or len(df) < 30:
            return VolatilityEstimate(convergence_ok=False)

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        open_p = df["Open"]

        # Log returns for GARCH
        log_returns = np.log(close / close.shift(1)).dropna()

        # 1. Realized volatility (20-day, annualized)
        realized = self._realized_volatility(close, window=20)

        # 2. Garman-Klass estimate
        gk = self._garman_klass(high, low, open_p, close, window=20)

        # 3. Parkinson estimate
        pk = self._parkinson(high, low, window=20)

        # Current values
        current_vol = float(realized.iloc[-1]) if not realized.empty else 0.0
        current_gk = float(gk.iloc[-1]) if not gk.empty else 0.0
        current_pk = float(pk.iloc[-1]) if not pk.empty else 0.0

        # 4. ATR ratio
        atr_series = close - low  # simplified ATR
        atr = float(
            df.get("ATR", atr_series.rolling(14).mean().bfill()).iloc[-1]
        )
        current_price = float(close.iloc[-1])
        atr_ratio = (atr / current_price * 100) if current_price > 0 else 0

        # 5. GARCH forecast
        garch_1d, garch_5d, garch_21d, converged = self._garch_forecast(log_returns)

        # 6. Volatility regime from percentile
        historical_vols = realized.dropna()
        if len(historical_vols) > 20:
            percentile = stats_percentile(historical_vols.values, current_vol)
            regime = self.classify_vol_regime(current_vol, historical_vols)
        else:
            percentile = 50.0
            regime = VolRegime.NORMAL

        # 7. Volatility cone
        cone = self.compute_volatility_cone(close)

        # 8. Vol of vol
        vol_of_vol = float(
            realized.rolling(60).std().iloc[-1]
        ) if len(realized) > 60 else 0.0

        return VolatilityEstimate(
            current_realized_vol=round(current_vol, 2),
            garch_forecast_1d=round(garch_1d, 2),
            garch_forecast_5d=round(garch_5d, 2),
            garch_forecast_21d=round(garch_21d, 2),
            vol_regime=regime,
            percentile_rank=round(percentile, 1),
            garman_klass=round(current_gk, 2),
            parkinson=round(current_pk, 2),
            atr_ratio=round(atr_ratio, 2),
            atr=round(atr, 2),
            vol_of_vol=round(vol_of_vol, 2),
            cone_percentiles=cone,
            convergence_ok=converged,
        )

    def classify_vol_regime(self, current_vol: float,
                            historical_vols: pd.Series) -> VolRegime:
        """
        Classify volatility into regimes based on percentile of historical distribution.

        Thresholds:
            LOW:     < 25th percentile
            NORMAL:  25th - 75th percentile
            HIGH:    75th - 90th percentile
            EXTREME: > 90th percentile
        """
        if len(historical_vols) < 20:
            return VolRegime.NORMAL

        pct = stats_percentile(historical_vols.values, current_vol)

        if pct >= 90:
            return VolRegime.EXTREME
        elif pct >= 75:
            return VolRegime.HIGH
        elif pct >= 25:
            return VolRegime.NORMAL
        return VolRegime.LOW

    def compute_volatility_cone(
        self, close: pd.Series, windows: list = None
    ) -> dict:
        """
        Compute volatility cone: min, 25th, median, 75th, max realized vol
        for each lookback window.

        The volatility cone shows where current vol sits historically
        across multiple time horizons — a standard institutional tool.

        Args:
            close: Close price series
            windows: List of lookback windows (default: [5, 10, 20, 60, 120])

        Returns:
            dict: {window: {"min": x, "p25": x, "p50": x, "p75": x, "max": x}}
        """
        if windows is None:
            windows = [5, 10, 20, 60, 120]

        cone = {}
        returns = close.pct_change().dropna() * 100  # daily returns in %

        for w in windows:
            if len(returns) < w + 5:
                continue

            # Rolling annualized vol for this window
            annualized = returns.rolling(w).std() * np.sqrt(252)
            vals = annualized.dropna().values

            if len(vals) > 10:
                cone[str(w)] = {
                    "min": round(float(np.min(vals)), 2),
                    "p25": round(float(np.percentile(vals, 25)), 2),
                    "p50": round(float(np.median(vals)), 2),
                    "p75": round(float(np.percentile(vals, 75)), 2),
                    "max": round(float(np.max(vals)), 2),
                }

        return cone

    def _garch_forecast(self, log_returns: pd.Series) -> tuple:
        """
        GARCH(1,1) conditional volatility forecast.

        Fits GARCH(1,1) on log returns and forecasts 1-step, 5-step, 21-step ahead.

        Falls back to EWMA (exponentially weighted moving average) if GARCH fails.

        Returns:
            (forecast_1d, forecast_5d, forecast_21d, convergence_ok) in annualized %
        """
        if len(log_returns) < 60:
            # Fall back to simple EWMA
            return self._ewma_forecast(log_returns)

        try:
            from arch import arch_model

            # Fit GARCH(1,1)
            model = arch_model(
                log_returns * 100,  # work in percentage points
                vol="GARCH", p=1, q=1,
                dist="normal",
                rescale=False
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit(disp="off", show_warning=False)

            # Extract parameters
            omega = float(res.params.get("omega", 0.01))
            alpha = float(res.params.get("alpha[1]", 0.05))
            beta = float(res.params.get("beta[1]", 0.90))

            # Check stationarity
            persistence = alpha + beta
            if persistence >= 1:
                # Non-stationary GARCH — fall back to EWMA
                return self._ewma_forecast(log_returns)

            # Current conditional volatility
            cond_vol = res.conditional_volatility.iloc[-1]

            # Forecast forward
            # For GARCH(1,1), h-step forecast: sigma^2_{t+h} = omega + (alpha+beta) * sigma^2_t
            # Approaches long-run variance as h → ∞
            long_run_var = omega / (1 - alpha - beta)
            current_var = cond_vol ** 2

            # Annualize: multiply by sqrt(252)
            ann_factor = np.sqrt(252)

            def forecast_variance(h):
                """Forecast variance h steps ahead."""
                return long_run_var + (alpha + beta) ** (h - 1) * (current_var - long_run_var)

            f1 = np.sqrt(forecast_variance(1)) * ann_factor
            f5 = np.sqrt(np.mean([forecast_variance(h) for h in range(1, 6)])) * ann_factor
            f21 = np.sqrt(np.mean([forecast_variance(h) for h in range(1, 22)])) * ann_factor

            return round(f1, 3), round(f5, 3), round(f21, 3), True

        except Exception:
            return self._ewma_forecast(log_returns)

    def _ewma_forecast(self, log_returns: pd.Series) -> tuple:
        """
        EWMA fallback for volatility forecasting.
        Uses lambda=0.94 (RiskMetrics standard).
        """
        squared_returns = (log_returns * 100) ** 2

        # Apply EWMA with lambda=0.94
        lam = 0.94
        ewma_var = squared_returns.ewm(span=2 / (1 - lam) - 1).mean()

        # Annualize
        ann_factor = np.sqrt(252)
        current_vol = np.sqrt(ewma_var.iloc[-1]) if not ewma_var.empty else 0.1

        # EWMA forecasts are flat (same for all horizons)
        f = round(current_vol * ann_factor, 3)
        return f, f, f, False

    @staticmethod
    def _realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
        """Close-to-close realized volatility, annualized."""
        log_returns = np.log(close / close.shift(1))
        return log_returns.rolling(window).std() * np.sqrt(252) * 100

    @staticmethod
    def _garman_klass(high: pd.Series, low: pd.Series,
                      open_p: pd.Series, close: pd.Series,
                      window: int = 20) -> pd.Series:
        """
        Garman-Klass volatility estimator.

        Formula:
        sigma^2 = 0.5 * ln(H/L)^2 - (2*ln2 - 1) * ln(C/O)^2

        More efficient than close-to-close because it uses intraday range.
        """
        hl = np.log(high / low) ** 2
        co = np.log(close / open_p) ** 2

        gk_var = 0.5 * hl - (2 * np.log(2) - 1) * co
        gk_var = gk_var.clip(lower=0)  # clip negative values
        return gk_var.rolling(window).mean().apply(np.sqrt) * np.sqrt(252) * 100

    @staticmethod
    def _parkinson(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """
        Parkinson volatility estimator (high-low range only).

        Formula:
        sigma^2 = (1 / 4 * ln(2)) * ln(H/L)^2

        Assumes continuous trading (slightly underestimates in practice).
        """
        hl = np.log(high / low) ** 2
        pk_var = hl / (4 * np.log(2))
        pk_var = pk_var.clip(lower=0)
        return pk_var.rolling(window).mean().apply(np.sqrt) * np.sqrt(252) * 100


def stats_percentile(data: np.ndarray, value: float) -> float:
    """
    Calculate percentile rank of a value within a dataset.
    Returns 0-100 representing where value sits in data distribution.
    """
    if len(data) < 5 or np.isnan(value) or value is None:
        return 50.0
    count_less = np.sum(data <= value)
    return float(count_less / len(data) * 100)
