"""
Feature Engineering Engine — transforms raw OHLCV data into 50+ predictive features
across 5 categories: momentum, mean reversion, trend, volatility, and volume.

Output feeds the ML prediction engine for supervised learning of price direction.
All features use only past data at each point (no look-ahead).

Feature categories:
    Momentum (15):  multi-period returns, RSI variants, ROC, MACD, StochRSI
    Mean Reversion (10): Z-scores, BB position, distance from MAs
    Trend (10): ADX, MA alignment, consecutive days, trend strength
    Volatility (10): realized vol, Garman-Klass, Parkinson, ATR, BB squeeze
    Volume (10): volume ratios, OBV, MFI, VWAP, accumulation/distribution
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    forward_horizon: int = 5               # N-day forward return for target
    classification_threshold: float = 1.5  # Min return % to classify as BUY/SELL
    adapt_threshold_to_vol: bool = True    # Whether to adjust threshold by volatility
    min_samples: int = 60                   # Minimum rows before features are valid
    use_adaptive_indicators: bool = False   # Whether to use adaptive indicator params


class FeatureEngineeringEngine:
    """
    Transforms OHLCV data into 50+ features and target labels.

    Usage:
        engine = FeatureEngineeringEngine(forward_horizon=5)
        df = engine.generate_features(df_with_ohlcv)
        X, y = engine.get_feature_matrix(df)
    """

    # Feature name lists for reference and monitoring
    MOMENTUM_FEATURES = [
        "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
        "rsi_14", "rsi_7", "rsi_21",
        "roc_5", "roc_10", "roc_20",
        "macd_hist", "macd_hist_slope",
        "stoch_rsi_k", "stoch_rsi_d",
    ]

    MEAN_REVERSION_FEATURES = [
        "zscore_20", "zscore_50",
        "bb_position", "bb_width",
        "dist_sma20", "dist_sma50", "dist_sma200",
        "dist_ema12", "dist_ema26",
        "rsi_dist_from_50",
    ]

    TREND_FEATURES = [
        "adx", "adx_slope_5d",
        "ma_alignment_score",
        "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
        "sma20_vs_sma50", "sma50_vs_sma200",
        "consecutive_up", "consecutive_down",
    ]

    VOLATILITY_FEATURES = [
        "realized_vol_20", "realized_vol_60",
        "garman_klass_20", "parkinson_20",
        "atr_ratio", "atr_ratio_ma",
        "bb_squeeze", "bb_width_zscore",
        "high_low_pct", "body_pct",
    ]

    VOLUME_FEATURES = [
        "volume_ratio_5d", "volume_ratio_20d",
        "obv_trend_10d",
        "mfi_14",
        "vwap_dist",
        "ad_trend_10d",
        "volume_price_corr_20d",
        "volume_trend_5d",
        "money_flow_ratio",
        "dollar_volume_20d",
    ]

    ALL_FEATURES = (
        MOMENTUM_FEATURES + MEAN_REVERSION_FEATURES
        + TREND_FEATURES + VOLATILITY_FEATURES + VOLUME_FEATURES
    )

    def __init__(self, forward_horizon: int = 5, config: Optional[FeatureConfig] = None):
        """
        Args:
            forward_horizon: Days ahead for target creation (default 5 = 1 trading week)
            config: Optional FeatureConfig overrides
        """
        self.forward_horizon = forward_horizon
        self.config = config or FeatureConfig()
        self._feature_names = list(self.ALL_FEATURES)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point. Generate all features and append to DataFrame.

        Args:
            df: DataFrame with columns Open, High, Low, Close, Volume
                (must have indicators from analyzer.calculate_all_indicators)

        Returns:
            DataFrame with all feature columns added
        """
        if df.empty or len(df) < self.config.min_samples:
            return df

        result = df.copy()

        # Generate feature groups
        result = self._momentum_features(result)
        result = self._mean_reversion_features(result)
        result = self._trend_features(result)
        result = self._volatility_features(result)
        result = self._volume_features(result)

        # Create target
        result = self.create_target(result)

        return result

    def create_target(self, df: pd.DataFrame, classification: bool = True) -> pd.DataFrame:
        """
        Create forward-looking target variable.

        Args:
            df: DataFrame with features
            classification: If True → BUY(1)/SELL(-1)/HOLD(0)
                           If False → continuous forward_return

        Returns:
            DataFrame with 'target' and 'forward_return' columns
        """
        result = df.copy()

        if "Close" not in result.columns:
            return result

        # Calculate forward return
        future_price = result["Close"].shift(-self.forward_horizon)
        result["forward_return"] = ((future_price - result["Close"]) / result["Close"]) * 100

        if classification:
            # Adaptive threshold based on recent volatility
            threshold = self.config.classification_threshold
            if self.config.adapt_threshold_to_vol and "realized_vol_20" in result.columns:
                # Use recent volatility to set classification threshold
                recent_vol = result["realized_vol_20"].rolling(20).mean()
                vol_50pct = result["realized_vol_20"].rolling(60).median()
                # Adjust: higher vol → wider threshold (harder to classify)
                vol_ratio = recent_vol / vol_50pct.replace(0, np.nan)
                threshold = threshold * vol_ratio.clip(lower=0.5, upper=2.0)
                threshold = threshold.fillna(self.config.classification_threshold)

            # Classify
            result["target"] = 0
            result.loc[result["forward_return"] > threshold, "target"] = 1   # BUY
            result.loc[result["forward_return"] < -threshold, "target"] = -1  # SELL

            # Shift target backward so each row's target uses only past data
            result["target"] = result["target"].shift(-self.forward_horizon)
            result["forward_return"] = result["forward_return"].shift(-self.forward_horizon)
        else:
            result["target"] = result["forward_return"].shift(-self.forward_horizon)

        return result

    def get_feature_matrix(self, df: pd.DataFrame, drop_na: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract feature matrix X and target vector y.

        Args:
            df: DataFrame generated by generate_features()
            drop_na: If True, drop rows with any NaN features

        Returns:
            (X: pd.DataFrame, y: pd.Series) ready for ML
        """
        # Get available feature columns (may not all be present if data is short)
        available = [f for f in self._feature_names if f in df.columns]
        X = df[available].copy()
        y = df["target"].copy() if "target" in df.columns else pd.Series(dtype=float)

        if drop_na:
            # Drop rows with NaN in features or target
            mask = X.notna().all(axis=1) & y.notna()
            X = X[mask]
            y = y[mask]

        return X, y

    def get_feature_names(self) -> List[str]:
        """Return ordered list of all feature names."""
        return list(self._feature_names)

    def get_feature_descriptions(self) -> dict:
        """Return human-readable descriptions for each feature."""
        return {
            "ret_2d": "2-day simple return (%)",
            "ret_3d": "3-day simple return (%)",
            "ret_5d": "5-day (1 week) simple return (%)",
            "ret_10d": "10-day (2 week) simple return (%)",
            "ret_20d": "20-day (1 month) simple return (%)",
            "rsi_14": "14-period Relative Strength Index",
            "rsi_7": "7-period RSI (faster)",
            "rsi_21": "21-period RSI (slower)",
            "roc_5": "5-day Rate of Change (%)",
            "roc_10": "10-day Rate of Change (%)",
            "roc_20": "20-day Rate of Change (%)",
            "macd_hist": "MACD histogram (MACD - signal line)",
            "macd_hist_slope": "5-day slope of MACD histogram",
            "stoch_rsi_k": "Stochastic RSI %K line",
            "stoch_rsi_d": "Stochastic RSI %D line (signal)",
            "zscore_20": "Z-score of price vs 20-day SMA",
            "zscore_50": "Z-score of price vs 50-day SMA",
            "bb_position": "Price position in Bollinger Bands (0=lower, 1=upper)",
            "bb_width": "Bollinger Band width (upper-lower)/middle",
            "dist_sma20": "Distance from SMA20 (%)",
            "dist_sma50": "Distance from SMA50 (%)",
            "dist_sma200": "Distance from SMA200 (%)",
            "dist_ema12": "Distance from EMA12 (%)",
            "dist_ema26": "Distance from EMA26 (%)",
            "rsi_dist_from_50": "RSI distance from neutral 50 level",
            "adx": "14-period ADX (trend strength)",
            "adx_slope_5d": "5-day slope of ADX",
            "ma_alignment_score": "SMA20/50/200 alignment (0=poor, 3=perfect)",
            "price_vs_sma20": "Price above SMA20 (1=yes, -1=no, 0=equal)",
            "price_vs_sma50": "Price above SMA50 (binary)",
            "price_vs_sma200": "Price above SMA200 (binary)",
            "sma20_vs_sma50": "SMA20 above SMA50 (binary)",
            "sma50_vs_sma200": "SMA50 above SMA200 (binary)",
            "consecutive_up": "Consecutive up days",
            "consecutive_down": "Consecutive down days",
            "realized_vol_20": "20-day realized volatility (annualized %)",
            "realized_vol_60": "60-day realized volatility (annualized %)",
            "garman_klass_20": "20-day Garman-Klass volatility estimate (%)",
            "parkinson_20": "20-day Parkinson volatility estimate (%)",
            "atr_ratio": "ATR / Close ratio (%)",
            "atr_ratio_ma": "50-day moving average of ATR ratio",
            "bb_squeeze": "Bollinger Band squeeze (1 if width < 50% of 50d avg)",
            "bb_width_zscore": "Z-score of BB width vs its 50-day history",
            "high_low_pct": "Today's high-low range as % of close",
            "body_pct": "Today's candle body as % of range",
            "volume_ratio_5d": "Volume / 5-day avg volume",
            "volume_ratio_20d": "Volume / 20-day avg volume",
            "obv_trend_10d": "Slope of On-Balance Volume over 10 days",
            "mfi_14": "14-period Money Flow Index",
            "vwap_dist": "Distance of close from VWAP (%)",
            "ad_trend_10d": "Slope of Accumulation/Distribution line over 10 days",
            "volume_price_corr_20d": "Correlation between volume and price over 20 days",
            "volume_trend_5d": "Slope of volume over 5 days",
            "money_flow_ratio": "Ratio of positive money flow to negative",
            "dollar_volume_20d": "20-day average dollar volume (Close * Volume)",
        }

    # ── Private Feature Generation Methods ──

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 15 momentum features."""
        close = df["Close"]
        result = df.copy()

        # Multi-period returns
        for period in [2, 3, 5, 10, 20]:
            result[f"ret_{period}d"] = close.pct_change(period) * 100

        # RSI variants (using period 7, 14, 21)
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            result[f"rsi_{period}"] = 100 - (100 / (1 + rs))
            # Clip to handle NaN from zero division
            result[f"rsi_{period}"] = result[f"rsi_{period}"].clip(0, 100)

        # Rate of Change variants
        for period in [5, 10, 20]:
            result[f"roc_{period}"] = close.pct_change(period) * 100

        # MACD histogram features (from existing indicator if available)
        if "MACD_Hist" in result.columns:
            result["macd_hist"] = (result["MACD_Hist"] - result["MACD_Hist"].rolling(50).mean()) / result["MACD_Hist"].rolling(50).std().replace(0, np.nan)
            result["macd_hist_slope"] = result["MACD_Hist"].diff(5) / result["MACD_Hist"].rolling(50).std().replace(0, np.nan)
        else:
            # Calculate from scratch
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
            result["macd_hist"] = (hist - hist.rolling(50).mean()) / hist.rolling(50).std().replace(0, np.nan)
            result["macd_hist_slope"] = hist.diff(5) / hist.rolling(50).std().replace(0, np.nan)

        # Stochastic RSI
        if "Stoch_RSI_K" in result.columns:
            result["stoch_rsi_k"] = result["Stoch_RSI_K"]
            result["stoch_rsi_d"] = result["Stoch_RSI_D"]
        else:
            rsi_14 = self._calc_rsi(close, 14)
            stoch_k = (rsi_14 - rsi_14.rolling(14).min()) / (rsi_14.rolling(14).max() - rsi_14.rolling(14).min()).replace(0, np.nan)
            result["stoch_rsi_k"] = stoch_k * 100
            result["stoch_rsi_d"] = result["stoch_rsi_k"].rolling(3).mean()

        return result

    def _mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 10 mean reversion features."""
        close = df["Close"]
        result = df.copy()

        # Z-scores of price vs moving averages
        for period in [20, 50]:
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            result[f"zscore_{period}"] = ((close - sma) / std.replace(0, np.nan))
            result[f"zscore_{period}"] = result[f"zscore_{period}"].clip(-5, 5)

        # Bollinger Band position (0 = lower band, 1 = upper band)
        bb_upper = result.get("BB_Upper", close.rolling(20).mean() + 2 * close.rolling(20).std())
        bb_lower = result.get("BB_Lower", close.rolling(20).mean() - 2 * close.rolling(20).std())
        bb_range = (bb_upper - bb_lower).replace(0, np.nan)
        result["bb_position"] = ((close - bb_lower) / bb_range).clip(0, 1)
        result["bb_width"] = ((bb_upper - bb_lower) / close) * 100

        # Distance from moving averages (%)
        for ma_name, period in [("sma20", 20), ("sma50", 50), ("sma200", 200)]:
            ma = close.rolling(period).mean()
            result[f"dist_{ma_name}"] = ((close - ma) / ma.replace(0, np.nan)) * 100

        # Distance from EMAs
        for ma_name, period in [("ema12", 12), ("ema26", 26)]:
            ema = close.ewm(span=period, adjust=False).mean()
            result[f"dist_{ma_name}"] = ((close - ema) / ema.replace(0, np.nan)) * 100

        # RSI distance from neutral 50
        if "RSI" in result.columns:
            result["rsi_dist_from_50"] = result["RSI"] - 50
        else:
            rsi = self._calc_rsi(close, 14)
            result["rsi_dist_from_50"] = rsi - 50

        return result

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 10 trend features."""
        close = df["Close"]
        result = df.copy()

        # ADX
        if "ADX" in result.columns:
            result["adx"] = result["ADX"]
        else:
            result["adx"] = self._calc_adx(df)

        result["adx_slope_5d"] = result["adx"].diff(5)

        # MA alignment score: 0-3 based on SMA20 > SMA50 > SMA200 ordering
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        alignment = pd.DataFrame(index=df.index, dtype=float)
        alignment["score"] = 0
        alignment.loc[sma20 > sma50, "score"] += 1
        alignment.loc[sma50 > sma200, "score"] += 1
        alignment.loc[close > sma20, "score"] += 1
        result["ma_alignment_score"] = alignment["score"]

        # Price vs MA binary signals
        for ma_name, period in [("sma20", 20), ("sma50", 50), ("sma200", 200)]:
            ma = close.rolling(period).mean()
            result[f"price_vs_{ma_name}"] = (close > ma).astype(int).diff().fillna(0)
            # Convert to -1/0/1
            result[f"price_vs_{ma_name}"] = np.where(close > ma, 1, np.where(close < ma, -1, 0))

        # MA crossovers
        result["sma20_vs_sma50"] = np.where(sma20 > sma50, 1, -1)
        result["sma50_vs_sma200"] = np.where(sma50 > sma200, 1, -1)

        # Consecutive up/down days
        returns = close.pct_change()
        up = returns > 0
        result["consecutive_up"] = up.groupby((~up).cumsum()).cumsum()
        down = returns < 0
        result["consecutive_down"] = down.groupby((~down).cumsum()).cumsum()

        return result

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 10 volatility features."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        result = df.copy()

        # Realized volatility (annualized %)
        log_ret = np.log(close / close.shift(1))
        for period in [20, 60]:
            result[f"realized_vol_{period}"] = log_ret.rolling(period).std() * np.sqrt(252) * 100

        # Garman-Klass estimator
        hl = np.log(high / low) ** 2
        co = np.log(close / df["Open"]) ** 2
        gk = (0.5 * hl - (2 * np.log(2) - 1) * co).clip(lower=0)
        result["garman_klass_20"] = gk.rolling(20).mean().apply(np.sqrt) * np.sqrt(252) * 100

        # Parkinson estimator
        pk = (hl / (4 * np.log(2))).clip(lower=0)
        result["parkinson_20"] = pk.rolling(20).mean().apply(np.sqrt) * np.sqrt(252) * 100

        # ATR ratio
        if "ATR" in result.columns:
            result["atr_ratio"] = (result["ATR"] / close) * 100
        else:
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            result["atr_ratio"] = (atr / close) * 100

        result["atr_ratio_ma"] = result["atr_ratio"].rolling(50).mean()

        # BB squeeze
        if "BB_Width" in result.columns:
            bb_w = result["BB_Width"]
        else:
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            bb_w = (2 * std / sma) * 100
        bb_w_ma50 = bb_w.rolling(50).mean()
        result["bb_squeeze"] = (bb_w < bb_w_ma50 * 0.5).astype(int)

        # BB width z-score
        result["bb_width_zscore"] = (
            (bb_w - bb_w.rolling(50).mean()) / bb_w.rolling(50).std().replace(0, np.nan)
        )

        # High-low range and body percentages
        result["high_low_pct"] = ((high - low) / close) * 100
        result["body_pct"] = (abs(close - df["Open"]) / (high - low).replace(0, np.nan)) * 100

        return result

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 10 volume features."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        result = df.copy()

        # Volume ratios
        for period in [5, 20]:
            avg_vol = volume.rolling(period).mean()
            result[f"volume_ratio_{period}d"] = volume / avg_vol.replace(0, np.nan)
            result[f"volume_ratio_{period}d"] = result[f"volume_ratio_{period}d"].clip(0, 10)

        # OBV (On-Balance Volume)
        obv = (volume * (close.diff() > 0).astype(int) - volume * (close.diff() < 0).astype(int)).cumsum()
        result["obv_trend_10d"] = obv.diff(10) / obv.rolling(10).std().replace(0, np.nan)

        # MFI (Money Flow Index)
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        mf_ratio = positive_flow / negative_flow.replace(0, np.nan)
        result["mfi_14"] = 100 - (100 / (1 + mf_ratio))

        # VWAP distance
        vwap = (volume * typical_price).rolling(14).sum() / volume.rolling(14).sum().replace(0, np.nan)
        result["vwap_dist"] = ((close - vwap) / vwap.replace(0, np.nan)) * 100

        # Accumulation/Distribution trend
        clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        ad = clv * volume
        result["ad_trend_10d"] = ad.diff(10) / ad.rolling(10).std().replace(0, np.nan)

        # Volume-price correlation (20-day)
        result["volume_price_corr_20d"] = close.rolling(20).corr(volume)

        # Volume trend slope
        result["volume_trend_5d"] = volume.diff(5) / volume.rolling(5).mean().replace(0, np.nan)

        # Money flow ratio
        result["money_flow_ratio"] = mf_ratio

        # Dollar volume (20-day average)
        result["dollar_volume_20d"] = (close * volume).rolling(20).mean() / 10000000  # in crores

        return result

    # ── Utility methods ──

    @staticmethod
    def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI manually."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX manually."""
        high, low, close = df["High"], df["Low"], df["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        pos_di = 100 * (pos_dm.rolling(period).mean() / atr.replace(0, np.nan))
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr.replace(0, np.nan))
        dx = 100 * ((pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan))
        return dx.rolling(period).mean()
