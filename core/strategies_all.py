"""
EVERY trading strategy — 20+ strategies for Indian stock market.
Each strategy outputs a SIGNAL (BUY/SELL/HOLD), CONFIDENCE (0-100), and REASON.
AI Analyst uses ALL of these to make its final recommendation.
"""
import pandas as pd
import numpy as np
from ta.trend import (
    SMAIndicator, EMAIndicator, WMAIndicator,
    MACD, ADXIndicator, CCIIndicator, IchimokuIndicator,
    PSARIndicator, VortexIndicator, KSTIndicator, TRIXIndicator,
    DPOIndicator, MassIndex
)
from ta.momentum import (
    RSIIndicator, StochasticOscillator, StochRSIIndicator,
    WilliamsRIndicator, AwesomeOscillatorIndicator, TSIIndicator,
    UltimateOscillator, ROCIndicator, KAMAIndicator as KAMA_momentum
)
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.volume import (
    VolumeWeightedAveragePrice, MFIIndicator, OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator, EaseOfMovementIndicator
)


def required_indicators(df, indicators):
    """Check if required indicators exist in the DataFrame."""
    for ind in indicators:
        if ind not in df.columns:
            return False
    return True


# ═══════════════════════════════════════════════════════════════
# 1. TREND FOLLOWING STRATEGIES
# ═══════════════════════════════════════════════════════════════

def supertrend_strategy(df):
    """
    SuperTrend — ATR-based trailing stop system.
    BUY when price closes above SuperTrend line (uptrend).
    SELL when price closes below SuperTrend line (downtrend).
    """
    if df.empty or len(df) < 22 or "ATR" not in df.columns:
        return "HOLD", 0, "Insufficient data for SuperTrend"

    # Calculate SuperTrend
    high, low, close = df["High"], df["Low"], df["Close"]
    atr = df["ATR"]
    multiplier = 3.0

    supertrend = [0] * len(df)
    trend = [1] * len(df)  # 1 = uptrend, -1 = downtrend

    for i in range(1, len(df)):
        hl_avg = (high.iloc[i] + low.iloc[i]) / 2
        upper_band = hl_avg + multiplier * atr.iloc[i]
        lower_band = hl_avg - multiplier * atr.iloc[i]

        if trend[i-1] == 1:
            if close.iloc[i] <= upper_band:
                trend[i] = 1
            else:
                trend[i] = -1
        else:
            if close.iloc[i] >= lower_band:
                trend[i] = -1
            else:
                trend[i] = 1

        supertrend[i] = lower_band if trend[i] == 1 else upper_band

    latest_trend = trend[-1]
    prev_trend = trend[-2] if len(trend) > 1 else latest_trend

    if latest_trend == 1 and prev_trend == -1:
        return "BUY", 80, "SuperTrend reversal: Uptrend started"
    elif latest_trend == -1 and prev_trend == 1:
        return "SELL", 80, "SuperTrend reversal: Downtrend started"
    elif latest_trend == 1:
        return "BUY", 55, "SuperTrend: Uptrend continues"
    elif latest_trend == -1:
        return "SELL", 55, "SuperTrend: Downtrend continues"
    return "HOLD", 30, "SuperTrend: No clear signal"


def ichimoku_strategy(df):
    """
    Ichimoku Cloud — Full cloud system.
    BUY: Price above cloud + Tenkan > Kijun + Chikou above price
    SELL: Price below cloud + Tenkan < Kijun + Chikou below price
    """
    if df.empty or len(df) < 60:
        return "HOLD", 0, "Need 60+ bars for Ichimoku"

    # Calculate Ichimoku components
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = ((high.rolling(9).max() + low.rolling(9).min()) / 2)

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = ((high.rolling(26).max() + low.rolling(26).min()) / 2)

    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, shifted 26 periods
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    # Chikou Span (Lagging Span): close shifted -26 periods
    chikou = close.shift(-26)

    latest = df.iloc[-1]
    price = close.iloc[-1]
    cloud_a = senkou_a.iloc[-1]
    cloud_b = senkou_b.iloc[-1]
    tenkan_v = tenkan.iloc[-1]
    kijun_v = kijun.iloc[-1]

    if pd.isna(cloud_a) or pd.isna(cloud_b):
        return "HOLD", 0, "Ichimoku: Not enough data (need 52+ bars)"

    cloud_top = max(cloud_a, cloud_b)
    cloud_bottom = min(cloud_a, cloud_b)
    bullish = 0
    bearish = 0

    # Price relative to cloud
    if price > cloud_top:
        bullish += 1
    elif price < cloud_bottom:
        bearish += 1

    # Tenkan/Kijun cross
    prev_tenkan = tenkan.iloc[-2]
    prev_kijun = kijun.iloc[-2]
    if prev_tenkan <= prev_kijun and tenkan_v > kijun_v:
        bullish += 2  # TK Cross bullish
    elif prev_tenkan >= prev_kijun and tenkan_v < kijun_v:
        bearish += 2

    # Cloud color (future cloud)
    if senkou_a.iloc[-1] > senkou_b.iloc[-1]:
        bullish += 1  # Bullish cloud
    else:
        bearish += 1

    if bullish > bearish and bullish >= 2:
        return "BUY", min(85, bullish * 25 + 30), "Ichimoku: Bullish configuration"
    elif bearish > bullish and bearish >= 2:
        return "SELL", min(85, bearish * 25 + 30), "Ichimoku: Bearish configuration"
    return "HOLD", 35, "Ichimoku: Mixed signals"


def psar_strategy(df):
    """
    Parabolic SAR — Trend reversal detection.
    BUY when SAR flips from above to below price.
    SELL when SAR flips from below to above price.
    """
    if df.empty or len(df) < 20:
        return "HOLD", 0, "Insufficient data for PSAR"

    psar = PSARIndicator(df["High"], df["Low"], df["Close"])
    sar = psar.psar()
    psar_up = psar.psar_up()
    psar_down = psar.psar_down()

    price = df["Close"].iloc[-1]
    sar_v = sar.iloc[-1]

    if pd.isna(sar_v):
        return "HOLD", 0, "PSAR not ready"

    # Check recent SAR flips
    flip_bullish = False
    flip_bearish = False
    for i in range(-5, 0):
        if len(sar) + i >= 0 and not pd.isna(psar_up.iloc[i]) and not pd.isna(psar_down.iloc[i]):
            if psar_up.iloc[i] > psar_down.iloc[i]:
                flip_bullish = True
            else:
                flip_bearish = True

    if flip_bullish and not flip_bearish:
        return "BUY", 65, "Parabolic SAR: Flipped bullish (price trending up)"
    elif flip_bearish and not flip_bullish:
        return "SELL", 65, "Parabolic SAR: Flipped bearish (price trending down)"
    elif psar_up.iloc[-1] < price if not pd.isna(psar_up.iloc[-1]) else False:
        return "BUY", 40, "Parabolic SAR: Bullish (SAR below price)"
    elif psar_down.iloc[-1] > price if not pd.isna(psar_down.iloc[-1]) else False:
        return "SELL", 40, "Parabolic SAR: Bearish (SAR above price)"
    return "HOLD", 25, "Parabolic SAR: No clear reversal"


def vortex_strategy(df):
    """
    Vortex Indicator — Trend direction and strength.
    BUY when VI+ crosses above VI-
    SELL when VI- crosses above VI+
    """
    if df.empty or len(df) < 26:
        return "HOLD", 0, "Insufficient data for Vortex"

    vortex = VortexIndicator(df["High"], df["Low"], df["Close"], window=14)
    vip = vortex.vortex_indicator_pos()
    vin = vortex.vortex_indicator_neg()

    if pd.isna(vip.iloc[-1]):
        return "HOLD", 0, "Vortex not ready"

    prev_vip = vip.iloc[-2]
    prev_vin = vin.iloc[-2]
    cur_vip = vip.iloc[-1]
    cur_vin = vin.iloc[-1]

    if prev_vip <= prev_vin and cur_vip > cur_vin:
        return "BUY", 70, "Vortex: VI+ crossed above VI- (uptrend)"
    elif prev_vip >= prev_vin and cur_vip < cur_vin:
        return "SELL", 70, "Vortex: VI- crossed above VI+ (downtrend)"
    elif cur_vip > cur_vin:
        return "BUY", 40, "Vortex: VI+ above VI- (positive trend)"
    elif cur_vin > cur_vip:
        return "SELL", 40, "Vortex: VI- above VI+ (negative trend)"
    return "HOLD", 20, "Vortex: Mixed"


def trix_strategy(df):
    """
    TRIX — Triple Exponential Moving Average.
    BUY when TRIX crosses above signal line (momentum positive)
    SELL when TRIX crosses below signal line
    """
    if df.empty or len(df) < 30:
        return "HOLD", 0, "Insufficient data for TRIX"

    trix_ind = TRIXIndicator(df["Close"], window=14)
    trix = trix_ind.trix()
    trix_signal = trix.rolling(9).mean()

    if pd.isna(trix.iloc[-1]):
        return "HOLD", 0, "TRIX not ready"

    prev_t = trix.iloc[-2]
    cur_t = trix.iloc[-1]
    prev_s = trix_signal.iloc[-2]
    cur_s = trix_signal.iloc[-1]

    if prev_t <= prev_s and cur_t > cur_s:
        return "BUY", 65, "TRIX crossed above signal line (momentum up)"
    elif prev_t >= prev_s and cur_t < cur_s:
        return "SELL", 65, "TRIX crossed below signal line (momentum down)"
    elif cur_t > cur_s:
        return "BUY", 35, "TRIX above signal (positive momentum)"
    elif cur_t < cur_s:
        return "SELL", 35, "TRIX below signal (negative momentum)"
    return "HOLD", 15, "TRIX flat"


def kst_strategy(df):
    """
    KST (Know Sure Thing) — Sum of four smoothed ROC periods.
    BUY when KST crosses above signal line.
    SELL when KST crosses below signal line.
    """
    if df.empty or len(df) < 60:
        return "HOLD", 0, "Insufficient data for KST"

    kst_ind = KSTIndicator(df["Close"])
    kst = kst_ind.kst()
    kst_sig = kst_ind.kst_sig()

    if pd.isna(kst.iloc[-1]):
        return "HOLD", 0, "KST not ready"

    prev_k = kst.iloc[-2]
    cur_k = kst.iloc[-1]
    prev_s = kst_sig.iloc[-2]
    cur_s = kst_sig.iloc[-1]

    if prev_k <= prev_s and cur_k > cur_s:
        return "BUY", 65, "KST bullish crossover"
    elif prev_k >= prev_s and cur_k < cur_s:
        return "SELL", 65, "KST bearish crossover"
    elif cur_k > cur_s:
        return "BUY", 35, "KST positive"
    elif cur_k < cur_s:
        return "SELL", 35, "KST negative"
    return "HOLD", 15, "KST flat"


def kama_strategy(df):
    """
    Kaufman's Adaptive Moving Average — Trend following with dynamic period.
    BUY when price > KAMA and KAMA rising.
    SELL when price < KAMA and KAMA falling.
    """
    if df.empty or len(df) < 30:
        return "HOLD", 0, "Insufficient data for KAMA"

    kama_ind = KAMAIndicator(df["Close"], window=10, pow1=2, pow2=30)
    kama = kama_ind.kama()

    if pd.isna(kama.iloc[-1]):
        return "HOLD", 0, "KAMA not ready"

    price = df["Close"].iloc[-1]
    kama_v = kama.iloc[-1]
    kama_rising = kama.iloc[-1] > kama.iloc[-5]

    if price > kama_v and kama_rising:
        return "BUY", 55, f"KAMA: Price ({price:.1f}) above KAMA ({kama_v:.1f}) uptrend"
    elif price < kama_v and not kama_rising:
        return "SELL", 55, f"KAMA: Price ({price:.1f}) below KAMA ({kama_v:.1f}) downtrend"
    elif price > kama_v:
        return "BUY", 30, "KAMA: Price above but KAMA flattening"
    elif price < kama_v:
        return "SELL", 30, "KAMA: Price below but KAMA flattening"
    return "HOLD", 15, "KAMA: No clear signal"


# ═══════════════════════════════════════════════════════════════
# 2. MEAN REVERSION STRATEGIES
# ═══════════════════════════════════════════════════════════════

def stochastic_strategy(df):
    """
    Stochastic Oscillator — %K and %D crossover.
    BUY: %K crosses above %D below 20 (oversold).
    SELL: %K crosses below %D above 80 (overbought).
    """
    if df.empty or len(df) < 20:
        return "HOLD", 0, "Insufficient data for Stochastic"

    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    k = stoch.stoch()
    d = stoch.stoch_signal()

    if pd.isna(k.iloc[-1]):
        return "HOLD", 0, "Stochastic not ready"

    k_cur = k.iloc[-1]
    d_cur = d.iloc[-1]
    k_prev = k.iloc[-2]
    d_prev = d.iloc[-2]

    # Bullish crossover below 20
    if k_prev <= d_prev and k_cur > d_cur and k_cur < 25:
        return "BUY", 80, f"Stochastic bullish crossover at {k_cur:.0f} (oversold)"
    # Bearish crossover above 80
    elif k_prev >= d_prev and k_cur < d_cur and k_cur > 75:
        return "SELL", 80, f"Stochastic bearish crossover at {k_cur:.0f} (overbought)"
    # Oversold bounce
    elif k_cur < 20 and k_cur > k_prev:
        return "BUY", 55, f"Stochastic oversold: {k_cur:.0f} bouncing up"
    # Overbought pullback
    elif k_cur > 80 and k_cur < k_prev:
        return "SELL", 55, f"Stochastic overbought: {k_cur:.0f} pulling back"
    elif k_cur < 20:
        return "BUY", 35, f"Stochastic deep oversold: {k_cur:.0f}"
    elif k_cur > 80:
        return "SELL", 35, f"Stochastic deep overbought: {k_cur:.0f}"
    return "HOLD", 20, f"Stochastic neutral: {k_cur:.0f}"


def williams_r_strategy(df):
    """
    Williams %R — Momentum oscillator similar to stochastic.
    BUY when %R crosses above -80 (coming out of oversold).
    SELL when %R crosses below -20 (coming out of overbought).
    """
    if df.empty or len(df) < 20:
        return "HOLD", 0, "Insufficient data for Williams %R"

    wr = WilliamsRIndicator(df["High"], df["Low"], df["Close"], lbp=14)
    w = wr.williams_r()

    if pd.isna(w.iloc[-1]):
        return "HOLD", 0, "Williams %R not ready"

    w_cur = w.iloc[-1]
    w_prev = w.iloc[-2]

    if w_prev <= -80 and w_cur > -80:
        return "BUY", 75, f"Williams %R: Left oversold ({w_prev:.0f} -> {w_cur:.0f})"
    elif w_prev >= -20 and w_cur < -20:
        return "SELL", 75, f"Williams %R: Left overbought ({w_prev:.0f} -> {w_cur:.0f})"
    elif w_cur < -80:
        return "BUY", 40, f"Williams %R: Oversold ({w_cur:.0f})"
    elif w_cur > -20:
        return "SELL", 40, f"Williams %R: Overbought ({w_cur:.0f})"
    return "HOLD", 20, f"Williams %R: Neutral ({w_cur:.0f})"


def cci_strategy(df):
    """
    Commodity Channel Index — Mean reversion and trend detection.
    BUY: CCI < -200 (extreme oversold) or CCI crosses above -100.
    SELL: CCI > 200 (extreme overbought) or CCI crosses below 100.
    """
    if df.empty or len(df) < 20:
        return "HOLD", 0, "Insufficient data for CCI"

    cci_ind = CCIIndicator(df["High"], df["Low"], df["Close"], window=20)
    cci = cci_ind.cci()

    if pd.isna(cci.iloc[-1]):
        return "HOLD", 0, "CCI not ready"

    cci_cur = cci.iloc[-1]
    cci_prev = cci.iloc[-2]

    if cci_cur < -200:
        return "BUY", 75, f"CCI: Deep oversold ({cci_cur:.0f})"
    elif cci_prev <= -100 and cci_cur > -100:
        return "BUY", 65, f"CCI: Crossed above -100 ({cci_cur:.0f})"
    elif cci_cur > 200:
        return "SELL", 75, f"CCI: Extreme overbought ({cci_cur:.0f})"
    elif cci_prev >= 100 and cci_cur < 100:
        return "SELL", 65, f"CCI: Crossed below 100 ({cci_cur:.0f})"
    elif cci_cur > 100:
        return "SELL", 35, f"CCI: Overbought zone ({cci_cur:.0f})"
    elif cci_cur < -100:
        return "BUY", 35, f"CCI: Oversold zone ({cci_cur:.0f})"
    return "HOLD", 15, f"CCI: Neutral ({cci_cur:.0f})"


def ultimate_oscillator_strategy(df):
    """
    Ultimate Oscillator — Multi-timeframe momentum (7, 14, 28 periods).
    BUY: Bullish divergence or value below 30 with upturn.
    SELL: Bearish divergence or value above 70 with downturn.
    """
    if df.empty or len(df) < 30:
        return "HOLD", 0, "Insufficient data for Ultimate Oscillator"

    uo = UltimateOscillator(df["High"], df["Low"], df["Close"],
                            window1=7, window2=14, window3=28,
                            weight1=4.0, weight2=2.0, weight3=1.0)
    uo_v = uo.ultimate_oscillator()

    if pd.isna(uo_v.iloc[-1]):
        return "HOLD", 0, "UO not ready"

    cur = uo_v.iloc[-1]
    prev = uo_v.iloc[-2]

    if cur < 30 and cur > prev:
        return "BUY", 70, f"UO: Oversold ({cur:.0f}) with upturn"
    elif cur > 70 and cur < prev:
        return "SELL", 70, f"UO: Overbought ({cur:.0f}) with downturn"
    elif cur < 25:
        return "BUY", 45, f"UO: Deep oversold ({cur:.0f})"
    elif cur > 75:
        return "SELL", 45, f"UO: Deep overbought ({cur:.0f})"
    return "HOLD", 20, f"UO: Neutral ({cur:.0f})"


# ═══════════════════════════════════════════════════════════════
# 3. VOLUME-BASED STRATEGIES
# ═══════════════════════════════════════════════════════════════

def mfi_strategy(df):
    """
    Money Flow Index — Volume-weighted RSI.
    BUY: MFI < 20 (oversold, accumulation)
    SELL: MFI > 80 (overbought, distribution)
    """
    if df.empty or len(df) < 20:
        return "HOLD", 0, "Insufficient data for MFI"

    mfi_ind = MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=14)
    mfi = mfi_ind.money_flow_index()

    if pd.isna(mfi.iloc[-1]):
        return "HOLD", 0, "MFI not ready"

    cur = mfi.iloc[-1]
    prev = mfi.iloc[-2]

    if cur < 20 and cur > prev:
        return "BUY", 75, f"MFI: Oversold with accumulation ({cur:.0f})"
    elif cur > 80 and cur < prev:
        return "SELL", 75, f"MFI: Overbought with distribution ({cur:.0f})"
    elif cur < 20:
        return "BUY", 45, f"MFI: Oversold ({cur:.0f})"
    elif cur > 80:
        return "SELL", 45, f"MFI: Overbought ({cur:.0f})"
    elif cur < 30:
        return "BUY", 25, f"MFI: Approaching oversold ({cur:.0f})"
    elif cur > 70:
        return "SELL", 25, f"MFI: Approaching overbought ({cur:.0f})"
    return "HOLD", 15, f"MFI: Neutral ({cur:.0f})"


def obv_strategy(df):
    """
    On-Balance Volume — Volume confirmation of price trend.
    BUY: OBV rising faster than price (accumulation).
    SELL: OBV falling while price rising (divergence).
    """
    if df.empty or len(df) < 20:
        return "HOLD", 0, "Insufficient data for OBV"

    obv_ind = OnBalanceVolumeIndicator(df["Close"], df["Volume"])
    obv = obv_ind.on_balance_volume()

    if len(obv) < 15:
        return "HOLD", 0, "OBV not ready"

    price_trend = df["Close"].iloc[-1] > df["Close"].iloc[-15]
    obv_trend = obv.iloc[-1] > obv.iloc[-15]

    if price_trend and obv_trend:
        return "BUY", 60, "OBV: Volume confirms uptrend (accumulation)"
    elif not price_trend and not obv_trend:
        return "SELL", 60, "OBV: Volume confirms downtrend (distribution)"
    elif price_trend and not obv_trend:
        return "SELL", 70, "OBV: Bearish divergence (price up, volume down)"
    elif not price_trend and obv_trend:
        return "BUY", 70, "OBV: Bullish divergence (price down, volume up)"
    return "HOLD", 25, "OBV: Mixed"


def vpt_strategy(df):
    """
    Volume Price Trend — Cumulative volume-weighted price trend.
    BUY: VPT rising (accumulation).
    SELL: VPT falling (distribution).
    """
    if df.empty or len(df) < 20:
        return "HOLD", 0, "Insufficient data for VPT"

    vpt_ind = VolumePriceTrendIndicator(df["Close"], df["Volume"])
    vpt = vpt_ind.volume_price_trend()

    if len(vpt) < 20:
        return "HOLD", 0, "VPT not ready"

    vpt_short = vpt.iloc[-1] > vpt.iloc[-5]
    vpt_medium = vpt.iloc[-1] > vpt.iloc[-20]
    price_up = df["Close"].iloc[-1] > df["Close"].iloc[-20]

    if vpt_short and vpt_medium:
        return "BUY", 55, "VPT: Rising (accumulation confirmed)"
    elif not vpt_short and not vpt_medium:
        return "SELL", 55, "VPT: Falling (distribution confirmed)"
    elif vpt_medium and not price_up:
        return "BUY", 65, "VPT: Bullish divergence (VPT up, price down)"
    elif not vpt_medium and price_up:
        return "SELL", 65, "VPT: Bearish divergence (VPT down, price up)"
    return "HOLD", 20, "VPT: Mixed"


# ═══════════════════════════════════════════════════════════════
# 4. VOLATILITY STRATEGIES
# ═══════════════════════════════════════════════════════════════

def keltner_strategy(df):
    """
    Keltner Channels — Volatility-based envelope.
    BUY: Price hits lower channel (mean reversion up).
    SELL: Price hits upper channel (mean reversion down).
    """
    if df.empty or len(df) < 20:
        return "HOLD", 0, "Insufficient data for Keltner"

    keltner = KeltnerChannel(df["High"], df["Low"], df["Close"], window=20, window_atr=10)
    upper = keltner.keltner_channel_hband()
    lower = keltner.keltner_channel_lband()
    mid = keltner.keltner_channel_mband()

    if pd.isna(upper.iloc[-1]):
        return "HOLD", 0, "Keltner not ready"

    price = df["Close"].iloc[-1]
    upper_v = upper.iloc[-1]
    lower_v = lower.iloc[-1]

    if price >= upper_v:
        return "SELL", 60, f"Keltner: Price hit upper band ({price:.2f})"
    elif price <= lower_v:
        return "BUY", 60, f"Keltner: Price hit lower band ({price:.2f})"
    elif price > mid.iloc[-1]:
        return "HOLD", 25, "Keltner: Above middle line"
    else:
        return "HOLD", 25, "Keltner: Below middle line"


def donchian_strategy(df):
    """
    Donchian Channels — Breakout system (Turtle Trading).
    BUY: Price breaks above 20-day high (new high).
    SELL: Price breaks below 20-day low (new low).
    """
    if df.empty or len(df) < 20:
        return "HOLD", 0, "Insufficient data for Donchian"

    donchian = DonchianChannel(df["High"], df["Low"], df["Close"], window=20)
    upper = donchian.donchian_channel_hband()
    lower = donchian.donchian_channel_lband()

    if pd.isna(upper.iloc[-1]):
        return "HOLD", 0, "Donchian not ready"

    price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[-2]
    upper_v = upper.iloc[-1]
    lower_v = lower.iloc[-1]
    prev_high = df["High"].rolling(20).max().iloc[-2]
    prev_low = df["Low"].rolling(20).min().iloc[-2]

    # Breakout = price crosses above the channel high
    if prev_price <= prev_high and price > upper_v:
        return "BUY", 75, f"Donchian: Breakout above {upper_v:.2f} (20-day high)"
    elif prev_price >= prev_low and price < lower_v:
        return "SELL", 75, f"Donchian: Breakdown below {lower_v:.2f} (20-day low)"
    elif price > upper_v:
        return "BUY", 45, "Donchian: Above 20-day high (strong trend)"
    elif price < lower_v:
        return "SELL", 45, "Donchian: Below 20-day low (weakness)"
    return "HOLD", 20, "Donchian: Inside channel"


def chandelier_exit_strategy(df):
    """
    Chandelier Exit — Volatility-based trailing stop.
    BUY when price > chandelier exit long (uptrend).
    SELL when price < chandelier exit short (downtrend).
    """
    if df.empty or len(df) < 22 or "ATR" not in df.columns:
        return "HOLD", 0, "Insufficient data for Chandelier Exit"

    high_22 = df["High"].rolling(22).max()
    low_22 = df["Low"].rolling(22).min()
    atr = df["ATR"]

    # Chandelier Exit Long = 22-day High - ATR * 3
    ce_long = high_22 - atr * 3
    # Chandelier Exit Short = 22-day Low + ATR * 3
    ce_short = low_22 + atr * 3

    price = df["Close"].iloc[-1]
    ce_long_v = ce_long.iloc[-1]
    ce_short_v = ce_short.iloc[-1]

    if pd.isna(ce_long_v):
        return "HOLD", 0, "Chandelier not ready"

    if price > ce_long_v:
        return "BUY", 55, f"Chandelier: Above long exit ({ce_long_v:.2f}) — uptrend"
    elif price < ce_short_v:
        return "SELL", 55, f"Chandelier: Below short exit ({ce_short_v:.2f}) — downtrend"
    return "HOLD", 25, "Chandelier: Between exits — consolidation"


# ═══════════════════════════════════════════════════════════════
# 5. MOMENTUM STRATEGIES
# ═══════════════════════════════════════════════════════════════

def awesome_oscillator_strategy(df):
    """
    Awesome Oscillator — Momentum (5 - 34 period).
    BUY: AO crosses above zero line (zero line crossover).
         OR saucer bottom: AO > prev > prev_prev (all positive, dip then rise).
    SELL: AO crosses below zero line.
         OR saucer top: AO < prev < prev_prev (all negative, peak then fall).
    """
    if df.empty or len(df) < 35:
        return "HOLD", 0, "Insufficient data for Awesome Oscillator"

    ao = AwesomeOscillatorIndicator(df["High"], df["Low"], window1=5, window2=34)
    ao_v = ao.awesome_oscillator()

    if pd.isna(ao_v.iloc[-1]):
        return "HOLD", 0, "AO not ready"

    cur = ao_v.iloc[-1]
    prev = ao_v.iloc[-2]
    prev2 = ao_v.iloc[-3] if len(ao_v) > 3 else prev

    # Zero line crossover
    if prev < 0 and cur > 0:
        return "BUY", 70, f"AO: Crossed above zero ({cur:.2f})"
    elif prev > 0 and cur < 0:
        return "SELL", 70, f"AO: Crossed below zero ({cur:.2f})"

    # Saucer pattern
    if cur > prev > prev2 and cur > 0:
        return "BUY", 60, "AO: Saucer bottom (momentum reviving)"
    elif cur < prev < prev2 and cur < 0:
        return "SELL", 60, "AO: Saucer top (momentum fading)"

    # Direction
    if cur > 0 and cur > prev:
        return "BUY", 35, f"AO: Positive momentum strengthening ({cur:.2f})"
    elif cur < 0 and cur < prev:
        return "SELL", 35, f"AO: Negative momentum deepening ({cur:.2f})"
    elif cur > 0:
        return "BUY", 25, "AO: Positive territory"
    elif cur < 0:
        return "SELL", 25, "AO: Negative territory"
    return "HOLD", 15, "AO: Flat"


def roc_strategy(df):
    """
    Rate of Change — Momentum speed.
    BUY: ROC accelerating upward (momentum increasing).
    SELL: ROC accelerating downward (momentum decreasing).
    """
    if df.empty or len(df) < 20:
        return "HOLD", 0, "Insufficient data for ROC"

    roc_ind = ROCIndicator(df["Close"], window=12)
    roc = roc_ind.roc()

    if pd.isna(roc.iloc[-1]):
        return "HOLD", 0, "ROC not ready"

    cur = roc.iloc[-1]
    prev = roc.iloc[-2]

    if cur > 5 and cur > prev:
        return "BUY", 55, f"ROC: Strong positive momentum ({cur:.1f})"
    elif cur < -5 and cur < prev:
        return "SELL", 55, f"ROC: Strong negative momentum ({cur:.1f})"
    elif cur > 0 and cur > prev:
        return "BUY", 35, f"ROC: Momentum improving ({cur:.1f})"
    elif cur < 0 and cur < prev:
        return "SELL", 35, f"ROC: Momentum declining ({cur:.1f})"
    elif cur > 2:
        return "BUY", 25, f"ROC: Positive ({cur:.1f})"
    elif cur < -2:
        return "SELL", 25, f"ROC: Negative ({cur:.1f})"
    return "HOLD", 15, f"ROC: Neutral ({cur:.1f})"


def mass_index_strategy(df):
    """
    Mass Index — Volatility reversal.
    BUY when Mass Index > 27 then drops below 26.5 (reversal up).
    SELL when Mass Index > 27 (high volatility, trend exhaustion).
    """
    if df.empty or len(df) < 30:
        return "HOLD", 0, "Insufficient data for Mass Index"

    mi = MassIndex(df["High"], df["Low"], window_fast=9, window_slow=25)
    mi_v = mi.mass_index()

    if pd.isna(mi_v.iloc[-1]):
        return "HOLD", 0, "MI not ready"

    cur = mi_v.iloc[-1]
    prev = mi_v.iloc[-2]

    # Reversal bulge: MI goes above 27, then drops below 26.5
    recent_high = mi_v.tail(10).max()
    if recent_high > 27 and cur < 26.5 and prev >= 26.5:
        return "BUY", 60, "Mass Index: Reversal bulge complete (trend reversal up)"
    elif cur > 27:
        return "SELL", 40, f"Mass Index: High volatility ({cur:.1f}) - potential reversal"
    return "HOLD", 20, f"Mass Index: Normal ({cur:.1f})"


def tsi_strategy(df):
    """
    True Strength Index — Double-smoothed momentum.
    BUY: TSI crosses above zero or bullish divergence.
    SELL: TSI crosses below zero or bearish divergence.
    """
    if df.empty or len(df) < 30:
        return "HOLD", 0, "Insufficient data for TSI"

    tsi_ind = TSIIndicator(df["Close"], window_slow=25, window_fast=13)
    tsi = tsi_ind.tsi()
    tsi_signal = tsi.rolling(7).mean()

    if pd.isna(tsi.iloc[-1]):
        return "HOLD", 0, "TSI not ready"

    cur = tsi.iloc[-1]
    sig = tsi_signal.iloc[-1]
    prev = tsi.iloc[-2]
    prev_sig = tsi_signal.iloc[-2]

    if prev <= prev_sig and cur > sig:
        fold_change = abs(cur - prev)
        if cur > 0:
            return "BUY", 65, f"TSI: Bullish crossover (momentum: {cur:.1f})"
        else:
            return "BUY", 50, f"TSI: Bullish crossover from negative ({cur:.1f})"
    elif prev >= prev_sig and cur < sig:
        if cur < 0:
            return "SELL", 65, f"TSI: Bearish crossover (momentum: {cur:.1f})"
        else:
            return "SELL", 50, f"TSI: Bearish crossover from positive ({cur:.1f})"
    elif cur > sig:
        return "BUY", 30, f"TSI: Above signal line ({cur:.1f})"
    elif cur < sig:
        return "SELL", 30, f"TSI: Below signal line ({cur:.1f})"
    return "HOLD", 15, "TSI: Flat"


def wma_strategy(df):
    """
    Weighted Moving Average crossover.
    BUY: Price > WMA10 > WMA30 (bullish alignment).
    SELL: Price < WMA10 < WMA30 (bearish alignment).
    Crossovers for stronger signals.
    """
    if df.empty or len(df) < 30:
        return "HOLD", 0, "Insufficient data for WMA"

    wma10 = WMAIndicator(df["Close"], window=10).wma()
    wma30 = WMAIndicator(df["Close"], window=30).wma()

    if pd.isna(wma10.iloc[-1]):
        return "HOLD", 0, "WMA not ready"

    price = df["Close"].iloc[-1]
    w10 = wma10.iloc[-1]
    w30 = wma30.iloc[-1]
    p_w10 = wma10.iloc[-2]
    p_w30 = wma30.iloc[-2]

    if p_w10 <= p_w30 and w10 > w30:
        return "BUY", 70, "WMA: Bullish crossover (WMA10 > WMA30)"
    elif p_w10 >= p_w30 and w10 < w30:
        return "SELL", 70, "WMA: Bearish crossover (WMA10 < WMA30)"
    elif price > w10 > w30:
        return "BUY", 50, "WMA: Bullish alignment (price > WMA10 > WMA30)"
    elif price < w10 < w30:
        return "SELL", 50, "WMA: Bearish alignment (price < WMA10 < WMA30)"
    return "HOLD", 20, "WMA: Mixed"


def tema_strategy(df):
    """
    Triple EMA — Smoother EMA calculated as:
    TEMA = 3*EMA1 - 3*EMA2 + EMA3
    BUY when TEMA crosses above price (momentum up).
    SELL when TEMA crosses below price (momentum down).
    """
    if df.empty or len(df) < 30:
        return "HOLD", 0, "Insufficient data for TEMA"

    close = df["Close"]
    ema1 = EMAIndicator(close, window=10).ema_indicator()
    ema2 = EMAIndicator(ema1, window=10).ema_indicator()
    ema3 = EMAIndicator(ema2, window=10).ema_indicator()
    tema = 3 * ema1 - 3 * ema2 + ema3

    if pd.isna(tema.iloc[-1]):
        return "HOLD", 0, "TEMA not ready"

    price = close.iloc[-1]
    t = tema.iloc[-1]
    p_t = tema.iloc[-2]

    if p_t <= price and t > price:
        return "BUY", 55, f"TEMA crossed above price ({t:.1f} > {price:.1f})"
    elif p_t >= price and t < price:
        return "SELL", 55, f"TEMA crossed below price ({t:.1f} < {price:.1f})"
    return "HOLD", 20, "TEMA: No crossover"


# ═══════════════════════════════════════════════════════════════
# 6. BREAKOUT & PATTERN STRATEGIES
# ═══════════════════════════════════════════════════════════════

def support_resistance_breakout(df):
    """
    S/R Breakout — Price breaks key support/resistance levels.
    BUY: Price breaks above recent resistance with volume.
    SELL: Price breaks below recent support with volume.
    """
    if df.empty or len(df) < 20:
        return "HOLD", 0, "Insufficient data for S/R"

    recent_high = df["High"].tail(20).max()
    recent_low = df["Low"].tail(20).min()
    price = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2]
    vol_avg = df["Volume"].tail(20).mean()
    vol_cur = df["Volume"].iloc[-1]
    vol_surge = vol_cur > vol_avg * 1.3

    if prev <= recent_high and price > recent_high:
        conf = 70 if vol_surge else 55
        return "BUY", conf, f"S/R: Broke resistance {recent_high:.2f}{' with volume' if vol_surge else ''}"
    elif prev >= recent_low and price < recent_low:
        conf = 70 if vol_surge else 55
        return "SELL", conf, f"S/R: Broke support {recent_low:.2f}{' with volume' if vol_surge else ''}"
    return "HOLD", 15, "S/R: No breakout"


def volume_breakout_strategy(df):
    """
    Volume Breakout — Price + volume surge.
    BUY: Price up > 2% + volume > 1.5x average (strong buying).
    SELL: Price down > 2% + volume > 1.5x average (strong selling).
    """
    if df.empty or len(df) < 20:
        return "HOLD", 0, "Insufficient data"

    price = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2]
    change_pct = ((price - prev) / prev) * 100
    vol_cur = df["Volume"].iloc[-1]
    vol_avg = df["Volume"].tail(20).mean()

    if vol_avg > 0 and vol_cur > vol_avg * 1.5 and abs(change_pct) > 1.5:
        if change_pct > 0:
            return "BUY", 70, f"Volume breakout: +{change_pct:.1f}% with {vol_cur/vol_avg:.1f}x avg volume"
        else:
            return "SELL", 70, f"Volume breakdown: {change_pct:.1f}% with {vol_cur/vol_avg:.1f}x avg volume"

    if vol_avg > 0 and vol_cur > vol_avg * 2:
        if change_pct > 0:
            return "BUY", 55, f"High volume surge ({vol_cur/vol_avg:.1f}x) with gain"
        elif change_pct < 0:
            return "SELL", 55, f"High volume surge ({vol_cur/vol_avg:.1f}x) with loss"
    return "HOLD", 10, "Volume: Normal"


# ═══════════════════════════════════════════════════════════════
# 7. MASTER LIST — All strategies for the AI Analyst
# ═══════════════════════════════════════════════════════════════

ALL_STRATEGIES = {
    # Trend Following
    "SuperTrend": supertrend_strategy,
    "Ichimoku": ichimoku_strategy,
    "PSAR": psar_strategy,
    "Vortex": vortex_strategy,
    "TRIX": trix_strategy,
    "KST": kst_strategy,
    "KAMA": kama_strategy,
    # Mean Reversion
    "Stochastic": stochastic_strategy,
    "Williams %R": williams_r_strategy,
    "CCI": cci_strategy,
    "Ultimate Osc": ultimate_oscillator_strategy,
    # Volume-based
    "MFI": mfi_strategy,
    "OBV": obv_strategy,
    "VPT": vpt_strategy,
    # Volatility
    "Keltner": keltner_strategy,
    "Donchian": donchian_strategy,
    "Chandelier": chandelier_exit_strategy,
    # Momentum
    "Awesome Osc": awesome_oscillator_strategy,
    "ROC": roc_strategy,
    "TSI": tsi_strategy,
    "WMA Cross": wma_strategy,
    "TEMA": tema_strategy,
    # Breakout
    "S/R Breakout": support_resistance_breakout,
    "Volume Breakout": volume_breakout_strategy,
}


def run_all_strategies(df):
    """
    Run ALL strategies on a DataFrame.
    Returns: list of (strategy_name, signal, confidence, reason)
    """
    results = []
    for name, strategy_fn in ALL_STRATEGIES.items():
        try:
            sig, conf, reason = strategy_fn(df)
            results.append((name, sig, conf, reason))
        except Exception as e:
            results.append((name, "HOLD", 0, f"Error: {e}"))
    return results


def get_strategy_votes(results):
    """Count BUY, SELL, HOLD votes across all strategies."""
    votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
    weighted_buy = 0
    weighted_sell = 0
    total_conf = 0

    for name, sig, conf, reason in results:
        votes[sig] += 1
        if sig == "BUY":
            weighted_buy += conf
        elif sig == "SELL":
            weighted_sell += conf
        total_conf += conf

    return votes, weighted_buy, weighted_sell, total_conf
