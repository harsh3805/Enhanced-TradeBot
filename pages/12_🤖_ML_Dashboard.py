"""
ML Dashboard — Monitor ML model status, feature importance, walk-forward
performance, and strategy leaderboard. Controls for training/retraining.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Import new modules
from core.ml_engine import MLEngine, WalkForwardConfig
from core.ml_features import FeatureEngineeringEngine
from core.data_fetcher import get_stock_data
from core.analyzer import calculate_all_indicators
from core.signal_combiner import SmartSignalCombiner
from core.volatility_model import VolatilityModel
from core.cost_model import IndianMarketCostModel
from utils.config import (
    POPULAR_INDIAN, ML_MODEL_DIR,
    ML_FORWARD_HORIZON, ML_MIN_TRAIN_DAYS, ML_TEST_DAYS
)

st.set_page_config(page_title="ML Dashboard", page_icon="🤖", layout="wide")

# ── Page Header ──────────────────────────────────────────────
st.markdown("## 🤖 ML Trading Dashboard")
st.caption("Machine learning signal engine — monitor models, view features, retrain")

# ── Sidebar Controls ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ML Settings")
    symbol = st.selectbox("Stock Symbol", POPULAR_INDIAN[:10], index=0)
    retrain = st.button("🔄 Retrain Model", type="primary")
    period = st.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
    st.divider()
    st.subheader("📊 Quick Stats")

    # Load or create engine
    engine = MLEngine()
    feature_engine = FeatureEngineeringEngine(forward_horizon=ML_FORWARD_HORIZON)

    if engine.needs_retrain(symbol):
        st.warning("⚠️ Model needs retraining")
    else:
        st.success("✅ Model is current")

# ── Main Content ─────────────────────────────────────────────
st.divider()

# ── Section 1: ML Model Status ──────────────────────────────
st.header("📈 ML Model Status")

col1, col2, col3, col4 = st.columns(4)

# Fetch data and check model
df = get_stock_data(symbol, period=period)
model_loaded = engine.load_model(symbol)
oos_perf = engine.get_cumulative_oos_performance()

with col1:
    st.metric("Model Status", "Trained ✅" if model_loaded else "Untrained ❌")
with col2:
    acc = oos_perf.get("accuracy", 0)
    st.metric("OOS Accuracy", f"{acc:.1%}" if acc > 0 else "N/A")
with col3:
    sharpe = oos_perf.get("sharpe", 0)
    st.metric("OOS Sharpe", f"{sharpe:.2f}" if sharpe > 0 else "N/A")
with col4:
    n_windows = oos_perf.get("n_windows", 0)
    st.metric("Walk-Forward Windows", n_windows if n_windows > 0 else "N/A")

# ── Section 2: Train or Retrain ─────────────────────────────
if retrain and not df.empty:
    with st.spinner(f"Training ML model on {symbol}... This may take 30-60 seconds."):
        progress = st.progress(0, text="Generating features...")
        progress.progress(10)

        df_ind = calculate_all_indicators(df)
        progress.progress(30, text="Running walk-forward validation...")
        perf = engine.train(df_ind, feature_engine)
        progress.progress(80, text="Saving model...")
        engine.save_model(symbol)
        progress.progress(100, text="Training complete!")
        st.success(f"✅ Model trained on {symbol} with {oos_perf.get('n_windows', 0)} walk-forward windows")
        st.rerun()

# ── Section 3: Feature Importance ───────────────────────────
st.header("🔍 Feature Importance")

fi = engine.get_feature_importance()
if not fi.empty:
    top_n = min(15, len(fi))
    top_features = fi.head(top_n)

    fig_data = top_features.set_index("feature")
    st.bar_chart(fig_data, color="#00D4AA", horizontal=True)

    st.markdown("---")
    with st.expander("View All Feature Descriptions"):
        descriptions = feature_engine.get_feature_descriptions()
        desc_df = pd.DataFrame([
            {"Feature": k, "Description": v}
            for k, v in sorted(descriptions.items())
        ])
        st.dataframe(desc_df, use_container_width=True, hide_index=True)
else:
    st.info("Train a model to see feature importance.")

# ── Section 4: ML Prediction for Current Stock ──────────────
st.header(f"🎯 ML Prediction: {symbol}")

if model_loaded and not df.empty:
    from core.ml_engine import PredictionResult
    result = engine.predict(calculate_all_indicators(df), feature_engine)

    if result.is_trained:
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        with pred_col1:
            color = "🟢" if result.signal == "BUY" else ("🔴" if result.signal == "SELL" else "🟡")
            st.metric("ML Signal", f"{color} {result.signal}")
        with pred_col2:
            st.metric("Confidence", f"{result.confidence}%")
        with pred_col3:
            st.metric("Model Agreement", f"{result.model_agreement}%")

        # Probability breakdown
        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame({
            "Signal": list(result.probabilities.keys()),
            "Probability": list(result.probabilities.values()),
        })
        st.bar_chart(prob_df.set_index("Signal"), color="#00A3FF")

        # Top features used
        if result.top_features:
            st.subheader("Top Features Driving This Prediction")
            feat_df = pd.DataFrame(result.top_features, columns=["Feature", "Importance"])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Model not yet trained. Click 'Retrain Model' in sidebar.")
else:
    st.info("Load a trained model or retrain to see predictions.")

# ── Section 5: Volatility & Cost Analysis ────────────────────
st.header("🌊 Volatility & Cost Analysis")

if not df.empty:
    vol_col1, vol_col2 = st.columns(2)

    with vol_col1:
        st.subheader("Volatility Regime")
        vol_model = VolatilityModel()
        df_ind = calculate_all_indicators(df) if "ATR" not in df.columns else df
        vol_est = vol_model.forecast_volatility(df_ind)

        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            st.metric("Current Vol", f"{vol_est.current_realized_vol:.1f}%")
        with v_col2:
            regime_emoji = {"LOW": "🟢", "NORMAL": "🟡", "HIGH": "🟠", "EXTREME": "🔴"}
            st.metric("Regime", f"{regime_emoji.get(vol_est.vol_regime.name, '')} {vol_est.vol_regime.name}")
        with v_col3:
            st.metric("Percentile", f"{vol_est.percentile_rank:.0f}th")

        st.metric("GARCH 1d Forecast", f"{vol_est.garch_forecast_1d:.1f}%")
        st.metric("Vol of Vol", f"{vol_est.vol_of_vol:.2f}")

    with vol_col2:
        st.subheader("Transaction Costs (Indian Market)")
        cost_model = IndianMarketCostModel()
        price = float(df["Close"].iloc[-1])

        # Show costs for a typical trade
        costs = cost_model.calculate_round_trip_costs(price, price * 1.05, 100, "delivery")
        st.metric("Round-Trip Cost (₹100 share)", f"₹{costs.round_trip:.2f}")
        st.metric("Break-Even Move", f"{costs.break_even_pct:.2f}%")

        with st.expander("Full Cost Breakdown"):
            st.markdown("""
| Component | Buy | Sell |
|---|---|---|
| Brokerage | ₹{:.2f} | ₹{:.2f} |
| STT | ₹{:.2f} | ₹{:.2f} |
| Stamp Duty | ₹{:.2f} | ₹{:.2f} |
| Exchange Charges | ₹{:.2f} | ₹{:.2f} |
| SEBI Charges | ₹{:.2f} | ₹{:.2f} |
| GST | ₹{:.2f} | ₹{:.2f} |
| **Total** | **₹{:.2f}** | **₹{:.2f}** |
            """.format(
                costs.buy_costs.brokerage, costs.sell_costs.brokerage,
                costs.buy_costs.stt, costs.sell_costs.stt,
                costs.buy_costs.stamp_duty, costs.sell_costs.stamp_duty,
                costs.buy_costs.exchange_charges, costs.sell_costs.exchange_charges,
                costs.buy_costs.sebi_charges, costs.sell_costs.sebi_charges,
                costs.buy_costs.gst, costs.sell_costs.gst,
                costs.buy_costs.total, costs.sell_costs.total,
            ))

# ── Section 6: Strategy Performance Leaderboard ─────────────
st.header("🏆 Strategy Performance")

combiner = SmartSignalCombiner()
st.info("Strategy performance is tracked dynamically as you use the system. Performance data resets when the page reloads.")

# Show what the regime-based multipliers would be
from core.market_regime import detect_regime
if not df.empty:
    df_reg = calculate_all_indicators(df)
    regime_info = detect_regime(df_reg)
    regime = regime_info["regime"]

    st.subheader(f"Current Regime: {regime}")
    from core.signal_combiner import REGIME_MULTIPLIERS
    mult = REGIME_MULTIPLIERS.get(regime, {})

    if mult:
        mult_df = pd.DataFrame([
            {"Strategy": k, "Weight Multiplier": v}
            for k, v in sorted(mult.items(), key=lambda x: -x[1])
        ])
        st.dataframe(mult_df, use_container_width=True, hide_index=True)
    else:
        st.info("No regime-specific adjustments for current market conditions.")

# ── Section 7: System Architecture ──────────────────────────
st.header("🧠 System Overview")
st.markdown("""
This ML trading system layers machine learning on top of 24+ traditional technical strategies:

| Component | Purpose |
|---|---|
| **Feature Engineering** | 50+ features across 5 categories (momentum, mean reversion, trend, volatility, volume) |
| **ML Ensemble** | RandomForest + HistGradientBoosting + Ridge classifier |
| **Walk-Forward Validation** | Train on 1 year → test on next quarter → roll forward (no look-ahead bias) |
| **Smart Signal Combiner** | Dynamic weighting by strategy hit rate + profit factor, regime multipliers |
| **Adaptive Parameters** | Indicator periods auto-adjust: faster in high volatility, slower in low |
| **Cost Model** | Full Indian market costs: STT, stamp duty, SEBI, GST, slippage by liquidity tier |
| **Monte Carlo** | 1000 bootstrap simulations for confidence intervals |
""")
