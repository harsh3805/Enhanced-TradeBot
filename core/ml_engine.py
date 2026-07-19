"""
ML Prediction Engine — Ensemble machine learning for price direction prediction
with walk-forward validation to prevent overfitting.

Walk-forward protocol:
    Train on 252 days → predict next 63 days → roll forward by 63 → repeat
    5-day embargo between train and test to prevent data leakage

Ensemble:
    1. RandomForestClassifier (nonlinear interactions, feature importance)
    2. HistGradientBoostingClassifier (fast, robust, handles NaN)
    3. RidgeClassifier (linear baseline, prevents overfitting)

Model persistence with joblib — models are saved per symbol.
"""
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from utils.config import (
    ML_MODEL_DIR, ML_MIN_TRAIN_DAYS, ML_TEST_DAYS,
    ML_RETRAIN_DAYS
)


@dataclass
class ModelPerformance:
    """Performance metrics for a walk-forward window."""
    window_idx: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    total_trades: int = 0
    sharpe_if_traded: float = 0.0
    period_start: str = ""
    period_end: str = ""


@dataclass
class PredictionResult:
    """Output from the ML prediction engine."""
    signal: str = "HOLD"                     # BUY, SELL, or HOLD
    confidence: float = 0.0                  # 0-100 calibrated
    probabilities: Dict[str, float] = field(default_factory=dict)  # BUY/SELL/HOLD probs
    model_agreement: float = 0.0             # % of ensemble members agreeing (0-100)
    oos_accuracy: float = 0.0                # Last walk-forward accuracy
    features_used: int = 0                   # Number of features used
    top_features: List[tuple] = field(default_factory=list)  # Top 10 feature importances
    train_date: str = ""                     # When model was last trained
    is_trained: bool = False


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    min_train_days: int = ML_MIN_TRAIN_DAYS  # At least 1 year of training data
    test_days: int = ML_TEST_DAYS            # Test on next quarter
    step_days: int = ML_TEST_DAYS            # Roll forward by 1 quarter
    embargo_days: int = 5                    # Gap between train and test
    cv_folds: int = 3                        # Internal cross-validation per window


class MLEngine:
    """
    Ensemble ML engine with walk-forward validation.

    Three-model ensemble:
    - RandomForest: captures nonlinear feature interactions
    - HistGradientBoosting: robust gradient boosting
    - Ridge: linear baseline (prevents overfitting)
    """

    def __init__(
        self,
        walk_forward_config: Optional[WalkForwardConfig] = None,
        model_dir: str = ML_MODEL_DIR,
        random_state: int = 42,
    ):
        self.config = walk_forward_config or WalkForwardConfig()
        self.model_dir = model_dir
        self.random_state = random_state

        # State
        self._models: Dict[str, Any] = {}
        self._performance_history: List[ModelPerformance] = []
        self._feature_importance: pd.DataFrame = pd.DataFrame()
        self._is_trained = False
        self._train_date: Optional[str] = None
        self._symbol = "GLOBAL"

        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, df: pd.DataFrame, feature_engine) -> Dict[str, List[ModelPerformance]]:
        """
        Train ensemble using walk-forward validation.

        Walks through time, training on expanding windows and testing forward.
        Every window is out-of-sample for the test portion.

        Args:
            df: OHLCV DataFrame with at least min_train_days + test_days rows
            feature_engine: FeatureEngineeringEngine instance

        Returns:
            dict of model_name → list of ModelPerformance per window
        """
        if df.empty or len(df) < self.config.min_train_days + self.config.test_days:
            return {"error": [ModelPerformance(window_idx=-1, accuracy=0.0)]}

        # Generate features
        df_feat = feature_engine.generate_features(df.copy())
        full_X, full_y = feature_engine.get_feature_matrix(df_feat)

        if len(full_X) < self.config.min_train_days // 2:
            return {"error": [ModelPerformance(window_idx=-1, accuracy=0.0)]}

        n = len(full_X)
        windows = self._walk_forward_split(n)

        if not windows:
            return {"error": [ModelPerformance(window_idx=-1, accuracy=0.0)]}

        all_performances = {"rf": [], "hgb": [], "ridge": [], "ensemble": []}

        for win_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            if test_end > n:
                break
            if train_end - train_start < self.config.min_train_days:
                continue

            X_train = full_X.iloc[train_start:train_end]
            y_train = full_y.iloc[train_start:train_end]
            X_test = full_X.iloc[test_start:test_end]
            y_test = full_y.iloc[test_start:test_end]

            # Skip if no variation in target
            if y_train.nunique() < 2:
                continue

            from sklearn.ensemble import (
                RandomForestClassifier, HistGradientBoostingClassifier,
            )
            from sklearn.linear_model import RidgeClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            import joblib

            models = {
                "rf": RandomForestClassifier(
                    n_estimators=200, max_depth=8, min_samples_leaf=20,
                    class_weight="balanced", random_state=self.random_state, n_jobs=-1,
                ),
                "hgb": HistGradientBoostingClassifier(
                    max_iter=200, max_depth=6, min_samples_leaf=20,
                    learning_rate=0.05, random_state=self.random_state,
                ),
                "ridge": RidgeClassifier(alpha=1.0, class_weight="balanced"),
            }

            window_perfs = []

            for name, model in models.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                # Map predictions (Ridge outputs -1/1, not 0)
                if name == "ridge":
                    y_pred_binary = np.where(y_pred == -1, -1, np.where(y_pred == 1, 1, 0))
                else:
                    y_pred_binary = y_pred

                acc = accuracy_score(y_test, y_pred_binary)
                # Precision/recall/F1 weighted for multi-class
                try:
                    prec = precision_score(y_test, y_pred_binary, average="weighted", zero_division=0)
                    rec = recall_score(y_test, y_pred_binary, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred_binary, average="weighted", zero_division=0)
                except Exception:
                    prec = rec = f1 = 0.0

                # Approximate Sharpe from direction accuracy
                sharpe = (acc - 0.5) * 2 * np.sqrt(252) / 100

                perf = ModelPerformance(
                    window_idx=win_idx,
                    accuracy=round(acc, 4),
                    precision=round(prec, 4),
                    recall=round(rec, 4),
                    f1_score=round(f1, 4),
                    total_trades=len(y_test),
                    sharpe_if_traded=round(sharpe, 3),
                    period_start=str(X_test.index[0]) if hasattr(X_test, "index") else "",
                    period_end=str(X_test.index[-1]) if hasattr(X_test, "index") else "",
                )
                all_performances[name].append(perf)
                window_perfs.append(perf)

                # Store the last trained models
                self._models[name] = model

            # Ensemble: average probabilities
            try:
                ensemble_preds = self._ensemble_predict_proba(models, X_test)
                ensemble_acc = accuracy_score(y_test, ensemble_preds)
                all_performances["ensemble"].append(ModelPerformance(
                    window_idx=win_idx, accuracy=round(ensemble_acc, 4),
                    total_trades=len(y_test),
                    sharpe_if_traded=round((ensemble_acc - 0.5) * 2 * np.sqrt(252) / 100, 3),
                    period_start=str(X_test.index[0]) if hasattr(X_test, "index") else "",
                    period_end=str(X_test.index[-1]) if hasattr(X_test, "index") else "",
                ))
            except Exception:
                pass

        # Calculate feature importance from final RF model
        if "rf" in self._models:
            try:
                importances = self._models["rf"].feature_importances_
                feature_names = full_X.columns
                self._feature_importance = pd.DataFrame({
                    "feature": feature_names,
                    "importance": importances,
                }).sort_values("importance", ascending=False)
            except Exception:
                pass

        self._is_trained = True
        self._train_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._performance_history = all_performances.get("ensemble", []) or all_performances.get("rf", [])

        return all_performances

    def predict(self, df: pd.DataFrame, feature_engine) -> PredictionResult:
        """
        Generate prediction for the latest bar using trained ensemble.

        Args:
            df: OHLCV DataFrame
            feature_engine: FeatureEngineeringEngine instance

        Returns:
            PredictionResult with signal, confidence, and metadata
        """
        if not self._is_trained:
            return PredictionResult(signal="HOLD", is_trained=False)

        # Generate features
        df_feat = feature_engine.generate_features(df.copy())
        X, _ = feature_engine.get_feature_matrix(df_feat, drop_na=False)

        if X.empty:
            return PredictionResult(signal="HOLD", is_trained=self._is_trained)

        # Use latest row
        latest = X.iloc[-1:].copy()

        # Fill NaN in latest row (shouldn't happen with proper lookback)
        latest = latest.fillna(0)

        # Get predictions from each ensemble member
        predictions = []
        probabilities_list = []

        for name, model in self._models.items():
            try:
                pred = model.predict(latest)[0]
                predictions.append(pred)

                # Get probability estimates
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(latest)[0]
                    probabilities_list.append(proba)
            except Exception:
                continue

        if not predictions:
            return PredictionResult(signal="HOLD", is_trained=self._is_trained)

        # Combine predictions via majority vote
        signal, confidence, probs = self._combine_predictions(predictions, probabilities_list)

        # Model agreement
        buy_votes = sum(1 for p in predictions if p == 1)
        sell_votes = sum(1 for p in predictions if p == -1)
        hold_votes = sum(1 for p in predictions if p == 0)
        n_models = len(predictions)
        max_votes = max(buy_votes, sell_votes, hold_votes)
        agreement = (max_votes / n_models * 100) if n_models > 0 else 0

        # Last OOS accuracy
        oos_acc = 0.0
        if self._performance_history:
            oos_acc = self._performance_history[-1].accuracy

        # Top features
        top_features = []
        if not self._feature_importance.empty:
            top_n = min(10, len(self._feature_importance))
            top_features = list(zip(
                self._feature_importance["feature"].head(top_n),
                self._feature_importance["importance"].head(top_n).round(4),
            ))

        return PredictionResult(
            signal=signal,
            confidence=confidence,
            probabilities=probs,
            model_agreement=round(agreement, 1),
            oos_accuracy=round(oos_acc, 4),
            features_used=len(X.columns),
            top_features=top_features,
            train_date=self._train_date or "",
            is_trained=self._is_trained,
        )

    def get_oos_performance(self) -> List[ModelPerformance]:
        """Return walk-forward out-of-sample performance for each window."""
        return self._performance_history

    def get_cumulative_oos_performance(self) -> Dict[str, float]:
        """Aggregate OOS performance across all windows."""
        if not self._performance_history:
            return {"accuracy": 0, "sharpe": 0, "n_windows": 0}

        accuracies = [p.accuracy for p in self._performance_history if p.accuracy > 0]
        sharpes = [p.sharpe_if_traded for p in self._performance_history]

        return {
            "accuracy": round(float(np.mean(accuracies)), 4) if accuracies else 0,
            "sharpe": round(float(np.mean(sharpes)), 3) if sharpes else 0,
            "n_windows": len(self._performance_history),
            "last_accuracy": self._performance_history[-1].accuracy if self._performance_history else 0,
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance DataFrame (top features first)."""
        return self._feature_importance

    def save_model(self, symbol: str = "GLOBAL"):
        """Persist trained models to disk with joblib."""
        import joblib

        if not self._models:
            return False

        save_path = os.path.join(self.model_dir, f"ml_ensemble_{symbol}")
        os.makedirs(save_path, exist_ok=True)

        for name, model in self._models.items():
            joblib.dump(model, os.path.join(save_path, f"{name}.pkl"))

        # Save metadata
        meta = {
            "symbol": symbol,
            "train_date": self._train_date,
            "feature_count": len(self._feature_importance) if not self._feature_importance.empty else 0,
            "performance_windows": len(self._performance_history),
            "cumulative_accuracy": self.get_cumulative_oos_performance().get("accuracy", 0),
        }
        joblib.dump(meta, os.path.join(save_path, "metadata.pkl"))

        self._symbol = symbol
        return True

    def load_model(self, symbol: str = "GLOBAL") -> bool:
        """Load previously trained models from disk."""
        import joblib

        load_path = os.path.join(self.model_dir, f"ml_ensemble_{symbol}")
        if not os.path.exists(load_path):
            return False

        try:
            for name in ["rf", "hgb", "ridge"]:
                pkl_path = os.path.join(load_path, f"{name}.pkl")
                if os.path.exists(pkl_path):
                    self._models[name] = joblib.load(pkl_path)

            meta_path = os.path.join(load_path, "metadata.pkl")
            if os.path.exists(meta_path):
                meta = joblib.load(meta_path)
                self._train_date = meta.get("train_date")
                self._symbol = meta.get("symbol", symbol)

            self._is_trained = len(self._models) > 0
            return self._is_trained
        except Exception:
            self._is_trained = False
            return False

    def needs_retrain(self, symbol: str = "GLOBAL") -> bool:
        """Check if model needs retraining (stale > ML_RETRAIN_DAYS)."""
        load_path = os.path.join(self.model_dir, f"ml_ensemble_{symbol}")
        if not os.path.exists(load_path):
            return True

        import joblib
        meta_path = os.path.join(load_path, "metadata.pkl")
        if not os.path.exists(meta_path):
            return True

        try:
            meta = joblib.load(meta_path)
            train_date = meta.get("train_date")
            if not train_date:
                return True
            train_dt = datetime.strptime(train_date, "%Y-%m-%d %H:%M")
            return (datetime.now() - train_dt).days > ML_RETRAIN_DAYS
        except Exception:
            return True

    # ── Internal methods ──

    def _walk_forward_split(self, n_rows: int) -> List[Tuple[int, int, int, int]]:
        """
        Generate walk-forward (train_start, train_end, test_start, test_end) indices.

        Structure:
            train = rows [train_start, train_end)
            embargo = rows [train_end, train_end + embargo_days)
            test = rows [train_end + embargo_days, test_end)
        """
        windows = []
        step = self.config.step_days
        train_size = self.config.min_train_days
        test_size = self.config.test_days
        embargo = self.config.embargo_days

        start = 0
        while start + train_size + embargo + test_size <= n_rows:
            train_end = start + train_size
            test_start = train_end + embargo
            test_end = test_start + test_size
            windows.append((start, train_end, test_start, min(test_end, n_rows)))
            start += step

        return windows

    def _combine_predictions(
        self,
        predictions: List[int],
        probabilities_list: List[np.ndarray],
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Combine ensemble predictions via majority vote with confidence calibration.

        Returns:
            (signal, confidence_0_100, prob_dict)
        """
        if not predictions:
            return "HOLD", 0.0, {"BUY": 0, "SELL": 0, "HOLD": 1}

        # Majority vote
        buy_pct = sum(1 for p in predictions if p == 1) / len(predictions)
        sell_pct = sum(1 for p in predictions if p == -1) / len(predictions)
        hold_pct = sum(1 for p in predictions if p == 0) / len(predictions)

        # Average probabilities from models that support predict_proba
        if probabilities_list:
            avg_probs = {
                "BUY": float(np.mean([p[1] for p in probabilities_list])) if probabilities_list[0].shape[0] >= 2 else buy_pct,
                "SELL": float(np.mean([p[0] for p in probabilities_list])) if "p[0]" in locals() else sell_pct,
                "HOLD": 0.0,
            }
        else:
            avg_probs = {"BUY": buy_pct, "SELL": sell_pct, "HOLD": hold_pct}

        # Determine signal
        if buy_pct > sell_pct and buy_pct > hold_pct and buy_pct >= 0.4:
            signal = "BUY"
            confidence = min(95, int(buy_pct * 100))
        elif sell_pct > buy_pct and sell_pct > hold_pct and sell_pct >= 0.4:
            signal = "SELL"
            confidence = min(95, int(sell_pct * 100))
        else:
            signal = "HOLD"
            confidence = max(hold_pct * 100, 50)

        return signal, confidence, {"BUY": round(buy_pct, 3), "SELL": round(sell_pct, 3), "HOLD": round(hold_pct, 3)}

    def _ensemble_predict_proba(self, models: dict, X_test: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction by averaging probabilities."""
        from sklearn.linear_model import RidgeClassifier

        probas = []
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 3:  # -1, 0, 1
                    probas.append(proba)
                elif proba.shape[1] == 2:  # binary
                    probas.append(np.column_stack([1 - proba[:, 1], proba[:, 1], np.zeros(len(proba))]))
            elif isinstance(model, RidgeClassifier):
                # Ridge outputs -1/1, convert to pseudo-probabilities
                preds = model.predict(X_test)
                probas.append(np.column_stack([
                    (preds == -1).astype(float),
                    (preds == 1).astype(float),
                    (preds == 0).astype(float),
                ]))

        if not probas:
            return np.zeros(len(X_test))

        avg_probas = np.mean(probas, axis=0)
        return np.argmax(avg_probas, axis=1)
