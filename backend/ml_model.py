# ml_model.py
# Uses GradientBoostingClassifier to predict patient status
# 0 = STABLE, 1 = WARNING, 2 = CRITICAL

import hashlib
import json
import numpy as np
import os
import time
import warnings
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")

# --- Model Persistence ---
MODEL_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = MODEL_DIR / "lifesync_model.joblib"


class DeteriorationModel:
    # simple ML model for the hackathon project

    LABELS = {0: "STABLE", 1: "WARNING", 2: "CRITICAL"}
    HYPERPARAMS = {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.1,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": 42,
    }

    def __init__(self):
        self.model = None
        self.accuracy = 0.0
        self.feature_importances = {}
        self.training_time_ms = 0
        self.is_ready = False
        self.model_version = self._compute_version_hash()

        # Evaluation metrics
        self.confusion_matrix = None
        self.classification_report_dict = None
        self.cv_scores = None
        self.cv_mean = 0.0
        self.cv_std = 0.0
        self.per_class_f1 = {}

        # Runtime analytics
        self.prediction_count = 0
        self.prediction_class_counts = {"STABLE": 0, "WARNING": 0, "CRITICAL": 0}

        # Try loading cached model, otherwise train from scratch
        if not self._load_cached_model():
            self._train()

    def _compute_version_hash(self):
        """Generate a reproducible model version hash from hyperparameters."""
        hp_string = json.dumps(self.HYPERPARAMS, sort_keys=True)
        return hashlib.sha256(hp_string.encode()).hexdigest()[:12]

    def _load_cached_model(self):
        """Attempt to load a previously trained model from disk."""
        try:
            import joblib

            if MODEL_PATH.exists():
                cached = joblib.load(MODEL_PATH)
                if cached.get("version") == self.model_version:
                    self.model = cached["model"]
                    self.accuracy = cached["accuracy"]
                    self.feature_importances = cached["feature_importances"]
                    self.training_time_ms = cached["training_time_ms"]
                    self.confusion_matrix = cached.get("confusion_matrix")
                    self.classification_report_dict = cached.get("classification_report")
                    self.cv_scores = cached.get("cv_scores")
                    self.cv_mean = cached.get("cv_mean", 0.0)
                    self.cv_std = cached.get("cv_std", 0.0)
                    self.per_class_f1 = cached.get("per_class_f1", {})
                    self.is_ready = True
                    print(f"Loaded cached model. Accuracy: {self.accuracy:.2%}")
                    return True
                else:
                    print(f"Model hash mismatch, training again...")
                    return False
        except Exception as e:
            print(f"Couldn't load model: {e}")
        return False

    def _save_model(self):
        """Persist trained model and metrics to disk."""
        try:
            import joblib

            MODEL_DIR.mkdir(exist_ok=True)
            joblib.dump(
                {
                    "version": self.model_version,
                    "model": self.model,
                    "accuracy": self.accuracy,
                    "feature_importances": self.feature_importances,
                    "training_time_ms": self.training_time_ms,
                    "confusion_matrix": self.confusion_matrix,
                    "classification_report": self.classification_report_dict,
                    "cv_scores": self.cv_scores,
                    "cv_mean": self.cv_mean,
                    "cv_std": self.cv_std,
                    "per_class_f1": self.per_class_f1,
                },
                MODEL_PATH,
            )
            print(f"[ML ENGINE] [SAVED] Model saved to {MODEL_PATH}")
        except Exception as e:
            print(f"[ML ENGINE] [WARN] Could not save model: {e}")

    def _generate_synthetic_data(self, n_samples=5000):
        """
        Generate clinically-representative synthetic vitals data.
        Based on Modified Early Warning Score (MEWS) clinical thresholds.

        Distributions are designed with slight overlap at boundaries to
        train a model that handles borderline cases gracefully.
        """
        np.random.seed(42)
        data = []
        labels = []

        # --- STABLE patients (60% of dataset) ---
        n_stable = int(n_samples * 0.6)
        for _ in range(n_stable):
            hr = np.random.normal(75, 8)  # Normal resting HR
            spo2 = np.random.normal(97, 1.2)  # Normal oxygen
            sbp = np.random.normal(120, 10)  # Normal systolic BP
            rr = np.random.normal(16, 2)  # Normal respiratory rate
            data.append([hr, spo2, sbp, rr])
            labels.append(0)

        # --- WARNING patients (25% of dataset) ---
        # Includes boundary samples near STABLE and CRITICAL thresholds
        n_warning = int(n_samples * 0.25)
        for i in range(n_warning):
            if i < n_warning * 0.15:
                # Low-warning (near stable boundary)
                hr = np.random.normal(92, 6)
                spo2 = np.random.normal(95, 1.5)
                sbp = np.random.normal(108, 10)
                rr = np.random.normal(19, 2)
            elif i > n_warning * 0.85:
                # High-warning (near critical boundary)
                hr = np.random.normal(118, 8)
                spo2 = np.random.normal(90, 1.5)
                sbp = np.random.normal(92, 12)
                rr = np.random.normal(25, 3)
            else:
                # Core warning distribution
                hr = np.random.normal(105, 12)
                spo2 = np.random.normal(93, 2)
                sbp = np.random.normal(100, 15)
                rr = np.random.normal(22, 3)
            data.append([hr, spo2, sbp, rr])
            labels.append(1)

        # --- CRITICAL patients (15% of dataset) ---
        n_critical = n_samples - n_stable - n_warning
        for _ in range(n_critical):
            hr = np.random.choice(
                [
                    np.random.normal(135, 15),  # Tachycardia
                    np.random.normal(42, 5),  # Bradycardia
                ]
            )
            spo2 = np.random.normal(85, 4)  # Dangerous hypoxemia
            sbp = np.random.normal(80, 20)  # Hypotension
            rr = np.random.normal(30, 5)  # Tachypnea
            data.append([hr, spo2, sbp, rr])
            labels.append(2)

        return np.array(data), np.array(labels)

    def _train(self):
        """
        Train the GradientBoosting model on synthetic clinical data.
        Includes full evaluation pipeline: cross-validation, confusion matrix,
        classification report, and per-class F1 scores.
        """
        start = time.time()

        X, y = self._generate_synthetic_data()

        # --- Cross-Validation (5-fold stratified) ---
        print("[ML ENGINE] Running 5-fold stratified cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        temp_model = GradientBoostingClassifier(**self.HYPERPARAMS)
        self.cv_scores = cross_val_score(temp_model, X, y, cv=cv, scoring="accuracy")
        self.cv_mean = float(np.mean(self.cv_scores))
        self.cv_std = float(np.std(self.cv_scores))
        print(f"[ML ENGINE] Cross-validation: {self.cv_mean:.2%} ± {self.cv_std:.2%}")
        print(f"[ML ENGINE] CV fold scores: {[round(s, 4) for s in self.cv_scores]}")

        # --- Train/Test Split & Final Model ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = GradientBoostingClassifier(**self.HYPERPARAMS)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)

        # --- Confusion Matrix ---
        self.confusion_matrix = confusion_matrix(y_test, y_pred).tolist()
        print("\n[ML ENGINE] === CONFUSION MATRIX ===")
        print(f"              Predicted:  STABLE  WARNING  CRITICAL")
        for i, row in enumerate(self.confusion_matrix):
            print(f"  Actual {self.LABELS[i]:>8}: {row}")

        # --- Classification Report ---
        report = classification_report(
            y_test, y_pred,
            target_names=["STABLE", "WARNING", "CRITICAL"],
            output_dict=True,
        )
        self.classification_report_dict = report
        print(f"\n[ML ENGINE] === CLASSIFICATION REPORT ===")
        print(
            classification_report(
                y_test, y_pred, target_names=["STABLE", "WARNING", "CRITICAL"]
            )
        )

        # --- Per-Class F1 Scores ---
        self.per_class_f1 = {
            "STABLE": round(report["STABLE"]["f1-score"], 4),
            "WARNING": round(report["WARNING"]["f1-score"], 4),
            "CRITICAL": round(report["CRITICAL"]["f1-score"], 4),
        }

        # --- Feature Importances ---
        feature_names = ["heart_rate", "spo2", "systolic_bp", "respiratory_rate"]
        self.feature_importances = dict(
            zip(
                feature_names,
                [round(float(x), 4) for x in self.model.feature_importances_],
            )
        )

        self.training_time_ms = round((time.time() - start) * 1000, 1)
        self.is_ready = True

        print(f"\nModel trained in {self.training_time_ms}ms | Accuracy: {self.accuracy:.2%}")
        print(f"Feature importances: {self.feature_importances}")

        # Save to disk for next restart
        self._save_model()

    def _derive_features(self, hr: int, spo2: int):
        """
        Derive systolic BP and respiratory rate from available vitals.
        Uses clinically-informed heuristic correlations since we don't
        have real BP/RR sensors in this prototype.
        """
        # Approximate systolic BP from HR (inverse correlation in shock)
        sbp = max(60, min(180, 160 - (hr - 75) * 0.6 + np.random.normal(0, 3)))

        # Approximate respiratory rate from SpO2 (compensatory tachypnea)
        rr = max(8, min(45, 40 - (spo2 - 80) * 0.5 + np.random.normal(0, 1.5)))

        return [hr, spo2, sbp, rr]

    def predict(self, hr: int, spo2: int):
        """
        Predict deterioration status from heart rate and SpO2.
        Returns dict with status, risk_score, confidence, triage_horizon.
        Tracks prediction analytics.
        """
        if not self.is_ready:
            return self._fallback(hr, spo2)

        try:
            # Validate inputs
            hr = max(20, min(250, int(hr)))
            spo2 = max(50, min(100, int(spo2)))

            features = np.array([self._derive_features(hr, spo2)])
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

            status = self.LABELS[prediction]
            confidence = round(float(max(probabilities)) * 100, 1)

            # Track prediction analytics
            self.prediction_count += 1
            self.prediction_class_counts[status] += 1

            # Risk score: weighted combination of WARNING and CRITICAL probabilities
            risk_score = min(100, int(probabilities[1] * 40 + probabilities[2] * 100))

            # Triage horizon based on probabilities
            if status == "CRITICAL":
                triage_horizon = "IMMEDIATE (0 mins)"
            elif status == "WARNING":
                minutes = max(1, int((1 - probabilities[2]) * 20))
                triage_horizon = f"CRITICAL in ~{minutes} mins"
            else:
                triage_horizon = "STABLE (> 45 mins)"

            return {
                "status": status,
                "risk_score": risk_score,
                "confidence": confidence,
                "triage_horizon": triage_horizon,
                "probabilities": {
                    "stable": round(float(probabilities[0]) * 100, 1),
                    "warning": round(float(probabilities[1]) * 100, 1),
                    "critical": round(float(probabilities[2]) * 100, 1),
                },
            }
        except Exception as e:
            print(f"[ML ENGINE] Prediction error: {e}, using fallback")
            return self._fallback(hr, spo2)

    def _fallback(self, hr: int, spo2: int):
        """Rule-based fallback if ML model is unavailable."""
        hr_risk = (abs(hr - 80) / 100) * 50 if hr > 100 or hr < 60 else 0
        spo2_risk = ((95 - spo2) / 25) * 50 if spo2 < 95 else 0
        total_risk = min(int(hr_risk + spo2_risk), 100)

        if spo2 < 90 or hr > 130 or hr < 50:
            total_risk = max(total_risk, 85)

        status = "CRITICAL" if total_risk > 80 else "WARNING" if total_risk > 50 else "STABLE"

        if status == "CRITICAL":
            triage_horizon = "IMMEDIATE (0 mins)"
        elif status == "WARNING":
            minutes = max(1, int(((80 - total_risk) / 30) ** 2 * 15))
            triage_horizon = f"CRITICAL in ~{minutes} mins"
        else:
            triage_horizon = "STABLE (> 45 mins)"

        return {
            "status": status,
            "risk_score": total_risk,
            "confidence": 0.0,
            "triage_horizon": triage_horizon,
            "probabilities": None,
        }

    def get_model_info(self):
        """Return model metadata for the /ml-status endpoint."""
        return {
            "model_type": "GradientBoostingClassifier",
            "model_version": self.model_version,
            "accuracy": round(self.accuracy * 100, 2),
            "training_samples": 5000,
            "features": [
                "heart_rate",
                "spo2",
                "systolic_bp (derived)",
                "respiratory_rate (derived)",
            ],
            "feature_importances": self.feature_importances,
            "per_class_f1": self.per_class_f1,
            "cross_validation": {
                "folds": 5,
                "mean_accuracy": round(self.cv_mean * 100, 2),
                "std": round(self.cv_std * 100, 2),
                "fold_scores": [round(float(s) * 100, 2) for s in (self.cv_scores if self.cv_scores is not None else [])],
            },
            "training_time_ms": self.training_time_ms,
            "is_ready": self.is_ready,
            "classes": ["STABLE", "WARNING", "CRITICAL"],
            "predictions_made": self.prediction_count,
            "prediction_distribution": self.prediction_class_counts,
        }

    def get_evaluation_report(self):
        """
        Return comprehensive model evaluation for the /ml-evaluation endpoint.
        Includes confusion matrix, classification report, cross-validation scores.
        """
        return {
            "model_version": self.model_version,
            "test_accuracy": round(self.accuracy * 100, 2),
            "cross_validation": {
                "strategy": "5-fold Stratified K-Fold",
                "mean_accuracy": round(self.cv_mean * 100, 2),
                "std_deviation": round(self.cv_std * 100, 2),
                "fold_scores": [round(float(s) * 100, 2) for s in (self.cv_scores if self.cv_scores is not None else [])],
            },
            "confusion_matrix": {
                "labels": ["STABLE", "WARNING", "CRITICAL"],
                "matrix": self.confusion_matrix,
            },
            "classification_report": self.classification_report_dict,
            "per_class_f1_scores": self.per_class_f1,
            "feature_importances": self.feature_importances,
            "hyperparameters": self.HYPERPARAMS,
            "training_data": {
                "total_samples": 5000,
                "class_distribution": {
                    "STABLE": "60% (3000 samples)",
                    "WARNING": "25% (1250 samples)",
                    "CRITICAL": "15% (750 samples)",
                },
                "split": "80% train / 20% test (stratified)",
            },
            "runtime_analytics": {
                "predictions_made": self.prediction_count,
                "prediction_distribution": self.prediction_class_counts,
            },
        }
