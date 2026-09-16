"""
LifeSync ML Engine — Clinical Deterioration Prediction Model
=============================================================
A GradientBoostingClassifier trained on synthetic clinical vitals data
to predict patient deterioration status (STABLE / WARNING / CRITICAL).

Architecture:
  - Features: heart_rate, spo2, systolic_bp (derived), respiratory_rate (derived)
  - Target: triage_label (0=STABLE, 1=WARNING, 2=CRITICAL)
  - Model: sklearn.ensemble.GradientBoostingClassifier
  - Training: 5000 synthetic samples generated from clinical distributions
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import time

warnings.filterwarnings("ignore")


class DeteriorationModel:
    """
    Real-time clinical deterioration prediction engine.
    Trained on synthetic vitals data modeled after clinical distributions.
    """

    LABELS = {0: "STABLE", 1: "WARNING", 2: "CRITICAL"}

    def __init__(self):
        self.model = None
        self.accuracy = 0.0
        self.feature_importances = {}
        self.training_time_ms = 0
        self.is_ready = False
        self._train()

    def _generate_synthetic_data(self, n_samples=5000):
        """
        Generate clinically-representative synthetic vitals data.
        Based on Modified Early Warning Score (MEWS) clinical thresholds.
        """
        np.random.seed(42)
        data = []
        labels = []

        # --- STABLE patients (60% of dataset) ---
        n_stable = int(n_samples * 0.6)
        for _ in range(n_stable):
            hr = np.random.normal(75, 8)       # Normal resting HR
            spo2 = np.random.normal(97, 1.2)   # Normal oxygen
            sbp = np.random.normal(120, 10)     # Normal systolic BP
            rr = np.random.normal(16, 2)        # Normal respiratory rate
            data.append([hr, spo2, sbp, rr])
            labels.append(0)

        # --- WARNING patients (25% of dataset) ---
        n_warning = int(n_samples * 0.25)
        for _ in range(n_warning):
            hr = np.random.normal(105, 12)      # Elevated HR
            spo2 = np.random.normal(93, 2)      # Slightly low O2
            sbp = np.random.normal(100, 15)     # Lower BP
            rr = np.random.normal(22, 3)         # Elevated RR
            data.append([hr, spo2, sbp, rr])
            labels.append(1)

        # --- CRITICAL patients (15% of dataset) ---
        n_critical = n_samples - n_stable - n_warning
        for _ in range(n_critical):
            hr = np.random.choice([
                np.random.normal(135, 15),     # Tachycardia
                np.random.normal(42, 5)         # Bradycardia
            ])
            spo2 = np.random.normal(85, 4)     # Dangerous hypoxemia
            sbp = np.random.normal(80, 20)     # Hypotension
            rr = np.random.normal(30, 5)        # Tachypnea
            data.append([hr, spo2, sbp, rr])
            labels.append(2)

        return np.array(data), np.array(labels)

    def _train(self):
        """Train the GradientBoosting model on synthetic clinical data."""
        start = time.time()

        X, y = self._generate_synthetic_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)

        feature_names = ["heart_rate", "spo2", "systolic_bp", "respiratory_rate"]
        self.feature_importances = dict(
            zip(feature_names, [round(float(x), 4) for x in self.model.feature_importances_])
        )

        self.training_time_ms = round((time.time() - start) * 1000, 1)
        self.is_ready = True

        print(f"[ML ENGINE] Model trained in {self.training_time_ms}ms | Accuracy: {self.accuracy:.2%}")
        print(f"[ML ENGINE] Feature importances: {self.feature_importances}")

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
        """
        if not self.is_ready:
            return self._fallback(hr, spo2)

        try:
            features = np.array([self._derive_features(hr, spo2)])
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

            status = self.LABELS[prediction]
            confidence = round(float(max(probabilities)) * 100, 1)

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
                    "critical": round(float(probabilities[2]) * 100, 1)
                }
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
            "probabilities": None
        }

    def get_model_info(self):
        """Return model metadata for the /ml-status endpoint."""
        return {
            "model_type": "GradientBoostingClassifier",
            "accuracy": round(self.accuracy * 100, 2),
            "training_samples": 5000,
            "features": ["heart_rate", "spo2", "systolic_bp (derived)", "respiratory_rate (derived)"],
            "feature_importances": self.feature_importances,
            "training_time_ms": self.training_time_ms,
            "is_ready": self.is_ready,
            "classes": ["STABLE", "WARNING", "CRITICAL"]
        }
