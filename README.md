# 🚑 LifeSync — Agentic AI Emergency Network

> **Team Commitment_Issues** | Morrow 1.0 Hackathon — Round 2 Submission

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Socket.IO](https://img.shields.io/badge/Socket.IO-4.7-010101?logo=socket.io)](https://socket.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Problem Statement

**Every year, ~1.35 million people die in road accidents globally (WHO, 2023). In India alone, over 1.7 lakh lives are lost annually — not always from the injury itself, but from the "Golden Hour" information blindspot:** ambulances don't know which ER is full, hospitals don't know what's incoming, and critical patient data is lost in transit.

LifeSync eliminates this gap with an **Agentic AI architecture** that autonomously perceives patient vitals, reasons about clinical deterioration using ML, acts by routing to the optimal hospital via multi-agent negotiation, and secures every transaction on a cryptographic ledger.

---

## 🧠 Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      LifeSync Agentic AI Pipeline                    │
│                                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────┐ │
│  │ PERCEIVE │───▶│    REASON    │───▶│    ACT     │───▶│  SECURE  │ │
│  │          │    │              │    │            │    │          │ │
│  │ Vitals   │    │ ML Model    │    │ Multi-Agent│    │ SHA-256  │ │
│  │ GPS      │    │ GradientBoost│   │ Handshake  │    │ Ledger   │ │
│  │ ABHA ID  │    │ Triage Index │    │ Routing    │    │ Audit    │ │
│  └──────────┘    └──────────────┘    └────────────┘    └──────────┘ │
│       ▲                                    │                         │
│       │              WebSocket             │                         │
│       └────────────────────────────────────┘                         │
└──────────────────────────────────────────────────────────────────────┘
```

### Core Design Pattern: **Perceive → Reason → Act → Secure**

| Phase | Description | Technology |
|-------|-------------|------------|
| **1. Perceive** | Real-time vitals ingestion (HR, SpO2) + GPS + ABHA patient identity | Socket.IO WebSockets |
| **2. Reason** | ML-powered deterioration prediction with confidence scoring | scikit-learn GradientBoostingClassifier |
| **3. Act** | Multi-agent hospital handshake protocol with capacity-aware routing | Custom agent negotiation engine |
| **4. Secure** | Every transaction hashed with SHA-256 and appended to an immutable audit ledger | Python hashlib + JSONL |

---

## 🤖 Machine Learning Model

### Model Specification

| Property | Value |
|----------|-------|
| **Algorithm** | `GradientBoostingClassifier` (scikit-learn) |
| **Type** | Supervised multi-class classification |
| **Accuracy** | **98.9%** on held-out test set (80/20 split, stratified) |
| **Training Samples** | 5,000 synthetic clinical records |
| **Inference Latency** | < 5ms per prediction |
| **Training Time** | ~2.5 seconds at server startup |

### Features

| Feature | Source | Clinical Basis |
|---------|--------|----------------|
| `heart_rate` | Direct sensor input | Primary hemodynamic indicator |
| `spo2` | Direct sensor input | Peripheral oxygen saturation |
| `systolic_bp` | Derived from HR | Inverse correlation in shock states (Poiseuille's law) |
| `respiratory_rate` | Derived from SpO2 | Compensatory tachypnea in hypoxemia |

### Target Classes

| Class | Label | Clinical Definition |
|-------|-------|---------------------|
| **STABLE** | 0 | Vitals within normal range, no immediate intervention needed |
| **WARNING** | 1 | Vitals trending abnormal, close monitoring required |
| **CRITICAL** | 2 | Life-threatening deterioration, immediate triage needed |

### Training Dataset

The model is trained on **synthetically generated clinical data** modeled after the **Modified Early Warning Score (MEWS)** clinical framework, which is the gold standard for in-hospital deterioration detection.

**Data Generation Strategy:**

```
STABLE (60%):   HR ~ N(75, 8)    SpO2 ~ N(97, 1.2)   SBP ~ N(120, 10)   RR ~ N(16, 2)
WARNING (25%):  HR ~ N(105, 12)  SpO2 ~ N(93, 2)     SBP ~ N(100, 15)   RR ~ N(22, 3)
CRITICAL (15%): HR ~ N(135, 15)  SpO2 ~ N(85, 4)     SBP ~ N(80, 20)    RR ~ N(30, 5)
                or HR ~ N(42, 5)  (bradycardia variant)
```

> **Why synthetic data?** In a production system, this model would be trained on real ICU datasets (e.g., MIMIC-III). For this prototype, we use clinically-representative synthetic distributions to demonstrate the ML pipeline architecture without requiring access to restricted medical datasets. The distributions are derived from published MEWS threshold literature.

### Hyperparameters

```python
GradientBoostingClassifier(
    n_estimators=150,      # Number of boosting stages
    max_depth=5,           # Maximum tree depth
    learning_rate=0.1,     # Shrinkage parameter
    min_samples_split=10,  # Minimum samples to split a node
    min_samples_leaf=5,    # Minimum samples per leaf
    random_state=42        # Reproducibility
)
```

### Feature Importance

| Feature | Importance | Interpretation |
|---------|-----------|----------------|
| `spo2` | **59.9%** | Most critical — oxygen saturation is the primary predictor of clinical deterioration |
| `heart_rate` | **29.5%** | Second most important — tachycardia/bradycardia signals cardiac distress |
| `respiratory_rate` | **9.1%** | Compensatory breathing rate changes |
| `systolic_bp` | **1.5%** | Derived feature, lower independent signal |

### Model API

The ML model exposes its predictions via the `/predict` POST endpoint and its metadata via `/ml-status` GET endpoint.

**Example — Predict deterioration:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"heartRate": 140, "spo2": 84, "lat": 18.52, "lng": 73.85}'
```

**Response:**
```json
{
  "risk_score": 99,
  "status": "CRITICAL",
  "secure_hash": "a3f2...8c1d",
  "assigned_hospital": {"name": "Advanced Trauma Institute", "type": "TRAUMA"},
  "triage_horizon": "IMMEDIATE (0 mins)",
  "ml_confidence": 100.0,
  "probabilities": {"stable": 0.0, "warning": 0.0, "critical": 100.0}
}
```

**Example — Model metadata:**
```bash
curl http://localhost:8000/ml-status
```

**Response:**
```json
{
  "model_type": "GradientBoostingClassifier",
  "accuracy": 98.9,
  "training_samples": 5000,
  "features": ["heart_rate", "spo2", "systolic_bp (derived)", "respiratory_rate (derived)"],
  "feature_importances": {"heart_rate": 0.2949, "spo2": 0.5992, "systolic_bp": 0.0149, "respiratory_rate": 0.091},
  "training_time_ms": 2489.4,
  "is_ready": true,
  "classes": ["STABLE", "WARNING", "CRITICAL"]
}
```

---

## 🏗️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | Python 3.12 + FastAPI | REST API, WebSocket handling, ML inference |
| **ML Engine** | scikit-learn + NumPy | GradientBoosting clinical deterioration model |
| **Real-time** | Socket.IO (python-socketio) | Bi-directional vitals streaming & chat |
| **Frontend** | HTML5 + CSS3 + Vanilla JS | Premium glassmorphism dark-mode UI |
| **Maps** | Leaflet.js + OpenStreetMap | Free, API-key-free GPS tracking & routing |
| **Charts** | Chart.js | Real-time ECG & SpO2 waveform rendering |
| **Security** | SHA-256 (hashlib) | Cryptographic audit ledger |
| **Deployment** | Render | Cloud hosting with health checks |

---

## ✨ Key Features

### Ambulance Edge (`/ambulance`)
- **Live ECG/SpO2 Waveforms** — Real-time Chart.js rendered clinical waveforms
- **GPS Map Tracking** — Leaflet map with animated route polyline to assigned hospital (OpenStreetMap tiles, zero API keys)
- **ABHA Identity Lookup** — Instant patient registry query (name, age, blood group, allergies)
- **Dead-Zone Burst Cache** — Offline-resilient queue that bursts cached vitals when connectivity resumes
- **ML Confidence Badge** — Live model confidence percentage display
- **Agent-Assisted Chat** — Encrypted communication channel with quick-action macros

### Hospital Command (`/dashboard`)
- **Agentic Command Center** — 4-phase pipeline visualization (Perceive → Reason → Act → Secure)
- **ML Deterioration Gauge** — Risk score progress bar with probability breakdown (Stable/Warning/Critical)
- **Live Ambulance Tracking** — Real-time GPS marker on CartoDB Dark Matter map
- **Multi-Hospital Markers** — All nearby hospitals shown with capacity status and bed counts
- **Green Wave Traffic Preemption** — Toggle button simulating traffic signal override for ambulance corridor
- **SHA-256 Vault** — Live cryptographic hash of every vitals transaction
- **AI System Alerts** — Automatic chat alerts when patient deteriorates to CRITICAL

### Multi-Agent Handshake Protocol
When the AI determines a patient needs a specific hospital type:
1. It queries the nearest matching hospital's "agent"
2. If that hospital is at capacity → **DENIED** → renegotiates with next-nearest
3. If accepted → **GRANTED** → route is locked and displayed on both maps
4. Blood Bank Agent pre-fetches required blood type at the destination hospital

---

## 📁 Project Structure

```
morrow/
├── run.py                      # Application entry point (port conflict resolution)
├── requirements.txt            # Python dependencies
├── render.yaml                 # Render cloud deployment config
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
├── backend/
│   ├── main.py                 # FastAPI + Socket.IO server + agentic pipeline
│   ├── ml_model.py             # GradientBoosting deterioration prediction model
│   └── requirements.txt        # Backend-specific dependencies
│
├── frontend/
│   ├── index.html              # Landing page (glassmorphism dark-mode)
│   ├── ambulance.html          # Ambulance telemetry + map + chat
│   └── dashboard.html          # Hospital command center + map + ML gauge
│
└── data/
    └── secure_ledger.jsonl     # Immutable SHA-256 audit trail (auto-generated)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ installed
- pip (Python package manager)
- A modern web browser (Chrome/Edge recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/M23adi/LifeSync-Morrow-Hackathon.git
cd morrow

# Install dependencies
pip install -r requirements.txt

# Start the server
python run.py
```

### Access the Application

| Portal | URL |
|--------|-----|
| 🌐 Landing Page | [http://localhost:8000/](http://localhost:8000/) |
| 🚑 Ambulance Edge | [http://localhost:8000/ambulance](http://localhost:8000/ambulance) |
| 🏥 Hospital Command | [http://localhost:8000/dashboard](http://localhost:8000/dashboard) |
| 📖 Swagger API Docs | [http://localhost:8000/docs](http://localhost:8000/docs) |
| 🧠 ML Model Status | [http://localhost:8000/ml-status](http://localhost:8000/ml-status) |
| 💚 Health Check | [http://localhost:8000/health](http://localhost:8000/health) |

### Demo Workflow
1. Open `/ambulance` and `/dashboard` in **two separate browser tabs**
2. On the ambulance page, enter ABHA ID: `14-1234-5678-9012` and click **Lookup**
3. Watch the vitals stream in real-time on both screens
4. Send messages between ambulance and hospital using the chat
5. Observe the AI status badge change as vitals fluctuate (STABLE → WARNING → CRITICAL)
6. On the dashboard, click **Activate Green Wave** to simulate traffic preemption

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Landing page |
| `GET` | `/ambulance` | Ambulance telemetry portal |
| `GET` | `/dashboard` | Hospital command dashboard |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `GET` | `/health` | Health check (for deployment monitoring) |
| `GET` | `/ml-status` | ML model metadata and performance metrics |
| `GET` | `/patient/{abha_id}` | ABHA patient registry lookup |
| `POST` | `/predict` | ML-powered triage prediction |

### Socket.IO Events

| Event | Direction | Payload |
|-------|-----------|---------|
| `vitals` | Client → Server | `{heartRate, spo2, lat, lng, timestamp}` |
| `update` | Server → All Clients | `{status, risk_score, ml_confidence, assigned_hospital, secure_hash, ...}` |
| `fetch_patient` | Client → Server | `abha_id` (string) |
| `patient_data_received` | Server → All Clients | `{name, age, blood_group, allergies, history}` |
| `send_message` | Client → Server | `{sender, message, time}` |
| `receive_message` | Server → All Clients | `{sender, message, time}` |

---

## 🔐 Security & Compliance

- **SHA-256 Audit Ledger**: Every vitals transmission is hashed and appended to an immutable JSONL file, creating a verifiable chain of custody for legal and medical audit purposes.
- **ABHA Integration**: Simulates India's Ayushman Bharat Health Account (ABHA) identity system for patient verification.
- **CORS Configured**: Cross-origin resource sharing properly configured for secure WebSocket connections.

---

## 🛣️ Phase 2.0 Roadmap

- [ ] Integration with real ABHA/ABDM APIs (National Health Authority)
- [ ] WebRTC two-way video for remote clinical guidance
- [ ] IoT sensor integration (real pulse oximeters, ECG patches)
- [ ] Training on MIMIC-III / eICU clinical datasets
- [ ] Multi-ambulance fleet management
- [ ] Green Wave integration with traffic management systems
- [ ] Mobile app (React Native) for paramedics

---

## 👥 Team Commitment_Issues

Built with conviction for **Morrow 1.0 Hackathon**.

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
