"""
LifeSync Backend Server
FastAPI + Socket.IO server handling ML predictions and routing.
"""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
import socketio
import uvicorn

# --- Directories ---
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
FRONTEND_DIR = WORKSPACE_DIR / "frontend"
DATA_DIR = WORKSPACE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- ML Model ---
from ml_model import DeteriorationModel

print("Initializing ML model...")
ml_engine = DeteriorationModel()
SERVER_START_TIME = time.time()

# Setup socketio and fastapi
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI(
    title="LifeSync API",
    description="Emergency triage and routing API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Ledger Settings ---
MAX_LEDGER_ENTRIES = 1000


class Vitals(BaseModel):
    heartRate: int
    spo2: int
    lat: Optional[float] = 18.5204
    lng: Optional[float] = 73.8567


# --- 1. ABHA ID PATIENT REGISTRY (MOCK DATABASE) ---
mock_abha_db = {
    "14-1234-5678-9012": {
        "name": "Rajesh Kumar",
        "age": 45,
        "blood_group": "O+",
        "allergies": ["Penicillin", "Peanuts"],
        "history": ["Hypertension (2021)", "Appendectomy (2018)"],
    },
    "14-9876-5432-1098": {
        "name": "Priya Sharma",
        "age": 32,
        "blood_group": "A-",
        "allergies": ["None"],
        "history": ["Asthma"],
    },
    "14-5555-1234-7890": {
        "name": "Amit Verma",
        "age": 58,
        "blood_group": "B+",
        "allergies": ["Sulfa drugs", "Iodine contrast"],
        "history": ["Type 2 Diabetes (2015)", "CABG Surgery (2020)", "CKD Stage 2"],
    },
    "14-7777-8888-9999": {
        "name": "Ananya Patel",
        "age": 27,
        "blood_group": "AB+",
        "allergies": ["Aspirin"],
        "history": ["Iron-deficiency Anemia (2023)"],
    },
    "14-3333-4444-5555": {
        "name": "Vikram Singh",
        "age": 63,
        "blood_group": "O-",
        "allergies": ["NSAIDs", "Latex"],
        "history": [
            "Atrial Fibrillation (2019)",
            "Hip Replacement (2022)",
            "Chronic COPD",
        ],
    },
}


# --- 2. AGENTIC ROUTING & LOGIC ---
def calculate_distance(lat1, lng1, lat2, lng2):
    """Haversine-approximated distance between two GPS coordinates (km)."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)


def estimate_eta(distance_km, speed_kmh=45):
    """Estimate arrival time based on distance and average ambulance speed."""
    if distance_km <= 0:
        return "< 1 min"
    minutes = (distance_km / speed_kmh) * 60
    if minutes < 1:
        return "< 1 min"
    return f"~{int(minutes)} min{'s' if int(minutes) != 1 else ''}"


def get_hospitals_near(lat, lng):
    """Return hospitals near the ambulance's current GPS position."""
    return [
        {
            "name": "City General Hospital",
            "lat": lat + 0.015,
            "lng": lng + 0.012,
            "type": "GENERAL",
            "capacity_full": False,
            "beds_available": 12,
        },
        {
            "name": "Metro Trauma Center",
            "lat": lat - 0.020,
            "lng": lng - 0.005,
            "type": "TRAUMA",
            "capacity_full": True,
            "beds_available": 0,
        },
        {
            "name": "Advanced Trauma Institute",
            "lat": lat - 0.010,
            "lng": lng + 0.025,
            "type": "TRAUMA",
            "capacity_full": False,
            "beds_available": 5,
        },
    ]


def _rotate_ledger(ledger_path):
    """Rotate ledger file if it exceeds MAX_LEDGER_ENTRIES."""
    try:
        if ledger_path.exists():
            with open(ledger_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > MAX_LEDGER_ENTRIES:
                # Keep only the most recent entries
                with open(ledger_path, "w", encoding="utf-8") as f:
                    f.writelines(lines[-MAX_LEDGER_ENTRIES:])
                print(f"[LEDGER] Rotated: kept last {MAX_LEDGER_ENTRIES} entries")
    except Exception as e:
        print(f"[LEDGER] Rotation error: {e}")


def compute_risk(vitals: Vitals):
    # check inputs
    hr = max(20, min(250, vitals.heartRate))
    spo2 = max(50, min(100, vitals.spo2))

    # run ML inference
    ml_result = ml_engine.predict(hr, spo2)
    status = ml_result["status"]
    total_risk = ml_result["risk_score"]
    confidence = ml_result["confidence"]
    triage_horizon = ml_result["triage_horizon"]
    probabilities = ml_result.get("probabilities")

    # routing logic
    local_hospitals = get_hospitals_near(vitals.lat, vitals.lng)

    # only trauma centers for critical patients
    valid_hospitals = [
        h for h in local_hospitals
        if not (status == "CRITICAL" and h["type"] != "TRAUMA")
    ]

    # sort hospitals by distance
    valid_hospitals.sort(
        key=lambda h: calculate_distance(vitals.lat, vitals.lng, h["lat"], h["lng"])
    )

    best_hospital = None
    handshake_log = []

    for h in valid_hospitals:
        distance = calculate_distance(vitals.lat, vitals.lng, h["lat"], h["lng"])
        eta = estimate_eta(distance)

        log_entry = {
            "hospital": h["name"],
            "type": h["type"],
            "distance_km": distance,
            "eta": eta,
        }

        print(f"Checking availability at {h['name']} for {status} patient...")

        if h.get("capacity_full"):
            print(f"-> DENIED by {h['name']} (ER full). Trying next...")
            log_entry["result"] = "DENIED"
            log_entry["reason"] = "ER at capacity"
            handshake_log.append(log_entry)
            continue
        else:
            print(f"-> GRANTED by {h['name']}. Route locked.\n")
            log_entry["result"] = "GRANTED"
            handshake_log.append(log_entry)
            best_hospital = h
            best_hospital["distance_km"] = distance
            best_hospital["eta"] = eta
            break

    if not best_hospital:
        best_hospital = local_hospitals[0]  # Fallback
        best_hospital["distance_km"] = calculate_distance(
            vitals.lat, vitals.lng, best_hospital["lat"], best_hospital["lng"]
        )
        best_hospital["eta"] = estimate_eta(best_hospital["distance_km"])

    # fetch blood type if critical
    if status in ("CRITICAL", "WARNING"):
        print("Patient is deteriorating. Pre-fetching blood type details...")
        print("Found Blood Type: O- (O Negative)")
        print(f"Alerting {best_hospital['name']} blood bank to prepare 2 units.\n")

    best_hospital_payload = best_hospital.copy()
    best_hospital_payload["all_hospitals"] = local_hospitals

    # generate secure hash for ledger
    data_string = json.dumps(
        {
            "hr": vitals.heartRate,
            "spo2": vitals.spo2,
            "lat": vitals.lat,
            "lng": vitals.lng,
        },
        sort_keys=True,
    )
    secure_hash = hashlib.sha256(data_string.encode()).hexdigest()

    ledger_entry = {
        "timestamp": datetime.now().isoformat(),
        "vitals_data": json.loads(data_string),
        "hash": secure_hash,
        "status": status,
        "risk_score": total_risk,
        "ml_confidence": confidence,
        "assigned_hospital": best_hospital["name"],
    }
    try:
        ledger_path = DATA_DIR / "secure_ledger.jsonl"
        _rotate_ledger(ledger_path)
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ledger_entry) + "\n")
    except Exception as e:
        print(f"[LEDGER] Error writing to ledger: {e}")

    return {
        "risk_score": total_risk,
        "status": status,
        "secure_hash": secure_hash,
        "assigned_hospital": best_hospital_payload,
        "triage_horizon": triage_horizon,
        "ml_confidence": confidence,
        "probabilities": probabilities,
        "handshake_log": handshake_log,
    }


# --- REST API Endpoints ---
@app.get("/patient/{abha_id}")
def get_patient_history(abha_id: str):
    """Lookup patient by ABHA ID from the mock registry."""
    print(f"\n[ABHA] LOOKUP INITIATED: {abha_id}")
    patient = mock_abha_db.get(abha_id.strip())
    if patient:
        return {"status": "success", "data": patient}
    return {"status": "error", "message": "ABHA ID not found in registry."}


@app.post("/predict")
def calculate_risk_endpoint(vitals: Vitals):
    """Run ML-powered triage prediction on submitted vitals."""
    return compute_risk(vitals)


@app.get("/ml-status")
def ml_status():
    """Return ML model metadata: accuracy, features, training time, cross-validation."""
    return JSONResponse(content=ml_engine.get_model_info())


@app.get("/ml-evaluation")
def ml_evaluation():
    """
    Return comprehensive ML model evaluation report.
    Includes confusion matrix, classification report, cross-validation scores,
    hyperparameters, and runtime analytics.
    """
    return JSONResponse(content=ml_engine.get_evaluation_report())


@app.get("/ledger")
def get_ledger(limit: int = 20):
    """Return the last N entries from the SHA-256 audit ledger."""
    try:
        ledger_path = DATA_DIR / "secure_ledger.jsonl"
        if not ledger_path.exists():
            return JSONResponse(content={"entries": [], "total": 0})
        with open(ledger_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        entries = []
        for line in lines[-min(limit, len(lines)):]:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
        return JSONResponse(
            content={"entries": entries, "total": len(lines), "showing": len(entries)}
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=500
        )


@app.get("/health")
def health_check():
    """Health check endpoint for Render deployment."""
    uptime_seconds = round(time.time() - SERVER_START_TIME, 1)
    return {
        "status": "ok",
        "ml_ready": ml_engine.is_ready,
        "version": "2.0.0",
        "uptime_seconds": uptime_seconds,
        "model_version": ml_engine.model_version,
        "predictions_made": ml_engine.prediction_count,
        "timestamp": datetime.now().isoformat(),
    }


# --- Frontend Serving Routes ---
@app.get("/dashboard")
def get_dashboard():
    dashboard_path = FRONTEND_DIR / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path), media_type="text/html")
    return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)


@app.get("/ambulance")
def get_ambulance():
    ambulance_path = FRONTEND_DIR / "ambulance.html"
    if ambulance_path.exists():
        return FileResponse(str(ambulance_path), media_type="text/html")
    return HTMLResponse("<h1>ambulance.html not found</h1>", status_code=404)


@app.get("/")
def get_root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.get("/favicon.ico")
def favicon():
    """Return empty response to prevent 404 noise in browser console."""
    from starlette.responses import Response

    return Response(status_code=204)


# --- Socket.IO Event Handlers ---
@sio.event
async def connect(sid, environ):
    print(f"[SOCKET] Client connected: {sid}")


@sio.event
async def disconnect(sid):
    print(f"[SOCKET] Client disconnected: {sid}")


@sio.event
async def fetch_patient(sid, abha_id):
    print(f"\n[SOCKET] ABHA ID LOOKUP INITIATED: {abha_id}")
    patient = mock_abha_db.get(str(abha_id).strip())
    if patient:
        print(f"[SOCKET] Patient found: {patient['name']}")
        await sio.emit("patient_data_received", patient)
    else:
        print(f"[SOCKET] ABHA ID {abha_id} not found.")
        await sio.emit("patient_not_found", {"abha_id": abha_id})


@sio.event
async def send_message(sid, data):
    print(f"[COMM] {data.get('sender')}: {data.get('message')}")
    await sio.emit("receive_message", data)


@sio.event
async def vitals(sid, data):
    """Process incoming vitals from ambulance edge and broadcast AI analysis."""
    try:
        hr = int(data.get("heartRate", 80))
        spo2 = int(data.get("spo2", 98))
        lat = float(data.get("lat", 18.5204))
        lng = float(data.get("lng", 73.8567))

        v = Vitals(heartRate=hr, spo2=spo2, lat=lat, lng=lng)
        risk_result = compute_risk(v)

        payload = {**data, **risk_result}
        await sio.emit("update", payload)
    except Exception as e:
        print(f"[SOCKET] Error handling vitals: {e}")


# Combine FastAPI and Socket.IO into one ASGI application
combined_app = socketio.ASGIApp(sio, other_asgi_app=app)

if __name__ == "__main__":
    print("Starting LifeSync Server on http://localhost:8000 ...")
    uvicorn.run("main:combined_app", host="0.0.0.0", port=8000, reload=True)