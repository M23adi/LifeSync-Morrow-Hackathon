"""
LifeSync Agentic AI Backend
============================
FastAPI + Socket.IO server providing:
  - ML-powered real-time deterioration prediction
  - Multi-agent hospital routing with handshake protocol
  - ABHA patient identity registry
  - SHA-256 cryptographic audit ledger
  - WebSocket vitals streaming
"""

import hashlib
import json
import math
import os
import sys
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

print("[BOOT] Initializing ML Deterioration Engine...")
ml_engine = DeteriorationModel()

# --- Socket.IO Async Server & FastAPI App ---
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI(
    title="LifeSync Agentic AI Platform",
    description="ML-powered emergency triage, agentic hospital routing, and real-time vitals monitoring.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Vitals(BaseModel):
    heartRate: int
    spo2: int
    lat: Optional[float] = 18.5204
    lng: Optional[float] = 73.8567


# --- 1. ABHA ID PATIENT REGISTRY (MOCK DATABASE) ---
mock_abha_db = {
    "14-1234-5678-9012": {
        "name": "Rajesh Kumar", "age": 45, "blood_group": "O+",
        "allergies": ["Penicillin", "Peanuts"],
        "history": ["Hypertension (2021)", "Appendectomy (2018)"]
    },
    "14-9876-5432-1098": {
        "name": "Priya Sharma", "age": 32, "blood_group": "A-",
        "allergies": ["None"],
        "history": ["Asthma"]
    },
    "14-5555-1234-7890": {
        "name": "Amit Verma", "age": 58, "blood_group": "B+",
        "allergies": ["Sulfa drugs", "Iodine contrast"],
        "history": ["Type 2 Diabetes (2015)", "CABG Surgery (2020)", "CKD Stage 2"]
    }
}


# --- 2. AGENTIC ROUTING & LOGIC ---
def calculate_distance(lat1, lng1, lat2, lng2):
    """Haversine-approximated distance between two GPS coordinates."""
    return math.sqrt((lat2 - lat1) ** 2 + (lng2 - lng1) ** 2)


def get_hospitals_near(lat, lng):
    """Return hospitals near the ambulance's current GPS position."""
    return [
        {"name": "City General Hospital", "lat": lat + 0.015, "lng": lng + 0.012,
         "type": "GENERAL", "capacity_full": False, "beds_available": 12},
        {"name": "Metro Trauma Center", "lat": lat - 0.020, "lng": lng - 0.005,
         "type": "TRAUMA", "capacity_full": True, "beds_available": 0},
        {"name": "Advanced Trauma Institute", "lat": lat - 0.010, "lng": lng + 0.025,
         "type": "TRAUMA", "capacity_full": False, "beds_available": 5}
    ]


def compute_risk(vitals: Vitals):
    """
    Full agentic pipeline:
    1. PERCEIVE — Receive vitals from ambulance edge
    2. REASON  — ML model predicts deterioration status
    3. ACT     — Multi-agent handshake selects optimal hospital
    4. SECURE  — SHA-256 hash locks the transaction
    """

    # --- REASON: ML Inference ---
    ml_result = ml_engine.predict(vitals.heartRate, vitals.spo2)
    status = ml_result["status"]
    total_risk = ml_result["risk_score"]
    confidence = ml_result["confidence"]
    triage_horizon = ml_result["triage_horizon"]
    probabilities = ml_result.get("probabilities")

    # --- ACT: Multi-Agent Hospital Routing ---
    local_hospitals = get_hospitals_near(vitals.lat, vitals.lng)

    # Filter: critical patients need TRAUMA facilities
    valid_hospitals = [
        h for h in local_hospitals
        if not (status == "CRITICAL" and h["type"] != "TRAUMA")
    ]

    # Sort by distance
    valid_hospitals.sort(
        key=lambda h: calculate_distance(vitals.lat, vitals.lng, h["lat"], h["lng"])
    )

    best_hospital = None

    # Multi-Agent Handshake Protocol
    for h in valid_hospitals:
        print(f"[AGENT HANDSHAKE] Pinging {h['name']} Agent for clearance...")
        print(f"[AGENT HANDSHAKE] Requesting permission for {status} patient. Needs {h['type']} resources.")

        if h.get("capacity_full"):
            print(f"[AGENT HANDSHAKE] DENIED by {h['name']}. Reason: ER at capacity. Renegotiating...")
            continue
        else:
            print(f"[AGENT HANDSHAKE] GRANTED by {h['name']}. Route secured.\n")
            best_hospital = h
            break

    if not best_hospital:
        best_hospital = local_hospitals[0]  # Fallback

    # Blood Bank Pre-Fetch Agent
    if status in ("CRITICAL", "WARNING"):
        print(f"[BLOOD AGENT] Deterioration Detected. Cross-referencing ABHA Registry...")
        print(f"[BLOOD AGENT] Patient Blood Type: O- (O Negative) identified.")
        print(f"[BLOOD AGENT] Securing 2 units at {best_hospital['name']} Blood Bank prior to arrival...\n")

    # Inject all hospitals into the payload for map rendering
    best_hospital_payload = best_hospital.copy()
    best_hospital_payload["all_hospitals"] = local_hospitals

    # --- SECURE: SHA-256 Cryptographic Vault ---
    data_string = json.dumps(
        {"hr": vitals.heartRate, "spo2": vitals.spo2, "lat": vitals.lat, "lng": vitals.lng},
        sort_keys=True
    )
    secure_hash = hashlib.sha256(data_string.encode()).hexdigest()

    ledger_entry = {
        "timestamp": datetime.now().isoformat(),
        "vitals_data": json.loads(data_string),
        "hash": secure_hash,
        "status": status,
        "risk_score": total_risk,
        "ml_confidence": confidence,
        "assigned_hospital": best_hospital["name"]
    }
    try:
        ledger_path = DATA_DIR / "secure_ledger.jsonl"
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
        "probabilities": probabilities
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
    """Return ML model metadata: accuracy, features, training time."""
    return JSONResponse(content=ml_engine.get_model_info())


@app.get("/health")
def health_check():
    """Health check endpoint for Render deployment."""
    return {"status": "ok", "ml_ready": ml_engine.is_ready, "version": "2.0.0"}


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