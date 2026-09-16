"""
LifeSync — Application Entry Point
====================================
Starts the LifeSync server with automatic port conflict resolution.
"""

import os
import sys
import signal
import socket
from pathlib import Path

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import uvicorn

PORT = int(os.environ.get("PORT", 8000))


def is_port_in_use(port):
    """Check if a port is currently bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def kill_port(port):
    """Kill any process holding the specified port (Windows only)."""
    if sys.platform == "win32":
        try:
            import subprocess
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
                    print(f"[BOOT] 🔄 Killed stale process on port {port} (PID: {pid})")
        except Exception as e:
            print(f"[BOOT] ⚠️ Could not kill port {port}: {e}")


if __name__ == "__main__":
    backend_dir = Path(__file__).resolve().parent / "backend"
    data_dir = Path(__file__).resolve().parent / "data"
    
    # Ensure directories exist
    sys.path.insert(0, str(backend_dir))
    data_dir.mkdir(exist_ok=True)

    # Resolve port conflicts
    if is_port_in_use(PORT):
        print(f"[BOOT] ⚠️ Port {PORT} is in use. Attempting to free it...")
        kill_port(PORT)

    print("=" * 60)
    print("⚡ LifeSync v2.0 — Agentic AI Emergency Network")
    print("=" * 60)
    print(f"🌐 Main Portal:        http://localhost:{PORT}/")
    print(f"🚑 Ambulance Edge:     http://localhost:{PORT}/ambulance")
    print(f"🏥 Hospital Command:   http://localhost:{PORT}/dashboard")
    print(f"📖 Swagger Docs:       http://localhost:{PORT}/docs")
    print(f"🧠 ML Model Status:    http://localhost:{PORT}/ml-status")
    print(f"💚 Health Check:       http://localhost:{PORT}/health")
    print("=" * 60)

    uvicorn.run(
        "main:combined_app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        app_dir=str(backend_dir)
    )
