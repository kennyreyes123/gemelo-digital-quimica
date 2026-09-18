"""
app.py
======
Capa de servicio del gemelo digital: FastAPI + WebSocket.

Rutas
-----
  GET   /                     -> sirve el dashboard (frontend/index.html)
  WS    /ws/telemetry         -> flujo de tramas JSON, 1 por segundo
  GET   /api/state            -> última trama (polling de respaldo)
  GET   /api/history?limit=N  -> histórico en memoria (recarga de gráficas)
  GET   /api/profile          -> perfil del proceso y límites PCC
  GET   /api/deviations       -> bitácora HACCP del lote
  POST  /api/control/start    -> arranca el hilo
  POST  /api/control/stop     -> pausa el hilo
  POST  /api/control/reset    -> nuevo lote (opcionalmente otro proceso)
  POST  /api/setpoints        -> cambia T, pH0, CA0 o velocidad en vivo

Ejecutar:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from config import PROFILES
from simulator import twin

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="Gemelo Digital de Procesos Alimentarios",
    description="Simulación en tiempo real de cinética química y PCC (Química I)",
    version="1.1.0",
)

# En desarrollo el frontend puede servirse desde otro puerto (Live Server, Vite).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Modelos de entrada
# ---------------------------------------------------------------------------


class SetpointRequest(BaseModel):
    """Cuerpo del POST /api/setpoints. Todos los campos son opcionales."""
    temperature_C: Optional[float] = Field(None, ge=0, le=140,
                                           description="Set point de temperatura [°C]")
    pH0: Optional[float] = Field(None, ge=3.0, le=9.0,
                                 description="pH inicial del medio")
    CA0: Optional[float] = Field(None, ge=0.0, le=1.0,
                                 description="Concentración inicial de A [mol/L]")
    speed: Optional[float] = Field(None, ge=1.0, le=600.0,
                                   description="Segundos de proceso por segundo real")


class ResetRequest(BaseModel):
    process: Optional[str] = Field(None, description="'pasteurizacion' o 'fermentacion'")


# ---------------------------------------------------------------------------
# Ciclo de vida
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    """Enlaza el event loop al hilo del simulador y lo arranca."""
    twin.bind_loop(asyncio.get_running_loop())
    twin.start()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    twin.stop()


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def dashboard():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return JSONResponse(
            {"error": "No se encontro frontend/index.html", "ruta_esperada": str(index)},
            status_code=404,
        )
    return FileResponse(index)


# ---------------------------------------------------------------------------
# WebSocket de telemetría
# ---------------------------------------------------------------------------


@app.websocket("/ws/telemetry")
async def telemetry(ws: WebSocket) -> None:
    """
    Canal push. Cada cliente recibe su propia cola acotada (maxsize=10):
    si el navegador se congela, las tramas viejas se descartan en lugar de
    acumularse en memoria del servidor.
    """
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    twin.subscribers.add(queue)

    try:
        # Enviar inmediatamente el último estado para que la UI no arranque vacía.
        await ws.send_json({"type": "snapshot", "payload": twin.snapshot()})
        await ws.send_json({"type": "history", "payload": twin.get_history(240)})

        while True:
            frame = await queue.get()
            await ws.send_json({"type": "telemetry", "payload": frame})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[ws] cierre inesperado: {exc!r}")
    finally:
        twin.subscribers.discard(queue)


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------


@app.get("/api/state")
async def get_state():
    return twin.snapshot()


@app.get("/api/history")
async def get_history(limit: int = 300):
    return {"count": min(limit, 1800), "frames": twin.get_history(limit)}


@app.get("/api/profile")
async def get_profile():
    """Perfil activo + catálogo de procesos disponibles, para poblar la UI."""
    p = twin.profile
    return {
        "active": p.kind,
        "label": p.label,
        "available": [{"key": k, "label": v.label} for k, v in PROFILES.items()],
        "limits": {k: vars(v) for k, v in p.limits.items()},
        "kinetics": {
            "main_reaction": vars(p.main_reaction),
            "acid_reaction": vars(p.acid_reaction),
            "microbial": vars(p.microbial),
        },
        "initial": {"T0_C": p.T0_C, "CA0": p.CA0, "pH0": p.pH0, "N0": p.N0},
        "lethality": {"T_ref_C": p.T_ref_leth_C, "z_C": p.z_leth_C,
                      "F_target_s": p.F_target_s},
    }


@app.get("/api/deviations")
async def get_deviations():
    """Bitácora de desviaciones: es el anexo del informe HACCP del lote."""
    return {"batch_id": twin.batch_id, "deviations": twin.deviation_log()}


@app.post("/api/setpoints")
async def post_setpoints(req: SetpointRequest):
    applied = twin.set_setpoint(
        temperature_C=req.temperature_C,
        pH0=req.pH0,
        CA0=req.CA0,
        speed=req.speed,
    )
    return {"ok": True, "applied": applied}


@app.post("/api/control/start")
async def control_start():
    twin.start()
    return {"ok": True, "running": twin.is_running}


@app.post("/api/control/stop")
async def control_stop():
    twin.stop()
    return {"ok": True, "running": twin.is_running}


@app.post("/api/control/reset")
async def control_reset(req: ResetRequest):
    if req.process is not None and req.process not in PROFILES:
        return JSONResponse(
            {"ok": False, "error": f"Proceso desconocido: {req.process}"},
            status_code=400,
        )
    twin.reset(req.process)
    return {"ok": True, "batch_id": twin.batch_id, "process": twin.profile.kind}
