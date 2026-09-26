"""
simulator.py
============
Corazón del gemelo digital: un hilo (thread) que avanza el modelo químico en
tiempo real, contamina las variables con ruido gaussiano para emular sensores
IoT y publica una trama JSON cada segundo.

Diseño:

    DigitalTwin (hilo demonio)
        |-- estado y = [C_A, C_B, C_HA, T, N, F]      <- verdad del modelo
        |-- sensores                                  <- y + ruido N(0, sigma)
        |-- AlertEngine                               <- evaluación HACCP
        |-- deque(history)                            <- buffer circular
        `-- subscribers: set[asyncio.Queue]           <- fan-out a WebSockets

El hilo es el único que escribe el estado; los lectores (API REST / WebSocket)
sólo consumen copias. Un `threading.Lock` protege las secciones donde se
modifican los set points desde el dashboard.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Set

import numpy as np

import chemistry as chem
from alerts import AlertEngine
from config import (
    HISTORY_MAXLEN,
    PROFILES,
    SIM_SPEED_DEFAULT,
    SIM_TICK_SECONDS,
    ProcessProfile,
    c_to_k,
    k_to_c,
)


# ---------------------------------------------------------------------------
# Sensores IoT: ruido gaussiano
# ---------------------------------------------------------------------------


class SensorArray:
    """
    Convierte la verdad del modelo en una *lectura* de sensor.

    Un sensor real nunca entrega el valor exacto: añade ruido blanco
    (incertidumbre de repetibilidad) y, opcionalmente, una deriva lenta
    (drift por ensuciamiento del electrodo). Modelar esto es lo que separa un
    gemelo digital de una simple gráfica de ecuaciones.

        lectura = valor_real + N(0, sigma) + deriva(t)
    """

    def __init__(self, sigma: Dict[str, float], seed: Optional[int] = None):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.drift_enabled = True

    def read(self, truth: Dict[str, float], t_process_s: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, value in truth.items():
            sigma = self.sigma.get(name, 0.0)
            noise = float(self.rng.normal(0.0, sigma)) if sigma > 0 else 0.0
            drift = self._drift(name, t_process_s) if self.drift_enabled else 0.0
            reading = value + noise + drift
            if name in self.NEVER_NEGATIVE:
                # El ruido gaussiano es simétrico, pero ninguna de estas
                # cantidades físicas puede ser negativa (no existe una
                # concentración, un Brix o una viscosidad "bajo cero"). Sin
                # este piso, un valor real cercano a 0 (p.ej. viscosidad de
                # 1.6 cP) mostraría lecturas negativas absurdas con cierta
                # frecuencia, en vez de simplemente acercarse a 0.
                reading = max(reading, 0.0)
            out[name] = reading
        return out

    NEVER_NEGATIVE = {
        "concentration_A", "concentration_B", "pressure_kPa", "brix", "viscosity_cP",
    }

    def _drift(self, name: str, t_s: float) -> float:
        """
        Deriva senoidal muy lenta (periodo de 20 min) de amplitud igual a
        medio sigma. Emula el ensuciamiento progresivo de un electrodo de pH
        o la dilatación térmica de un transmisor de presión.
        """
        sigma = self.sigma.get(name, 0.0)
        if sigma <= 0:
            return 0.0
        return 0.5 * sigma * math.sin(2.0 * math.pi * t_s / 1200.0)


# ---------------------------------------------------------------------------
# Gemelo digital
# ---------------------------------------------------------------------------


class DigitalTwin:
    """
    Motor de simulación en tiempo real.

    Uso típico:
        twin = DigitalTwin("pasteurizacion")
        twin.start()
        ...
        twin.set_setpoint(temperature_C=74.0)
        snapshot = twin.snapshot()
    """

    def __init__(self, profile_key: str = "pasteurizacion",
                 speed: float = SIM_SPEED_DEFAULT, seed: Optional[int] = 42):
        self.profile: ProcessProfile = PROFILES[profile_key]
        self.speed = speed                      # segundos de proceso por segundo real
        self._seed = seed

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self.subscribers: Set[asyncio.Queue] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.history: Deque[dict] = deque(maxlen=HISTORY_MAXLEN)
        self._reset_state()

    # -- Ciclo de vida ------------------------------------------------------

    def _reset_state(self) -> None:
        p = self.profile
        self.y = chem.initial_state(p)
        self.t_process_s = 0.0
        self.tick_count = 0
        self.setpoint_C = p.T_setpoint_C
        self.sensors = SensorArray(p.noise_sigma, self._seed)
        self.alert_engine = AlertEngine(p)
        # Los PCC se "arman" sólo cuando el producto entra en la zona de
        # proceso. Durante el calentamiento la leche está a 8 °C y eso NO es
        # una desviación: todavía no ha llegado al tubo de retención. Sin esta
        # lógica el tablero arranca en rojo y el operario aprende a ignorarlo.
        self.ccp_armed = False
        self.phase = "CALENTAMIENTO"
        self.history.clear()
        self.batch_id = datetime.now(timezone.utc).strftime("LOTE-%Y%m%d-%H%M%S")

    def start(self) -> None:
        """Lanza el hilo de simulación si no está corriendo."""
        with self._lock:
            if self._running:
                return
            self._stop_event.clear()
            self._running = True
            self._thread = threading.Thread(
                target=self._run, name="digital-twin", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Pausa la simulación conservando el estado."""
        with self._lock:
            self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def reset(self, profile_key: Optional[str] = None) -> None:
        """Reinicia el lote desde t = 0, opcionalmente cambiando de proceso."""
        was_running = self._running
        self.stop()
        with self._lock:
            if profile_key is not None:
                self.profile = PROFILES[profile_key]
            self._reset_state()
        if was_running:
            self.start()

    @property
    def is_running(self) -> bool:
        return self._running

    # -- Entradas desde el dashboard ---------------------------------------

    def set_setpoint(self, temperature_C: Optional[float] = None,
                     pH0: Optional[float] = None,
                     CA0: Optional[float] = None,
                     speed: Optional[float] = None) -> dict:
        """
        Aplica cambios en vivo solicitados por el usuario.

        - `temperature_C` actúa sobre el lazo de control (efecto inmediato).
        - `pH0` y `CA0` son condiciones iniciales: sólo tienen sentido si se
          reinyecta el lote, así que se aplican al estado actual como si se
          hubiera hecho una corrección de formulación.
        """
        with self._lock:
            if temperature_C is not None:
                self.setpoint_C = float(
                    np.clip(temperature_C, self.profile.jacket_min_C,
                            self.profile.jacket_max_C)
                )
            if pH0 is not None:
                self.profile.pH0 = float(np.clip(pH0, 3.0, 9.0))
            if CA0 is not None:
                CA0 = float(np.clip(CA0, 0.0, 1.0))
                self.profile.CA0 = CA0
                self.y[chem.IDX_CA] = CA0
            if speed is not None:
                self.speed = float(np.clip(speed, 1.0, 600.0))
            return {
                "setpoint_C": self.setpoint_C,
                "pH0": self.profile.pH0,
                "CA0": self.profile.CA0,
                "speed": self.speed,
            }

    # -- Bucle principal ----------------------------------------------------

    def _run(self) -> None:
        """
        Bucle de tiempo real. Usa un reloj monotónico y calcula el siguiente
        instante objetivo en lugar de hacer `sleep(1)`: así el tick no se
        desfasa aunque la integración tarde algunos milisegundos.
        """
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            next_tick += SIM_TICK_SECONDS
            try:
                frame = self._advance()
                self._publish(frame)
            except Exception as exc:                      # nunca matar el hilo
                print(f"[DigitalTwin] error en el tick: {exc!r}")

            delay = next_tick - time.monotonic()
            if delay > 0:
                self._stop_event.wait(delay)
            else:
                next_tick = time.monotonic()              # resincronizar

    def _advance(self) -> dict:
        """Integra un tick, lee sensores, evalúa alertas y arma la trama JSON."""
        with self._lock:
            dt_process = SIM_TICK_SECONDS * self.speed

            # 1) Integrar el modelo químico
            self.y = chem.step(self.y, dt_process, self.profile, self.setpoint_C)
            self.t_process_s += dt_process
            self.tick_count += 1

            p = self.profile
            metrics_now = chem.derived_metrics(self.y, p)

            truth = {
                "temperature_C": k_to_c(self.y[chem.IDX_T]),
                "pH": metrics_now["pH"],
                "pressure_kPa": metrics_now["pressure_kPa"],
                "concentration_A": self.y[chem.IDX_CA],
                "concentration_B": self.y[chem.IDX_CB],
                "brix": metrics_now["brix"],
                "viscosity_cP": metrics_now["viscosity_cP"],
            }

            # 2) Emular sensores IoT (ruido gaussiano + deriva)
            readings = self.sensors.read(truth, self.t_process_s)

            # 3) Armar los PCC al entrar en la zona de proceso
            self._update_phase(readings)

            # 4) Evaluar los PCC contra la LECTURA, no contra la verdad:
            #    un sistema real sólo conoce lo que mide.
            if self.ccp_armed:
                alerts = self.alert_engine.evaluate(readings, self.t_process_s)
                status = AlertEngine.overall_status(alerts)
            else:
                alerts = []
                status = "CALENTAMIENTO"

            # 5) Métricas derivadas
            T_K = self.y[chem.IDX_T]
            k_now = chem.arrhenius(p.main_reaction.A_pre, p.main_reaction.Ea, T_K)
            frame = self._build_frame(truth, readings, alerts, status, k_now, metrics_now)

            self.history.append(frame)
            return frame

    def _update_phase(self, readings: Dict[str, float]) -> None:
        """
        Arma los PCC la primera vez que la temperatura entra en la banda de
        operación. Una vez armados no se desarman: a partir de ese punto,
        cualquier caída SÍ es una desviación que debe registrarse.
        """
        if self.ccp_armed:
            return
        lim = self.profile.limits.get("temperature_C")
        t_read = readings.get("temperature_C")
        if lim is None or t_read is None:
            self.ccp_armed = True
            self.phase = "PROCESO"
            return
        if t_read >= lim.warn_low:
            self.ccp_armed = True
            self.phase = "PROCESO"

    def _build_frame(self, truth, readings, alerts, status, k_now, metrics_now) -> dict:
        """Construye la trama JSON descrita en docs/schema_telemetria.json."""
        p = self.profile
        return {
            "schema_version": "1.2",
            "batch_id": self.batch_id,
            "process": p.kind,
            "process_label": p.label,
            "seq": self.tick_count,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "t_process_s": round(self.t_process_s, 2),
            "readings": {k: round(v, 5) for k, v in readings.items()},
            "truth": {k: round(v, 5) for k, v in truth.items()},
            "state": {
                "concentration_A": round(self.y[chem.IDX_CA], 6),
                "concentration_B": round(self.y[chem.IDX_CB], 6),
                "concentration_HA": round(self.y[chem.IDX_CHA], 6),
                "temperature_K": round(self.y[chem.IDX_T], 3),
                "microbial_count": float(f"{self.y[chem.IDX_N]:.4g}"),
                "lethality_F_s": round(self.y[chem.IDX_F], 3),
                "water_mass_kg": round(self.y[chem.IDX_W], 4),
            },
            "metrics": {
                "k_rate_s": float(f"{k_now:.5g}"),
                "conversion": round(chem.conversion(self.y[chem.IDX_CA], p.CA0), 4),
                "log_reduction": round(
                    chem.log_reduction(self.y[chem.IDX_N], p.N0), 3
                ),
                "lethality_progress": round(
                    min(self.y[chem.IDX_F] / p.F_target_s, 1.0), 4
                ),
                "half_life_s": float(
                    f"{chem.half_life(p.main_reaction, p.CA0, self.y[chem.IDX_T]):.5g}"
                ),
                "water_activity": round(metrics_now["water_activity"], 4),
                "boiling_point_elevation_C": round(
                    metrics_now["boiling_point_elevation_C"], 3
                ),
            },
            "control": {
                "setpoint_C": round(self.setpoint_C, 2),
                "jacket_C": round(
                    chem.jacket_temperature(self.y[chem.IDX_T], p, self.setpoint_C), 2
                ),
                "speed_factor": self.speed,
                "running": self._running,
            },
            "phase": self.phase,
            "ccp_armed": self.ccp_armed,
            "status": status,
            "alerts": [a.to_dict() for a in alerts],
        }

    # -- Publicación a los clientes ----------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Registra el event loop de FastAPI para poder empujar desde el hilo."""
        self._loop = loop

    def _publish(self, frame: dict) -> None:
        """
        Fan-out a todos los WebSockets conectados. Como estamos en un hilo
        distinto al del event loop, se usa `call_soon_threadsafe`.
        Si la cola de un cliente está llena (navegador congelado), se descarta
        la trama: preferimos perder un dato a bloquear el simulador.
        """
        if self._loop is None or not self.subscribers:
            return
        for queue in list(self.subscribers):
            try:
                self._loop.call_soon_threadsafe(queue.put_nowait, frame)
            except (asyncio.QueueFull, RuntimeError):
                continue

    # -- Lectura para la API ------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            if self.history:
                return self.history[-1]
            return {"status": "IDLE", "message": "La simulacion aun no ha producido datos."}

    def get_history(self, limit: int = 300) -> List[dict]:
        with self._lock:
            return list(self.history)[-limit:]

    def deviation_log(self) -> List[dict]:
        with self._lock:
            return list(self.alert_engine.deviation_log)


# Instancia única compartida por la API (patrón singleton simple).
twin = DigitalTwin("pasteurizacion")


if __name__ == "__main__":
    # Prueba rápida sin servidor: python simulator.py
    twin.start()
    try:
        for _ in range(10):
            time.sleep(1.0)
            snap = twin.snapshot()
            r = snap.get("readings", {})
            print(
                f"t={snap.get('t_process_s'):>7.1f}s  "
                f"T={r.get('temperature_C', 0):6.2f} C  "
                f"pH={r.get('pH', 0):5.3f}  "
                f"P={r.get('pressure_kPa', 0):7.2f} kPa  "
                f"log-red={snap.get('metrics', {}).get('log_reduction'):5.2f}  "
                f"[{snap.get('status')}]"
            )
    finally:
        twin.stop()
