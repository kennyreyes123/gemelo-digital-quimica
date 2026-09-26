"""
alerts.py
=========
Motor de alertas de inocuidad. Evalúa cada lectura del sensor contra los
Límites Críticos de Control (PCC) definidos en config.py y clasifica el estado
del lote.

Niveles:
    OK       -> dentro de la banda de operación
    WARNING  -> fuera de la banda deseada, aún dentro del límite crítico
    CRITICAL -> límite crítico violado: el lote requiere acción correctiva

Se aplica HISTÉRESIS: una alarma no se limpia en cuanto la variable vuelve a
cruzar el umbral, sino cuando entra con un margen. Sin esto, el ruido del
sensor produciría decenas de alarmas por minuto (fenómeno de "chattering",
el motivo nº 1 por el que los operarios silencian los tableros reales).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from config import CriticalLimit, ProcessProfile

OK, WARNING, CRITICAL = "OK", "WARNING", "CRITICAL"

# Fracción del ancho de banda que la variable debe recuperar para limpiar la alarma
HYSTERESIS_FRACTION = 0.08


@dataclass
class Alert:
    """Una desviación detectada en un PCC."""
    ccp_id: str
    variable: str
    level: str
    value: float
    unit: str
    limit_low: float
    limit_high: float
    message: str
    corrective_action: str

    def to_dict(self) -> dict:
        return {
            "ccp_id": self.ccp_id,
            "variable": self.variable,
            "level": self.level,
            "value": round(self.value, 4),
            "unit": self.unit,
            "limit_low": self.limit_low,
            "limit_high": self.limit_high,
            "message": self.message,
            "corrective_action": self.corrective_action,
        }


# Acciones correctivas sugeridas por variable y sentido de la desviación.
CORRECTIVE_ACTIONS: Dict[str, Dict[str, str]] = {
    "temperature_C": {
        "low": "Desviar el flujo a la válvula de retorno y recircular hasta "
               "recuperar el set point. El producto no puede pasar a llenado.",
        "high": "Abrir el bypass de agua de enfriamiento y reducir la apertura "
                "de vapor. Verificar sabor a cocido en el lote.",
    },
    "pH": {
        "low": "Detener el calentamiento: riesgo de coagulación ácida. Tomar "
               "muestra para recuento de flora láctica en la materia prima.",
        "high": "Verificar calibración del electrodo y descartar adición "
                "accidental de neutralizantes (práctica prohibida).",
    },
    "pressure_kPa": {
        "low": "Revisar la bomba de alimentación. Si la presión del producto cae "
               "por debajo del medio calefactor hay riesgo de contaminación cruzada.",
        "high": "Aliviar por la válvula de seguridad y revisar obstrucción en la "
                "línea de salida antes de continuar.",
    },
    "brix": {
        "low": "El concentrado no alcanzó el °Brix objetivo: extender el tiempo "
               "de evaporación. Un producto bajo en sólidos tiene actividad de "
               "agua demasiado alta para ser estable sin refrigeración.",
        "high": "Detener la evaporación de inmediato: sobre 70 °Bx hay riesgo de "
                "cristalización de azúcares en el tanque y en la tubería.",
    },
    "viscosity_cP": {
        "low": "Verificar el viscosímetro en línea; una lectura anormalmente "
               "baja puede indicar dilución accidental o falla del sensor.",
        "high": "Detener la concentración: el producto está en riesgo de dejar "
                "de ser bombeable. Diluir con jugo de menor Brix si es posible "
                "para recuperar la fluidez antes de continuar el trasiego.",
    },
}


class AlertEngine:
    """
    Evaluador con memoria. Mantener una instancia por sesión de simulación.
    """

    def __init__(self, profile: ProcessProfile):
        self.profile = profile
        self._active: Dict[str, str] = {}      # variable -> nivel activo
        self.deviation_log: List[dict] = []    # bitácora para el informe HACCP

    # -- API pública --------------------------------------------------------

    def evaluate(self, readings: Dict[str, float], t_process_s: float) -> List[Alert]:
        """
        Evalúa todas las lecturas y devuelve las alertas vigentes.

        Parameters
        ----------
        readings : dict {nombre_variable: valor_medido}
        t_process_s : tiempo de proceso transcurrido [s], para la bitácora
        """
        alerts: List[Alert] = []
        for var, limit in self.profile.limits.items():
            if var not in readings:
                continue
            alert = self._evaluate_one(limit, readings[var], t_process_s)
            if alert is not None:
                alerts.append(alert)
        return alerts

    @staticmethod
    def overall_status(alerts: List[Alert]) -> str:
        """Estado agregado del lote: el peor nivel presente."""
        if any(a.level == CRITICAL for a in alerts):
            return CRITICAL
        if any(a.level == WARNING for a in alerts):
            return WARNING
        return OK

    def reset(self) -> None:
        """Limpia el estado al arrancar un lote nuevo."""
        self._active.clear()
        self.deviation_log.clear()

    # -- Interno ------------------------------------------------------------

    def _evaluate_one(self, limit: CriticalLimit, value: float,
                      t_process_s: float) -> Optional[Alert]:
        previous = self._active.get(limit.variable, OK)
        level, side = self._classify(limit, value, previous)

        if level == OK:
            if previous != OK:
                self._active[limit.variable] = OK
                self._log(limit, value, "RECUPERADO", t_process_s)
            return None

        if level != previous:
            self._log(limit, value, level, t_process_s)
        self._active[limit.variable] = level

        direction = "por debajo de" if side == "low" else "por encima de"
        threshold = (
            (limit.crit_low if side == "low" else limit.crit_high)
            if level == CRITICAL
            else (limit.warn_low if side == "low" else limit.warn_high)
        )
        msg = (
            f"{limit.ccp_id}: {limit.variable} = {value:.2f} {limit.unit}, "
            f"{direction} {threshold:.2f} {limit.unit}."
        )
        return Alert(
            ccp_id=limit.ccp_id,
            variable=limit.variable,
            level=level,
            value=value,
            unit=limit.unit,
            limit_low=limit.crit_low,
            limit_high=limit.crit_high,
            message=msg,
            corrective_action=CORRECTIVE_ACTIONS.get(limit.variable, {}).get(side, ""),
        )

    @staticmethod
    def _classify(limit: CriticalLimit, value: float, previous: str):
        """
        Devuelve (nivel, lado). Aplica histéresis: si ya había alarma activa,
        los umbrales se "encogen" para que la variable tenga que recuperarse
        con margen antes de limpiar.
        """
        span = max(limit.crit_high - limit.crit_low, 1e-9)
        h = span * HYSTERESIS_FRACTION if previous != OK else 0.0

        if value < limit.crit_low + h:
            return CRITICAL, "low"
        if value > limit.crit_high - h:
            return CRITICAL, "high"
        if value < limit.warn_low + h:
            return WARNING, "low"
        if value > limit.warn_high - h:
            return WARNING, "high"
        return OK, ""

    def _log(self, limit: CriticalLimit, value: float, level: str,
             t_process_s: float) -> None:
        self.deviation_log.append(
            {
                "t_process_s": round(t_process_s, 1),
                "ccp_id": limit.ccp_id,
                "variable": limit.variable,
                "value": round(value, 4),
                "level": level,
                "rationale": limit.rationale,
            }
        )
        # La bitácora no crece sin control durante demostraciones largas.
        if len(self.deviation_log) > 500:
            del self.deviation_log[:100]
