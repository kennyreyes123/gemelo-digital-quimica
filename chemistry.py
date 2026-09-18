"""
chemistry.py
============
MOTOR QUÍMICO del gemelo digital. Este módulo es *puro*: no sabe nada de redes,
hilos ni interfaces. Sólo recibe números y devuelve números, lo que permite
probarlo de forma aislada (ver tests/test_chemistry.py).

Contenido:
    1. Ecuación de Arrhenius            k(T) = A·exp(-Ea/RT)
    2. Ley de velocidad de reacción     r = k·[A]^n
    3. Balance de energía del reactor   m·cp·dT/dt = Q_camisa + Q_reacción - Q_pérdidas
    4. Equilibrio ácido-base débil      Ka = [H+][A-]/[HA]  ->  pH
    5. Presión de la cámara             Dalton + Antoine + ley de Gay-Lussac
    6. Letalidad acumulada              F = ∫ 10^((T-Tref)/z) dt
    7. Integrador del sistema de EDOs   (SciPy solve_ivp, método LSODA)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

from config import (
    ANTOINE_H2O,
    KineticParams,
    MMHG_TO_KPA,
    P_ATM,
    ProcessProfile,
    R_GAS,
    c_to_k,
    k_to_c,
)

# ---------------------------------------------------------------------------
# 1. ECUACIÓN DE ARRHENIUS
# ---------------------------------------------------------------------------


def arrhenius(A_pre: float, Ea: float, T: float) -> float:
    """
    Constante de velocidad según Arrhenius.

        k(T) = A · exp( -Ea / (R · T) )

    Parameters
    ----------
    A_pre : factor pre-exponencial (mismas unidades que k)
    Ea    : energía de activación [J/mol]
    T     : temperatura ABSOLUTA [K]

    Returns
    -------
    k : constante de velocidad

    Notes
    -----
    Se acota el exponente para evitar overflow/underflow numérico cuando el
    usuario mueve el slider de temperatura a valores extremos. Físicamente,
    exp(-700) ya es cero a efectos prácticos.
    """
    if T <= 0:
        return 0.0
    exponent = -Ea / (R_GAS * T)
    exponent = float(np.clip(exponent, -700.0, 700.0))
    return A_pre * np.exp(exponent)


def k_ratio_two_temperatures(Ea: float, T1: float, T2: float) -> float:
    """
    Forma logarítmica de Arrhenius, útil para que el equipo verifique a mano:

        ln(k2/k1) = -(Ea/R) · (1/T2 - 1/T1)

    Devuelve k2/k1. Sirve para responder "¿cuánto se acelera la reacción si
    subo 10 °C?" sin volver a integrar todo el modelo.
    """
    return float(np.exp(-(Ea / R_GAS) * (1.0 / T2 - 1.0 / T1)))


# ---------------------------------------------------------------------------
# 2. LEY DE VELOCIDAD DE REACCIÓN
# ---------------------------------------------------------------------------


def reaction_rate(kin: KineticParams, C: float, T: float) -> float:
    """
    Velocidad de consumo del reactivo A para un orden n arbitrario.

        r = k(T) · [A]^n      con n = 0, 1, 2, ...

    La concentración se satura en 0 para que el integrador nunca evalúe
    potencias de números negativos (fuente clásica de NaN).
    """
    C_safe = max(C, 0.0)
    if C_safe <= 0.0:
        # Sin reactivo no hay reacción. El caso crítico es el orden CERO:
        # como C^0 = 1, sin esta guarda la velocidad seguiría siendo k y la
        # concentración se volvería negativa (un absurdo físico).
        return 0.0
    k = arrhenius(kin.A_pre, kin.Ea, T)
    return k * (C_safe ** kin.order)


def half_life(kin: KineticParams, C0: float, T: float) -> float:
    """
    Tiempo de vida media analítico, según el orden de reacción.
    Se usa como control de calidad del resultado numérico.

        orden 0 : t½ = C0 / (2k)
        orden 1 : t½ = ln(2) / k
        orden 2 : t½ = 1 / (k · C0)
    """
    k = arrhenius(kin.A_pre, kin.Ea, T)
    if k <= 0:
        return float("inf")
    if kin.order == 0:
        return C0 / (2.0 * k)
    if kin.order == 1:
        return float(np.log(2.0) / k)
    if kin.order == 2:
        return 1.0 / (k * C0) if C0 > 0 else float("inf")
    raise ValueError(f"Orden {kin.order} sin solución analítica implementada.")


# ---------------------------------------------------------------------------
# 3. EQUILIBRIO ÁCIDO-BASE  ->  pH
# ---------------------------------------------------------------------------


def ph_free_acid(C_HA: float, Ka: float) -> float:
    """
    pH que tendría el ácido débil SOLO en agua, sin efecto tampón.

    Para HA <-> H+ + A-, con Ka = [H+][A-]/[HA] y [H+] = [A-] = x:

        x² + Ka·x - Ka·C_HA = 0
        x = ( -Ka + sqrt(Ka² + 4·Ka·C_HA) ) / 2
        pH = -log10(x)

    Es el "piso": el medio nunca puede estar más ácido que esto.
    """
    C_HA = max(C_HA, 0.0)
    if C_HA <= 0.0:
        return 7.0
    x = (-Ka + np.sqrt(Ka * Ka + 4.0 * Ka * C_HA)) / 2.0
    return float(np.clip(-np.log10(max(x, 1e-14)), 0.0, 14.0))


def ph_from_weak_acid(C_HA: float, Ka: float, pH0: float, beta: float) -> float:
    """
    pH del alimento considerando su capacidad tampón.

    La leche no es agua: caseínas, fosfatos y citratos captan protones. La
    capacidad tampón β se define como los moles de ácido por litro necesarios
    para bajar el pH una unidad:

        pH = pH0 - C_HA / β

    Esa recta vale mientras el tampón no se agote. El límite inferior lo pone
    la disociación del propio ácido (`ph_free_acid`), así que:

        pH = max( pH0 - C_HA/β ,  pH_ácido_libre )

    Ignorar β es el error clásico de estos modelos: predice pH 2.4 para un
    yogur que en realidad está en 4.4.

    Parameters
    ----------
    C_HA : concentración de ácido generado [mol/L]
    Ka   : constante de acidez del ácido dominante
    pH0  : pH inicial del medio
    beta : capacidad tampón [mol/(L·unidad de pH)]
    """
    C_HA = max(C_HA, 0.0)
    ph_buffered = pH0 - C_HA / max(beta, 1e-9)
    # El piso nunca puede quedar por encima del pH inicial: añadir un ácido
    # jamás alcaliniza el medio. Sin este `min` la fórmula del ácido libre,
    # que ignora la autoprotólisis del agua, devolvería pH > 7 cuando C_HA es
    # casi cero y el `max` de abajo elegiría la rama equivocada.
    ph_floor = min(ph_free_acid(C_HA, Ka), pH0)
    return float(np.clip(max(ph_buffered, ph_floor), 0.0, 14.0))


# ---------------------------------------------------------------------------
# 4. PRESIÓN DEL SISTEMA
# ---------------------------------------------------------------------------


def water_vapor_pressure_kPa(T_kelvin: float) -> float:
    """
    Presión de vapor del agua por la ecuación de Antoine.

        log10(P_mmHg) = A - B / (C + T_°C)
    """
    t_c = float(np.clip(k_to_c(T_kelvin), 0.5, 99.5))
    p_mmhg = 10.0 ** (ANTOINE_H2O["A"] - ANTOINE_H2O["B"] / (ANTOINE_H2O["C"] + t_c))
    return p_mmhg * MMHG_TO_KPA


def system_pressure_kPa(T_kelvin: float, T0_kelvin: float, gas_extra_mol: float = 0.0,
                        headspace_L: float = 10.0) -> float:
    """
    Presión total del espacio de cabeza, por la ley de Dalton:

        P_total = P_aire(T) + P_vapor_agua(T) + P_gases_reacción

    - Aire seco: se expande a volumen constante (Gay-Lussac) => P = P0·(T/T0)
    - Vapor de agua: Antoine
    - Gases de reacción (CO2 en fermentación heteroláctica): gas ideal PV = nRT

    Returns
    -------
    Presión absoluta [kPa]
    """
    p_air = P_ATM * (T_kelvin / T0_kelvin)
    p_vap = water_vapor_pressure_kPa(T_kelvin)
    # PV = nRT con V en m³ => P en Pa; headspace_L / 1000 => m³
    p_gas = (gas_extra_mol * R_GAS * T_kelvin) / (headspace_L / 1000.0) / 1000.0
    return float(p_air + p_vap + p_gas)


# ---------------------------------------------------------------------------
# 5. LETALIDAD ACUMULADA (valor F)
# ---------------------------------------------------------------------------


def lethality_rate(T_kelvin: float, T_ref_C: float, z_C: float) -> float:
    """
    Velocidad letal instantánea L(T) usada en el cálculo del valor F:

        L(T) = 10^((T - T_ref) / z)          [s equivalentes por segundo real]
        F    = ∫ L(T) dt

    F >= F_objetivo significa que el lote alcanzó la reducción logarítmica
    exigida por la normativa, aunque el perfil de temperatura no fuera plano.
    """
    return float(10.0 ** ((k_to_c(T_kelvin) - T_ref_C) / z_C))


# ---------------------------------------------------------------------------
# 6. BALANCE DE MATERIA Y ENERGÍA: SISTEMA DE EDOs
# ---------------------------------------------------------------------------
#
# Vector de estado y = [C_A, C_B, C_HA, T, N, F]
#
#   dC_A/dt  = -(r_main + r_acid)                         balance de materia A
#   dC_B/dt  = +r_main                                    producto principal
#   dC_HA/dt = +r_acid                                    ácido -> define el pH
#   dT/dt    = (Q_camisa + Q_reacción - Q_pérdidas)/(m·cp)   balance de energía
#   dN/dt    = -k_micro(T) · N                            inactivación térmica
#   dF/dt    = 10^((T-Tref)/z)                            letalidad acumulada
#
# ---------------------------------------------------------------------------

IDX_CA, IDX_CB, IDX_CHA, IDX_T, IDX_N, IDX_F = range(6)


def jacket_temperature(T_kelvin: float, profile: ProcessProfile,
                       setpoint_C: float) -> float:
    """
    Controlador proporcional del medio de calentamiento/enfriamiento.

        T_camisa = T_setpoint + Kp · (T_setpoint - T_producto)

    Emula un lazo de control industrial simple. La salida se satura en los
    límites físicos del servicio (vapor / agua helada).
    """
    error_C = setpoint_C - k_to_c(T_kelvin)
    t_jacket = setpoint_C + profile.Kp * error_C
    return float(np.clip(t_jacket, profile.jacket_min_C, profile.jacket_max_C))


def derivatives(t: float, y: np.ndarray, profile: ProcessProfile,
                setpoint_C: float) -> np.ndarray:
    """
    Función f(t, y) que entrega solve_ivp. Aquí vive toda la física.
    """
    C_A, C_B, C_HA, T, N, _F = y
    C_A = max(C_A, 0.0)
    N = max(N, 0.0)

    # --- Cinética -----------------------------------------------------------
    r_main = reaction_rate(profile.main_reaction, C_A, T)   # mol/(L·s)
    r_acid = reaction_rate(profile.acid_reaction, C_A, T)   # mol/(L·s)
    k_micro = arrhenius(profile.microbial.A_pre, profile.microbial.Ea, T)

    # --- Balance de materia -------------------------------------------------
    dCA = -(r_main + r_acid)          # A se consume por ambas vías
    dCB = r_main                      # vía principal: A -> B
    dCHA = profile.stoich_acid * r_acid   # vía ácida: A -> ν·HA
    dN = -k_micro * N

    # --- Balance de energía -------------------------------------------------
    # Q_camisa  [W] = UA · (T_camisa - T_producto)
    t_jacket_K = c_to_k(jacket_temperature(T, profile, setpoint_C))
    q_jacket = profile.UA * (t_jacket_K - T)

    # Q_reacción [W] = (-ΔH) · r · V      (r en mol/(L·s), V en L => mol/s)
    q_rxn = (
        (-profile.main_reaction.delta_H) * r_main
        + (-profile.acid_reaction.delta_H) * r_acid
    ) * profile.volume_L

    # Q_pérdidas [W] = UA_loss · (T_producto - T_ambiente)
    q_loss = profile.UA_loss * (T - c_to_k(profile.ambient_C))

    dT = (q_jacket + q_rxn - q_loss) / (profile.mass_kg * profile.cp)

    # --- Letalidad ----------------------------------------------------------
    dF = lethality_rate(T, profile.T_ref_leth_C, profile.z_leth_C)

    return np.array([dCA, dCB, dCHA, dT, dN, dF], dtype=float)


def initial_state(profile: ProcessProfile) -> np.ndarray:
    """Vector de estado en t = 0 a partir del perfil de proceso."""
    return np.array(
        [profile.CA0, 0.0, 0.0, c_to_k(profile.T0_C), profile.N0, 0.0],
        dtype=float,
    )


def step(y: np.ndarray, dt: float, profile: ProcessProfile,
         setpoint_C: float) -> np.ndarray:
    """
    Avanza el estado del gemelo digital un intervalo dt (segundos de proceso).

    Se usa LSODA porque el sistema es RÍGIDO (stiff): la inactivación
    microbiana tiene una constante de tiempo de segundos mientras que el
    calentamiento del tanque tarda minutos. Un Euler explícito con dt = 1 s
    divergiría.

    Returns
    -------
    Nuevo vector de estado, saneado (sin negativos, sin NaN).
    """
    sol = solve_ivp(
        fun=derivatives,
        t_span=(0.0, dt),
        y0=y,
        args=(profile, setpoint_C),
        method="LSODA",
        rtol=1e-6,
        atol=1e-10,
        max_step=dt,
    )
    y_new = sol.y[:, -1] if sol.success else y.copy()
    y_new = np.nan_to_num(y_new, nan=0.0, posinf=0.0, neginf=0.0)
    y_new[IDX_CA] = max(y_new[IDX_CA], 0.0)
    y_new[IDX_CB] = max(y_new[IDX_CB], 0.0)
    y_new[IDX_CHA] = max(y_new[IDX_CHA], 0.0)
    y_new[IDX_N] = max(y_new[IDX_N], 0.0)
    return y_new


# ---------------------------------------------------------------------------
# 7. MÉTRICAS DERIVADAS
# ---------------------------------------------------------------------------


def log_reduction(N: float, N0: float) -> float:
    """
    Reducción logarítmica alcanzada:  log10(N0 / N).
    La pasteurización exige >= 5 log para patógenos vegetativos.
    """
    if N <= 0:
        return 12.0        # tope de presentación: "más de 12 log"
    if N0 <= 0:
        return 0.0
    return float(np.clip(np.log10(N0 / N), 0.0, 12.0))


def conversion(C_A: float, C_A0: float) -> float:
    """Conversión fraccional del reactivo limitante: X = (C_A0 - C_A)/C_A0."""
    if C_A0 <= 0:
        return 0.0
    return float(np.clip((C_A0 - C_A) / C_A0, 0.0, 1.0))


def derived_metrics(y: np.ndarray, profile: ProcessProfile) -> Tuple[float, float]:
    """Devuelve (pH, presión_kPa) calculados a partir del estado actual."""
    ph = ph_from_weak_acid(
        y[IDX_CHA], profile.Ka, profile.pH0, profile.buffer_capacity
    )
    # Sólo una fracción del producto B se libera como gas al espacio de cabeza
    # (CO2 de la ruta heteroláctica). En pasteurización gas_yield = 0.
    gas_mol = y[IDX_CB] * profile.volume_L * profile.gas_yield
    pressure = system_pressure_kPa(
        y[IDX_T], c_to_k(profile.T0_C), gas_mol, profile.headspace_L
    )
    return ph, pressure
