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
    6. Propiedades coligativas          °Brix, actividad de agua (Raoult),
                                        elevación del punto de ebullición
    7. Viscosidad                      η(T) = A·exp(+Eη/RT)  (Arrhenius invertido)
    8. Letalidad acumulada              F = ∫ 10^((T-Tref)/z) dt
    9. Integrador del sistema de EDOs   (SciPy solve_ivp, método LSODA)
"""

from __future__ import annotations

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
    t_c = float(np.clip(k_to_c(T_kelvin), 0.5, 100.0))
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
# 6. PROPIEDADES COLIGATIVAS: °BRIX, ACTIVIDAD DE AGUA, PUNTO DE EBULLICIÓN
# ---------------------------------------------------------------------------
#
# Las propiedades coligativas dependen del NÚMERO de partículas de soluto
# disueltas, no de su identidad química. Las tres funciones de este bloque
# son manifestaciones distintas del mismo principio (Ley de Raoult), aplicado
# a un alimento real en vez de a una solución ideal de laboratorio.


def brix_from_masses(solids_kg: float, water_kg: float) -> float:
    """
    Grados Brix: gramos de sólidos solubles por 100 g de solución.

        °Bx = solidos / (solidos + agua) × 100

    Un refractómetro mide esto indirectamente (índice de refracción), pero
    la escala se calibra precisamente contra esta definición gravimétrica.
    """
    total = max(solids_kg + water_kg, 1e-9)
    return float(np.clip(100.0 * solids_kg / total, 0.0, 100.0))


def molality(solids_kg: float, water_kg: float, M_solute: float) -> float:
    """
    Molalidad del soluto: mol de soluto por kg de DISOLVENTE (no de solución).

    Es la unidad de concentración correcta para propiedades coligativas,
    porque estas dependen de la relación soluto/disolvente, no del volumen
    total (que cambia con la temperatura).

        m = (solidos_kg · 1000 / M_solute) / agua_kg      [mol/kg]
    """
    water_kg = max(water_kg, 1e-9)
    mol_solute = (solids_kg * 1000.0) / max(M_solute, 1e-9)
    return float(mol_solute / water_kg)


def antoine_boiling_point_C(P_kPa: float) -> float:
    """
    Punto de ebullición del AGUA PURA a una presión dada, invirtiendo la
    ecuación de Antoine (la misma que usa `water_vapor_pressure_kPa`, pero
    despejada para T en vez de para P):

        log10(P_mmHg) = A - B/(C + T)   =>   T = B / (A - log10(P_mmHg)) - C

    Es la razón física de por qué un evaporador al vacío hierve mucho más
    frío que una olla a presión atmosférica: bajar P baja T de ebullición.
    """
    p_mmhg = max(P_kPa / MMHG_TO_KPA, 1e-6)
    log_p = np.log10(p_mmhg)
    denom = ANTOINE_H2O["A"] - log_p
    if denom <= 1e-9:
        return 200.0     # fuera del rango válido de Antoine; tope de seguridad
    t_c = ANTOINE_H2O["B"] / denom - ANTOINE_H2O["C"]
    return float(np.clip(t_c, -10.0, 200.0))


def boiling_point_elevation_C(m: float, Kb: float, van_t_hoff_i: float = 1.0) -> float:
    """
    Elevación ebulloscópica, una propiedad coligativa clásica de Química I:

        ΔT_b = i · Kb · m

    `i` es el factor de van't Hoff (1 para un no-electrolito como la sacarosa
    o la lactosa, que no se disocian en iones). `Kb` del agua es 0.512
    °C·kg/mol: es la MISMA constante en cualquier alimento, porque la
    propiedad coligativa depende del disolvente (agua), no del soluto.
    """
    return float(van_t_hoff_i * Kb * max(m, 0.0))


def water_activity_raoult(solids_kg: float, water_kg: float, M_solute: float) -> float:
    """
    Actividad de agua a_w por la Ley de Raoult para una solución ideal:

        a_w = x_agua = mol_agua / (mol_agua + mol_soluto)

    a_w es la humedad relativa de equilibrio del alimento y es EL indicador
    que usa la industria para predecir si un microorganismo puede crecer
    (la mayoría de bacterias necesitan a_w > 0.90-0.95; los mohos resisten
    hasta ~0.70). Concentrar un jugo baja su a_w precisamente porque reduce
    la fracción molar de agua disponible, no porque "elimine" microbios.
    """
    mol_water = max(water_kg, 0.0) * 1000.0 / 18.015
    mol_solute = max(solids_kg, 0.0) * 1000.0 / max(M_solute, 1e-9)
    total = mol_water + mol_solute
    if total <= 0:
        return 1.0
    return float(np.clip(mol_water / total, 0.0, 1.0))


def latent_heat_vaporization_kJkg(T_celsius: float) -> float:
    """
    Calor latente de vaporización del agua, correlación lineal de Watson
    (válida 0-200 °C, error < 1 % frente a tablas de vapor):

        λ(T) ≈ 2500 - 2.36·T_°C     [kJ/kg]

    A 100 °C da ≈2264 kJ/kg (tabla: 2257); a 54 °C da ≈2373 kJ/kg (tabla:
    ≈2373). Es la energía que hay que entregarle al agua para evaporarla SIN
    subir su temperatura: por eso un evaporador que hierve consume mucho
    calor sin que el termómetro se mueva.
    """
    return float(2500.0 - 2.36 * T_celsius)


# ---------------------------------------------------------------------------
# 7. VISCOSIDAD: ARRHENIUS CON EL SIGNO INVERTIDO
# ---------------------------------------------------------------------------


def viscosity_cP(T_kelvin: float, brix: float, pH: float, profile: ProcessProfile) -> float:
    """
    Viscosidad dinámica, en centipoise (cP; el agua a 20 °C tiene 1.0 cP).

    Usa la ECUACIÓN DE ANDRADE, que tiene la misma forma matemática que
    Arrhenius pero con el signo del exponente invertido:

        η(T) = η_A · exp(+Eη / (R·T))

    Una reacción química se ACELERA al calentar porque más moléculas superan
    la barrera de activación. Un líquido FLUYE MEJOR al calentar por la misma
    razón física —las moléculas necesitan superar una barrera para moverse
    unas respecto a otras— pero aquí "reaccionar" es "dejar de fluir", así
    que el efecto observable es el opuesto: subir T hace bajar η.

    A esa base se le multiplican dos efectos, ninguno de los cuales es
    Arrhenius pero ambos son igual de reales:

    - Sólidos disueltos (°Brix): más azúcar disuelto = más fricción interna.
      exp(k_brix · °Bx) crece rápido: a 65 °Bx la viscosidad de un jugo puede
      ser 100-300 veces la del jugo fresco.

    - Gelificación por pH (sólo en fermentación): al cruzar el pH isoeléctrico
      de la caseína, las proteínas de la leche se agregan y la viscosidad se
      dispara en un rango de pH muy estrecho. Se modela con una sigmoide
      (función logística) centrada en `profile.gel_pH`, NO con Arrhenius: es
      un fenómeno de agregación coloidal, un capítulo distinto de la química.
    """
    T_kelvin = max(T_kelvin, 1.0)
    eta_base = profile.eta_A * np.exp(profile.Ea_eta / (R_GAS * T_kelvin))
    eta_base *= np.exp(profile.k_brix * max(brix, 0.0))

    if profile.gel_max_factor > 0:
        # Sigmoide: factor ≈ 1 lejos del punto de gel, y ≈ 1+gel_max_factor
        # una vez que el pH cae por debajo de gel_pH.
        exponent = float(np.clip(profile.gel_steepness * (pH - profile.gel_pH), -50, 50))
        gel_factor = 1.0 + profile.gel_max_factor / (1.0 + np.exp(exponent))
    else:
        gel_factor = 1.0

    eta_pas = eta_base * gel_factor          # Pa·s
    return float(eta_pas * 1000.0)           # 1 Pa·s = 1000 cP


# ---------------------------------------------------------------------------
# 8. LETALIDAD ACUMULADA (valor F)
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
# 9. BALANCE DE MATERIA Y ENERGÍA: SISTEMA DE EDOs
# ---------------------------------------------------------------------------
#
# Vector de estado y = [C_A, C_B, C_HA, T, N, F, W]
#
#   dC_A/dt  = -(r_main + r_acid)                         balance de materia A
#   dC_B/dt  = +r_main                                    producto principal
#   dC_HA/dt = +r_acid                                    ácido -> define el pH
#   dT/dt    = (Q_camisa + Q_reacción - Q_pérdidas)/(m·cp)   balance de energía
#   dN/dt    = -k_micro(T) · N                            inactivación térmica
#   dF/dt    = 10^((T-Tref)/z)                            letalidad acumulada
#   dW/dt    = -Q_evaporación / λ(T)                      agua evaporada (sólo
#                                                          en procesos de
#                                                          concentración)
#
# ---------------------------------------------------------------------------

IDX_CA, IDX_CB, IDX_CHA, IDX_T, IDX_N, IDX_F, IDX_W = range(7)


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

    Bifurca en dos regímenes físicos distintos:

    - Reacción + control de temperatura (pasteurización, fermentación): el
      lazo de control mantiene T en un set point y toda la química ocurre
      en fase líquida sin cambio de fase. Es el modelo original del proyecto.

    - Evaporación al vacío (concentración): NO hay lazo de control por error
      de temperatura -- un evaporador industrial simplemente entrega vapor a
      temperatura fija (`heat_source_C`) y dEJA que el producto encuentre su
      propio punto de ebullición. Mientras el producto está más frío que su
      punto de ebullición, el calor lo calienta (sube T). Una vez que llega a
      hervir, TODO el calor neto se usa en cambiar agua líquida a vapor
      (calor latente) y la temperatura deja de subir -- por eso un termómetro
      en una olla hirviendo se queda quieto en 100 °C aunque el fuego siga
      encendido. El punto de ebullición mismo va subiendo lentamente a
      medida que el producto se concentra (elevación ebulloscópica), así que
      T sube un poco, pero por eso, no porque el balance de energía cambie.
    """
    if profile.kind == "concentracion":
        return _derivatives_evaporacion(y, profile)
    return _derivatives_reaccion(y, profile, setpoint_C)


def _derivatives_reaccion(y: np.ndarray, profile: ProcessProfile,
                          setpoint_C: float) -> np.ndarray:
    """Modelo de reacción en fase líquida con control de temperatura (§9)."""
    C_A, C_B, C_HA, T, N, _F, _W = y
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

    # --- Agua: sin evaporación en este régimen --------------------------
    dW = 0.0

    return np.array([dCA, dCB, dCHA, dT, dN, dF, dW], dtype=float)


def _derivatives_evaporacion(y: np.ndarray, profile: ProcessProfile) -> np.ndarray:
    """
    Modelo de evaporador al vacío por lotes (§6, §7 y §9 combinados).

    Simplificación deliberada, propia de un curso de Química I: se asume que
    el producto hierve de forma continua una vez alcanza su punto de
    ebullición, y que TODO el calor neto entregado a partir de ese momento se
    convierte en evaporación (nada se pierde en sobrecalentar el vapor). Es
    la misma idealización de un diagrama de calentamiento con meseta de
    cambio de fase que se enseña para el agua.
    """
    C_A, C_B, C_HA, T, N, _F, W = y
    C_A = max(C_A, 0.0)
    W = max(W, 1e-6)                 # nunca se evapora más agua de la que hay
    S = profile.solids_mass_kg       # los sólidos no se evaporan: constante

    # --- Cinética: degradación térmica de un nutriente sensible (Arrhenius) -
    r_main = reaction_rate(profile.main_reaction, C_A, T)
    dCA = -r_main
    dCB = r_main
    dCHA = 0.0
    dN = 0.0
    dF = 0.0

    # --- Punto de ebullición ACTUAL de la solución (Antoine + colligativa) --
    brix_now = brix_from_masses(S, W)
    m_now = molality(S, W, profile.M_solute)
    T_boil_pure_C = antoine_boiling_point_C(profile.vacuum_kPa)
    dTb_C = boiling_point_elevation_C(m_now, profile.Kb_ebullioscopic)
    T_boil_soln_K = c_to_k(T_boil_pure_C + dTb_C)

    # --- Calor neto entregado por el medio de calentamiento -----------------
    # Fuente de calor a temperatura FIJA (vapor de planta), no un lazo de
    # control por error: así opera un evaporador real, muy distinto del
    # control de un tanque de pasteurización.
    t_source_K = c_to_k(profile.heat_source_C)
    q_in = profile.UA * (t_source_K - T)
    q_loss = profile.UA_loss * (T - c_to_k(profile.ambient_C))
    q_net = q_in - q_loss                       # [W] = [J/s]

    mass_total_kg = max(S + W, 1e-6)

    if T < T_boil_soln_K - 0.3:
        # Aún no hierve: todo el calor neto se va en calentamiento sensible.
        dT = q_net / (mass_total_kg * profile.cp)
        dW = 0.0
    else:
        # Hirviendo: la temperatura persigue (rápido, pero sin discontinuidad
        # brusca) al punto de ebullición, que sube muy lentamente a medida
        # que el producto se concentra.
        dT = 0.08 * (T_boil_soln_K - T)
        lambda_kJkg = latent_heat_vaporization_kJkg(k_to_c(T))
        lambda_Jkg = max(lambda_kJkg, 100.0) * 1000.0
        dW = -max(q_net, 0.0) / lambda_Jkg      # [kg/s]

    return np.array([dCA, dCB, dCHA, dT, dN, dF, dW], dtype=float)


def initial_state(profile: ProcessProfile) -> np.ndarray:
    """Vector de estado en t = 0 a partir del perfil de proceso."""
    return np.array(
        [profile.CA0, 0.0, 0.0, c_to_k(profile.T0_C), profile.N0, 0.0,
         profile.water_mass0_kg],
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
    y_new[IDX_W] = max(y_new[IDX_W], 0.0)
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


def derived_metrics(y: np.ndarray, profile: ProcessProfile) -> dict:
    """
    Calcula todas las propiedades que no son parte del vector de estado pero
    se derivan de él: pH, presión, °Brix, viscosidad, actividad de agua y
    elevación del punto de ebullición.

    Devuelve un diccionario (no una tupla) para que agregar una propiedad más
    en el futuro no rompa a quien ya llama a esta función por posición.
    """
    ph = ph_from_weak_acid(
        y[IDX_CHA], profile.Ka, profile.pH0, profile.buffer_capacity
    )

    W = max(y[IDX_W], 1e-6)
    S = profile.solids_mass_kg
    brix = brix_from_masses(S, W)
    m_now = molality(S, W, profile.M_solute)
    water_activity = water_activity_raoult(S, W, profile.M_solute)
    boiling_elevation = boiling_point_elevation_C(m_now, profile.Kb_ebullioscopic)
    viscosity = viscosity_cP(y[IDX_T], brix, ph, profile)

    if profile.kind == "concentracion":
        # Bajo vacío, el espacio de cabeza está dominado por vapor de agua:
        # la presión ES, en esencia, la presión de vapor a la temperatura
        # actual (por eso sube ligeramente conforme sube el punto de
        # ebullición de la solución).
        pressure = water_vapor_pressure_kPa(y[IDX_T])
    else:
        # Sólo una fracción del producto B se libera como gas al espacio de
        # cabeza (CO2 de la ruta heteroláctica). En pasteurización gas_yield=0.
        gas_mol = y[IDX_CB] * profile.volume_L * profile.gas_yield
        pressure = system_pressure_kPa(
            y[IDX_T], c_to_k(profile.T0_C), gas_mol, profile.headspace_L
        )

    return {
        "pH": ph,
        "pressure_kPa": pressure,
        "brix": brix,
        "viscosity_cP": viscosity,
        "water_activity": water_activity,
        "boiling_point_elevation_C": boiling_elevation,
    }
