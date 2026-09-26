"""
config.py
=========
Fuente única de verdad para constantes fisicoquímicas, parámetros del proceso
y Límites Críticos de Control (LCC / CCP del plan HACCP).

Todo lo que el estudiante de Ing. de Alimentos necesite ajustar vive AQUÍ.
Ningún otro módulo debe contener números "mágicos".

Convención de unidades (SI, salvo donde se indique):
    Temperatura .......... K  (internamente)  /  °C (interfaz de usuario)
    Concentración ........ mol/L
    Tiempo ............... s
    Energía de activación  J/mol
    Entalpía de reacción . J/mol
    Presión .............. kPa
    Volumen .............. L
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Literal

# ---------------------------------------------------------------------------
# 1. CONSTANTES UNIVERSALES
# ---------------------------------------------------------------------------

R_GAS = 8.314462618        # Constante universal de los gases [J/(mol·K)]
KELVIN_0 = 273.15          # Offset °C -> K
P_ATM = 101.325            # Presión atmosférica estándar [kPa]

# Coeficientes de Antoine para el AGUA (válidos 1–100 °C, P en mmHg, T en °C)
# log10(P_mmHg) = A - B / (C + T_celsius)
ANTOINE_H2O = {"A": 8.07131, "B": 1730.63, "C": 233.426}
MMHG_TO_KPA = 0.1333223684


def c_to_k(t_celsius: float) -> float:
    """Convierte grados Celsius a Kelvin."""
    return t_celsius + KELVIN_0


def k_to_c(t_kelvin: float) -> float:
    """Convierte Kelvin a grados Celsius."""
    return t_kelvin - KELVIN_0


# ---------------------------------------------------------------------------
# 2. PARÁMETROS CINÉTICOS
# ---------------------------------------------------------------------------

@dataclass
class KineticParams:
    """
    Parámetros de una reacción que sigue la ley de velocidad:

        r = k(T) · [A]^n            (ley de velocidad, orden n)
        k(T) = A_pre · exp(-Ea / (R·T))    (ecuación de Arrhenius)

    Attributes
    ----------
    name : nombre legible de la reacción
    A_pre : factor pre-exponencial (o factor de frecuencia) [unidades según orden]
    Ea : energía de activación [J/mol]
    order : orden global de la reacción respecto a [A]
    delta_H : entalpía de reacción [J/mol]. Negativa => exotérmica.
    """
    name: str
    A_pre: float
    Ea: float
    order: float = 1.0
    delta_H: float = 0.0


def arrhenius_from_D_z(D_ref_s: float, T_ref_C: float, z_C: float) -> KineticParams:
    """
    Construye parámetros de Arrhenius a partir de los valores D y z que usa
    la microbiología predictiva de alimentos. Este puente es el que conecta
    Química I con Ingeniería de Alimentos.

    Relaciones:
        k_ref = ln(10) / D_ref                     (destrucción térmica de orden 1)
        Ea    = 2.303 · R · T_ref · (T_ref + z) / z
        A_pre = k_ref · exp(Ea / (R · T_ref))

    Parameters
    ----------
    D_ref_s : tiempo de reducción decimal a T_ref [s]
    T_ref_C : temperatura de referencia [°C]
    z_C : incremento de T que reduce D en un factor de 10 [°C]
    """
    T_ref = c_to_k(T_ref_C)
    k_ref = 2.302585092994046 / D_ref_s
    Ea = 2.302585092994046 * R_GAS * T_ref * (T_ref + z_C) / z_C
    A_pre = k_ref * pow(2.718281828459045, Ea / (R_GAS * T_ref))
    return KineticParams(
        name="Inactivacion microbiana", A_pre=A_pre, Ea=Ea, order=1.0, delta_H=0.0
    )


# ---------------------------------------------------------------------------
# 3. LÍMITES CRÍTICOS DE CONTROL (HACCP)
# ---------------------------------------------------------------------------

@dataclass
class CriticalLimit:
    """
    Límite crítico de un Punto Crítico de Control (PCC).

    Se definen dos bandas:
      - [warn_low, warn_high]  -> banda de operación deseada
      - [crit_low, crit_high]  -> fuera de aquí el lote se considera NO INOCUO
    """
    variable: str
    unit: str
    warn_low: float
    warn_high: float
    crit_low: float
    crit_high: float
    ccp_id: str = ""
    rationale: str = ""


# ---------------------------------------------------------------------------
# 4. PERFILES DE PROCESO
# ---------------------------------------------------------------------------

ProcessKind = Literal["pasteurizacion", "fermentacion", "concentracion"]


@dataclass
class ProcessProfile:
    """Configuración completa de un proceso simulable por el gemelo digital."""

    kind: ProcessKind
    label: str

    # --- Geometría y propiedades del lote -----------------------------------
    volume_L: float                 # Volumen de producto [L]
    mass_kg: float                  # Masa de producto [kg]
    cp: float                       # Calor específico [J/(kg·K)]
    UA: float                       # Coef. global × área de transferencia [W/K]
    ambient_C: float                # Temperatura ambiente [°C]
    UA_loss: float                  # Pérdidas al ambiente [W/K]
    headspace_L: float              # Volumen del espacio de cabeza [L]
    gas_yield: float                # mol de gas por mol de producto B formado

    # --- Condiciones iniciales ----------------------------------------------
    T0_C: float                     # Temperatura inicial del producto [°C]
    CA0: float                      # Concentración inicial del reactivo A [mol/L]
    pH0: float                      # pH inicial
    N0: float                       # Carga microbiana inicial [UFC/mL]

    # --- Cinéticas ----------------------------------------------------------
    main_reaction: KineticParams    # A -> B (reacción de interés)
    acid_reaction: KineticParams    # A -> HA (acidificación, define el pH)
    microbial: KineticParams        # N -> inactivación térmica

    # --- Química ácido-base -------------------------------------------------
    Ka: float                       # Constante de acidez del ácido dominante
    buffer_capacity: float          # Capacidad tampón β [mol/(L·unidad de pH)]
    stoich_acid: float              # mol de HA por mol de A consumido en esa vía

    # --- Composición: sólidos solubles (grados Brix) -------------------------
    solids_mass_kg: float           # Masa de sólidos solubles disueltos [kg]
    water_mass0_kg: float           # Masa de agua inicial [kg]
    M_solute: float                 # Masa molar del soluto dominante [g/mol]

    # --- Control ------------------------------------------------------------
    T_setpoint_C: float             # Set point de temperatura [°C]
    Kp: float                       # Ganancia proporcional del lazo de camisa
    jacket_min_C: float
    jacket_max_C: float

    # --- Letalidad ----------------------------------------------------------
    T_ref_leth_C: float             # T de referencia para F [°C]
    z_leth_C: float                 # valor z para F [°C]
    F_target_s: float               # Letalidad acumulada objetivo [s]

    # --- Sensores IoT (ruido gaussiano) -------------------------------------
    noise_sigma: Dict[str, float] = field(default_factory=dict)

    # --- PCC ----------------------------------------------------------------
    limits: Dict[str, CriticalLimit] = field(default_factory=dict)

    # --- Propiedades coligativas: constante ebulloscópica --------------------
    Kb_ebullioscopic: float = 0.512  # Constante ebulloscópica del agua [°C·kg/mol]

    # --- Viscosidad: Arrhenius/Andrade + efecto de sólidos y de gel ----------
    eta_A: float = 1.0e-3           # Factor pre-exponencial de viscosidad [Pa·s]
    Ea_eta: float = 15_000.0        # "Energía de activación" para flujo viscoso [J/mol]
    k_brix: float = 0.0             # Sensibilidad de la viscosidad al °Brix [1/°Bx]
    gel_pH: float = 0.0             # pH de gelificación (cuajada), 0 = sin efecto
    gel_steepness: float = 10.0     # Qué tan abrupta es la transición al gelificar
    gel_max_factor: float = 0.0     # Multiplicador máximo de viscosidad por gelificación

    # --- Evaporación al vacío (sólo procesos de concentración) ---------------
    vacuum_kPa: float = P_ATM       # Presión absoluta de operación [kPa]
    heat_source_C: float = 0.0      # Temperatura del medio de calentamiento [°C]
                                     # (0 = usar el lazo P estándar; >0 = fuente
                                     #  de calor a temperatura fija, como un
                                     #  evaporador alimentado con vapor de planta)

    def to_dict(self) -> dict:
        return asdict(self)


# --- 4.1 Pasteurización HTST de leche (72 °C / 15 s) ------------------------

PASTEURIZACION = ProcessProfile(
    kind="pasteurizacion",
    label="Pasteurización HTST de leche entera",
    volume_L=50.0,
    mass_kg=51.5,
    cp=3930.0,
    UA=1020.0,
    ambient_C=22.0,
    UA_loss=35.0,
    headspace_L=8.0,
    gas_yield=0.0,                  # proceso sin generación de gas
    T0_C=8.0,
    CA0=0.140,                      # lactosa aprovechable aproximada [mol/L]
    pH0=6.70,
    N0=1.0e6,                       # UFC/mL de carga inicial
    main_reaction=KineticParams(
        name="Degradacion termica de vitamina B1 (A -> B)",
        A_pre=1.36e11, Ea=100_000.0, order=1.0, delta_H=-12_000.0,
    ),
    acid_reaction=KineticParams(
        name="Acidificacion residual (A -> HA)",
        A_pre=4.0e8, Ea=95_000.0, order=1.0, delta_H=-8_000.0,
    ),
    microbial=arrhenius_from_D_z(D_ref_s=15.0, T_ref_C=72.0, z_C=6.5),
    Ka=1.38e-4,                     # ácido láctico
    buffer_capacity=0.050,          # ver nota de calibración al final del archivo
    stoich_acid=4.0,                # 1 lactosa -> 4 ácido láctico
    solids_mass_kg=4.635,           # ≈9.0 °Bx (sólidos solubles típicos de leche entera)
    water_mass0_kg=46.865,
    M_solute=342.3,                 # lactosa, C12H22O11 (misma masa molar que la sacarosa)
    eta_A=7.78e-6, Ea_eta=13_500.0,  # viscosidad de la leche (~3 cP fría, ~1 cP a 72 °C)
    k_brix=0.02,                    # a este Brix casi no aporta espesor
    T_setpoint_C=72.0,
    Kp=20.0,
    jacket_min_C=4.0,
    jacket_max_C=95.0,
    T_ref_leth_C=72.0,
    z_leth_C=6.5,
    F_target_s=15.0,
    noise_sigma={
        "temperature_C": 0.12,      # RTD Pt100 clase A
        "pH": 0.015,                # electrodo de vidrio
        "pressure_kPa": 0.8,        # transmisor piezorresistivo
        "concentration_A": 0.0008,  # refractómetro en línea
    },
    limits={
        "temperature_C": CriticalLimit(
            variable="temperature_C", unit="°C",
            warn_low=71.7, warn_high=75.0, crit_low=71.0, crit_high=78.0,
            ccp_id="PCC-1",
            rationale="Debajo de 71.7 °C no se garantiza la reduccion 5-log de "
                      "Coxiella burnetii; arriba de 78 °C hay desnaturalizacion "
                      "de proteinas del suero y sabor a cocido.",
        ),
        "pH": CriticalLimit(
            variable="pH", unit="-",
            warn_low=6.55, warn_high=6.85, crit_low=6.40, crit_high=6.95,
            ccp_id="PCC-2",
            rationale="pH < 6.4 indica acidificacion por flora lactica previa; "
                      "la leche coagula al calentarse.",
        ),
        "pressure_kPa": CriticalLimit(
            variable="pressure_kPa", unit="kPa",
            warn_low=95.0, warn_high=220.0, crit_low=80.0, crit_high=260.0,
            ccp_id="PCC-3",
            rationale="La presion del lado producto debe superar la del medio de "
                      "calentamiento para evitar contaminacion cruzada por fuga.",
        ),
    },
)

# --- 4.2 Fermentación láctica (yogur, 43 °C) -------------------------------

FERMENTACION = ProcessProfile(
    kind="fermentacion",
    label="Fermentación láctica de yogur (S. thermophilus / L. bulgaricus)",
    volume_L=200.0,
    mass_kg=206.0,
    cp=3800.0,
    UA=2400.0,
    ambient_C=22.0,
    UA_loss=90.0,
    headspace_L=30.0,
    gas_yield=0.02,                 # fermentación homoláctica: poco CO2
    T0_C=43.0,
    CA0=0.135,
    pH0=6.60,
    N0=1.0e7,
    main_reaction=KineticParams(
        name="Consumo de lactosa (A -> B)",
        A_pre=6.3e4, Ea=62_000.0, order=1.0, delta_H=-84_000.0,
    ),
    acid_reaction=KineticParams(
        name="Produccion de acido lactico (A -> HA)",
        A_pre=2.2e5, Ea=62_000.0, order=1.0, delta_H=-75_000.0,
    ),
    # En fermentación la "inactivación" se apaga: Ea alta y A bajo.
    microbial=KineticParams(name="Sin letalidad termica", A_pre=1.0, Ea=250_000.0),
    Ka=1.38e-4,
    buffer_capacity=0.050,
    stoich_acid=4.0,                # C12H22O11 + H2O -> 4 CH3-CHOH-COOH
    solids_mass_kg=19.57,           # ≈9.5 °Bx, similar a la leche de partida
    water_mass0_kg=186.43,
    M_solute=342.3,
    eta_A=8.76e-6, Ea_eta=13_500.0,  # misma base líquida que la leche...
    k_brix=0.02,                    # ...el Brix casi no cambia en la fermentación
    gel_pH=4.65, gel_steepness=9.0, gel_max_factor=250.0,  # ...pero SÍ gelifica
    T_setpoint_C=43.0,
    Kp=4.0,
    jacket_min_C=10.0,
    jacket_max_C=60.0,
    T_ref_leth_C=43.0,
    z_leth_C=6.5,
    F_target_s=1.0e9,               # no aplica letalidad
    noise_sigma={
        "temperature_C": 0.08,
        "pH": 0.012,
        "pressure_kPa": 0.5,
        "concentration_A": 0.0010,
    },
    limits={
        "temperature_C": CriticalLimit(
            variable="temperature_C", unit="°C",
            warn_low=41.0, warn_high=45.0, crit_low=38.0, crit_high=48.0,
            ccp_id="PCC-1",
            rationale="Fuera de 41–45 °C se pierde el equilibrio simbiotico entre "
                      "las dos cepas y aparecen defectos de textura.",
        ),
        "pH": CriticalLimit(
            variable="pH", unit="-",
            warn_low=4.55, warn_high=6.70, crit_low=4.20, crit_high=6.80,
            ccp_id="PCC-2",
            rationale="El corte de fermentacion se hace a pH 4.6 (punto isoelectrico "
                      "de la caseina). Por debajo de 4.2 el producto sinéresis.",
        ),
        "pressure_kPa": CriticalLimit(
            variable="pressure_kPa", unit="kPa",
            warn_low=95.0, warn_high=140.0, crit_low=85.0, crit_high=180.0,
            ccp_id="PCC-3",
            rationale="Sobrepresion en el tanque indica fermentacion heterolactica "
                      "con produccion de CO2 (contaminacion por levaduras).",
        ),
    },
)

PROFILES: Dict[str, ProcessProfile] = {
    "pasteurizacion": PASTEURIZACION,
    "fermentacion": FERMENTACION,
}

# --- 4.3 Concentración de jugo de naranja por evaporación al vacío ---------
#
# Este es el proceso donde el °Brix y la viscosidad dejan de ser un adorno y
# se vuelven la variable de control: el objetivo del proceso ES subir el
# Brix, y saber cuándo detenerse depende de la viscosidad (pasado cierto
# punto, el producto ya no se puede bombear).
#
# También es el proceso que mejor ilustra POR QUÉ existe el vacío en
# Química I: bajar la presión baja el punto de ebullición (ecuación de
# Antoine), lo que permite evaporar agua a ~54 °C en vez de 100 °C y así
# evitar que la vitamina C y los aromas se destruyan por Arrhenius.

CONCENTRACION = ProcessProfile(
    kind="concentracion",
    label="Concentración de jugo de naranja por evaporación al vacío",
    volume_L=480.0,
    mass_kg=500.0,                  # sólidos + agua, ver más abajo
    cp=3900.0,
    UA=2600.0,                      # intercambiador del evaporador (mayor área que un tanque)
    ambient_C=22.0,
    UA_loss=25.0,                   # buen aislamiento en un evaporador al vacío
    headspace_L=60.0,
    gas_yield=0.0,                  # no hay fermentación aquí, no se genera CO2
    T0_C=22.0,                      # jugo recién extraído, a temperatura ambiente
    CA0=0.00256,                    # vitamina C inicial ≈ 45 mg/100 mL (176.12 g/mol)
    pH0=3.70,                       # pH típico del jugo de naranja (ácido cítrico)
    N0=1.0e5,                       # no es el foco de este proceso, valor de referencia
    main_reaction=KineticParams(
        name="Degradacion termica de vitamina C (A -> B)",
        A_pre=1.2e4, Ea=55_000.0, order=1.0, delta_H=-9_000.0,
    ),
    # Sin acidificación ni letalidad microbiana: no son el objeto de este PCC.
    acid_reaction=KineticParams(name="Inactiva en este proceso", A_pre=0.0, Ea=1.0),
    microbial=KineticParams(name="Inactiva en este proceso", A_pre=0.0, Ea=1.0),
    Ka=7.4e-4,                      # ácido cítrico (Ka1, referencia)
    buffer_capacity=0.030,
    stoich_acid=0.0,
    solids_mass_kg=59.0,            # 500 kg × 11.8 °Bx (jugo fresco de naranja)
    water_mass0_kg=441.0,
    M_solute=342.3,                 # sacarosa-equivalente: así se define el °Brix
    eta_A=7.15e-7, Ea_eta=16_000.0,  # viscosidad base de un líquido acuoso
    k_brix=0.111,                   # a 65 °Bx esto multiplica la viscosidad ~195×
    vacuum_kPa=15.0,                # bajo vacío, el agua hierve a ≈54 °C, no a 100 °C
    heat_source_C=90.0,             # vapor de planta que calienta la camisa del evaporador
    T_setpoint_C=54.0,              # referencia inicial; el punto real lo fija Antoine + ΔTb
    Kp=0.0,                         # no se usa lazo P aquí, ver heat_source_C
    jacket_min_C=20.0,
    jacket_max_C=90.0,
    T_ref_leth_C=54.0,
    z_leth_C=6.5,
    F_target_s=1.0e9,               # no aplica: este proceso no es un tratamiento térmico letal
    noise_sigma={
        "temperature_C": 0.15,
        "pH": 0.02,
        "pressure_kPa": 0.3,        # transmisor de vacío, buena resolución
        "concentration_A": 0.00004,
        "brix": 0.10,               # refractómetro en línea, típico ±0.1 °Bx
        "viscosity_cP": 0.25,       # viscosímetro en línea, ruido absoluto pequeño
    },
    limits={
        "brix": CriticalLimit(
            variable="brix", unit="°Bx",
            warn_low=0.0, warn_high=67.0, crit_low=0.0, crit_high=70.0,
            ccp_id="PCC-1",
            rationale="El Brix SUBE durante todo el lote por diseño (no se "
                      "alarma por estar bajo, sólo por excederse): por encima "
                      "de 70 °Bx hay riesgo de cristalizacion de azucares en "
                      "el tanque de almacenamiento. Por debajo de 60 °Bx al "
                      "final del lote, la actividad de agua queda demasiado "
                      "alta para la estabilidad sin refrigeracion (ver metrica "
                      "de a_w, que sí se reporta de forma continua).",
        ),
        "viscosity_cP": CriticalLimit(
            variable="viscosity_cP", unit="cP",
            warn_low=0.0, warn_high=300.0, crit_low=0.0, crit_high=450.0,
            ccp_id="PCC-2",
            rationale="Sobre ~450 cP el concentrado deja de ser bombeable con "
                      "los equipos estandar de la planta: riesgo de cavitacion "
                      "en la bomba y taponamiento de la tuberia de descarga.",
        ),
        "temperature_C": CriticalLimit(
            variable="temperature_C", unit="°C",
            warn_low=48.0, warn_high=60.0, crit_low=40.0, crit_high=75.0,
            ccp_id="PCC-3",
            rationale="Temperatura muy por encima de la de ebullicion esperada "
                      "bajo vacio indica perdida de vacio: el producto se acerca "
                      "a condiciones atmosfericas y se degrada la vitamina C y "
                      "el aroma (Arrhenius) mucho mas rapido de lo previsto.",
        ),
    },
)

PROFILES["concentracion"] = CONCENTRACION

# ---------------------------------------------------------------------------
# NOTA DE CALIBRACIÓN (justificación para la sustentación)
# ---------------------------------------------------------------------------
# `buffer_capacity = 0.050 mol/(L·pH)` es el valor medido para la leche: las
# caseínas, los fosfatos y los citratos absorben protones y frenan la caída de
# pH. Se comprueba así: un yogur terminado contiene ~0.9 % p/v de ácido láctico
# (0.10 mol/L) y su pH real es 4.4, es decir 2.2 unidades por debajo del inicial.
#   β = 0.10 mol/L ÷ 2.2 unidades ≈ 0.045–0.050 mol/(L·pH)
# Si se usara sólo el equilibrio del ácido débil aislado saldría pH 2.4, que es
# justamente el error que comete un modelo que ignora el efecto tampón.
#
# `stoich_acid = 4` es la estequiometría de la vía homoláctica: una molécula de
# lactosa se hidroliza a glucosa + galactosa y cada hexosa rinde 2 lactatos.
#
# Los factores pre-exponenciales de fermentación se ajustaron para que la
# lactosa se consuma en ~5 h y el pH llegue a 4.6 (punto de corte industrial)
# alrededor de las 4.5 h, que es lo que reporta la literatura de yogur.
#
# --- Brix, viscosidad y evaporación al vacío --------------------------------
#
# `solids_mass_kg` / `water_mass0_kg` fijan el °Brix inicial de cada producto:
# °Brix = solids / (solids + water) × 100. Para la leche y el yogur se usó
# ~9-9.5 °Bx (sólidos solubles típicos de leche entera); para el jugo de
# naranja fresco, 11.8 °Bx, el valor de referencia de la industria citrícola,
# con meta de concentrado a 65 °Bx (el estándar "FCOJ 65 °Brix" del mercado).
#
# `M_solute = 342.3 g/mol` se usa para los TRES procesos porque así se
# CALIBRA el grado Brix: un refractómetro reporta "gramos de sacarosa
# equivalente por 100 g de solución", y la sacarosa y la lactosa comparten
# casualmente la misma masa molar. Es una simplificación deliberada de
# Química I: el Brix real de leche/yogur no es sacarosa, pero la escala se
# define así por convención y el error que introduce es pequeño.
#
# `eta_A` y `Ea_eta` en viscosity_arrhenius() usan la MISMA forma matemática
# que k(T) de Arrhenius, pero con el signo contrario a propósito: una reacción
# se ACELERA al calentar (k sube con T), mientras un líquido FLUYE MEJOR al
# calentar (la viscosidad BAJA con T). Es el mismo fenómeno físico —moléculas
# necesitan superar una barrera de energía para moverse— aplicado a dos
# procesos opuestos, y es exactamente el tipo de comparación que un profesor
# de Química I espera ver explicada, no solo calculada.
#
# `k_brix` en el yogur es bajo (0.02): el Brix casi no cambia en la
# fermentación (la lactosa se convierte en ácido láctico, no se evapora
# nada), así que el engrosamiento del yogur NO viene del Brix. Viene de
# `gel_max_factor`: al bajar el pH hacia el punto isoeléctrico de la
# caseína (~4.6), las micelas de proteína se agregan y la viscosidad se
# dispara. Por eso la industria monitorea el CORTE de fermentación por pH,
# no por Brix — el modelo reproduce esa asimetría a propósito.
#
# En el jugo de naranja, en cambio, `k_brix = 0.085` es alto: casi toda la
# viscosidad final viene de la concentración de azúcares (Brix), no hay
# gelificación (`gel_max_factor = 0`), y por eso ese proceso sí usa el Brix
# como variable de control principal.
#
# `vacuum_kPa = 15` produce, por la ecuación de Antoine invertida, un punto
# de ebullición del agua de ≈54 °C (en vez de 100 °C a presión atmosférica).
# Es la razón de ser del vacío en la industria de jugos: a menor temperatura,
# la velocidad de degradación de la vitamina C por Arrhenius cae varios
# órdenes de magnitud, preservando nutrientes y aroma.

# Parámetros globales del bucle de simulación
SIM_TICK_SECONDS = 1.0      # cada cuánto se publica una lectura al dashboard
SIM_SPEED_DEFAULT = 60.0    # factor de aceleración: 1 s real = 60 s de proceso
HISTORY_MAXLEN = 1800       # muestras retenidas en memoria (30 min de gráfica)
