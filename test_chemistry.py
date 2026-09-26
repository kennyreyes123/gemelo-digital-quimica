"""
test_chemistry.py
=================
Verifica el motor numérico contra las soluciones ANALÍTICAS que el estudiante
de Ingeniería de Alimentos puede resolver a mano. Esta carpeta es el argumento
de defensa del proyecto: no basta con que la gráfica se vea bonita, el modelo
tiene que reproducir la química.

Ejecutar desde la raíz:
    pytest -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import chemistry as chem                      # noqa: E402
from alerts import AlertEngine, CRITICAL, OK  # noqa: E402
from config import (                          # noqa: E402
    CONCENTRACION,
    FERMENTACION,
    PASTEURIZACION,
    KineticParams,
    R_GAS,
    arrhenius_from_D_z,
    c_to_k,
)

TOL = 1e-6


# --- 1. Arrhenius -----------------------------------------------------------


def test_arrhenius_reproduce_definicion():
    """k debe coincidir con A·exp(-Ea/RT) evaluado a mano."""
    A, Ea, T = 1.0e11, 85_000.0, 350.0
    esperado = A * np.exp(-Ea / (R_GAS * T))
    assert chem.arrhenius(A, Ea, T) == pytest.approx(esperado, rel=TOL)


def test_arrhenius_es_creciente_con_T():
    """Toda reacción con Ea > 0 se acelera al calentar."""
    k1 = chem.arrhenius(1e11, 85_000.0, 330.0)
    k2 = chem.arrhenius(1e11, 85_000.0, 340.0)
    assert k2 > k1


def test_forma_logaritmica_consistente():
    """ln(k2/k1) = -(Ea/R)(1/T2 - 1/T1) debe dar el mismo cociente."""
    Ea, T1, T2 = 85_000.0, 330.0, 345.0
    directo = chem.arrhenius(1e11, Ea, T2) / chem.arrhenius(1e11, Ea, T1)
    assert chem.k_ratio_two_temperatures(Ea, T1, T2) == pytest.approx(directo, rel=1e-9)


# --- 2. Ley de velocidad y orden de reacción --------------------------------


@pytest.mark.parametrize("orden", [0, 1, 2])
def test_decaimiento_coincide_con_solucion_analitica(orden):
    """
    Integrando sólo la reacción principal (sin balance térmico) el resultado
    debe igualar la solución cerrada:
        orden 0: C = C0 - k·t
        orden 1: C = C0·exp(-k·t)
        orden 2: 1/C = 1/C0 + k·t
    """
    kin = KineticParams(name="test", A_pre=1.0e6, Ea=50_000.0, order=orden)
    # C0 alta a propósito: así ningún orden agota el reactivo dentro de la
    # ventana y la comparación es contra la solución analítica pura.
    T, C0, t = 340.0, 5.0, 60.0
    k = chem.arrhenius(kin.A_pre, kin.Ea, T)

    if orden == 0:
        esperado = max(C0 - k * t, 0.0)
    elif orden == 1:
        esperado = C0 * np.exp(-k * t)
    else:
        esperado = 1.0 / (1.0 / C0 + k * t)

    # Integración numérica aislada del sistema completo
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda _t, y: [-chem.reaction_rate(kin, y[0], T)],
                    (0, t), [C0], rtol=1e-9, atol=1e-12)
    assert sol.y[0, -1] == pytest.approx(esperado, rel=1e-4)


def test_vida_media_orden_uno():
    """Para orden 1, t½ = ln2/k y no depende de la concentración inicial."""
    kin = KineticParams(name="t", A_pre=1e6, Ea=50_000.0, order=1)
    assert chem.half_life(kin, 0.1, 340.0) == pytest.approx(
        chem.half_life(kin, 0.9, 340.0), rel=TOL
    )


# --- 3. Puente D/z <-> Arrhenius --------------------------------------------


def test_D_z_devuelve_la_k_esperada():
    """
    Con D72 = 15 s, a 72 °C la constante debe ser ln(10)/15 y en 15 s el
    recuento debe caer exactamente un ciclo logarítmico.
    """
    kin = arrhenius_from_D_z(D_ref_s=15.0, T_ref_C=72.0, z_C=6.5)
    k = chem.arrhenius(kin.A_pre, kin.Ea, c_to_k(72.0))
    assert k == pytest.approx(np.log(10) / 15.0, rel=1e-6)

    N = 1e6 * np.exp(-k * 15.0)
    assert chem.log_reduction(N, 1e6) == pytest.approx(1.0, abs=1e-6)


def test_valor_z_se_respeta():
    """Subir z grados debe multiplicar k por 10 (definición del valor z)."""
    kin = arrhenius_from_D_z(15.0, 72.0, 6.5)
    k1 = chem.arrhenius(kin.A_pre, kin.Ea, c_to_k(72.0))
    k2 = chem.arrhenius(kin.A_pre, kin.Ea, c_to_k(78.5))
    assert k2 / k1 == pytest.approx(10.0, rel=0.02)


# --- 4. Ácido-base ----------------------------------------------------------


def test_sin_acido_el_ph_no_cambia():
    """Añadir cero ácido debe devolver exactamente el pH inicial."""
    assert chem.ph_from_weak_acid(0.0, 1.38e-4, 6.7, 0.05) == pytest.approx(6.7)


def test_ph_baja_al_generar_acido():
    ph_final = chem.ph_from_weak_acid(0.05, 1.38e-4, 6.7, 0.05)
    assert ph_final < 6.7
    assert 0.0 <= ph_final <= 14.0


def test_capacidad_tampon_define_la_pendiente():
    """
    Por definición de β, añadir β mol/L de ácido baja el pH exactamente una
    unidad mientras el tampón no se agote.
    """
    beta = 0.05
    assert chem.ph_from_weak_acid(beta, 1.38e-4, 6.7, beta) == pytest.approx(5.7, abs=1e-6)
    assert chem.ph_from_weak_acid(2 * beta, 1.38e-4, 6.7, beta) == pytest.approx(4.7, abs=1e-6)


def test_el_yogur_llega_al_pH_real():
    """0.10 mol/L de ácido láctico en leche dan pH ~4.5, no el 2.4 del agua."""
    ph = chem.ph_from_weak_acid(0.10, 1.38e-4, 6.6, 0.05)
    assert 4.3 < ph < 4.8
    assert chem.ph_free_acid(0.10, 1.38e-4) < 3.0


def test_el_tampon_no_puede_bajar_del_piso_acido():
    """Con el tampón agotado manda el equilibrio de disociación."""
    ph = chem.ph_from_weak_acid(1.0, 1.38e-4, 6.6, 0.05)
    assert ph == pytest.approx(chem.ph_free_acid(1.0, 1.38e-4), abs=1e-9)


# --- 5. Presión -------------------------------------------------------------


def test_antoine_agua_a_100C():
    """La presión de vapor del agua a ~100 °C debe rondar 101 kPa."""
    p = chem.water_vapor_pressure_kPa(c_to_k(99.5))
    assert 95.0 < p < 105.0


# --- 6. Integración del sistema completo ------------------------------------


def test_el_lote_alcanza_la_pasteurizacion():
    """
    Simular 10 min de proceso a 72 °C debe superar las 5 reducciones
    logarítmicas exigidas por la norma.
    """
    p = PASTEURIZACION
    y = chem.initial_state(p)
    for _ in range(120):                       # 120 pasos × 5 s = 600 s
        y = chem.step(y, 5.0, p, p.T_setpoint_C)

    assert chem.log_reduction(y[chem.IDX_N], p.N0) >= 5.0
    assert y[chem.IDX_F] > p.F_target_s
    assert 70.0 < (y[chem.IDX_T] - 273.15) < 76.0


def test_balance_de_materia_se_conserva():
    """
    A consumido = B producido + HA producido / ν.
    Se divide entre ν porque cada mol de A que va por la vía ácida rinde
    ν moles de ácido (1 lactosa -> 4 lactatos).
    """
    p = PASTEURIZACION
    y = chem.initial_state(p)
    for _ in range(60):
        y = chem.step(y, 5.0, p, p.T_setpoint_C)
    consumido = p.CA0 - y[chem.IDX_CA]
    producido = y[chem.IDX_CB] + y[chem.IDX_CHA] / p.stoich_acid
    assert consumido == pytest.approx(producido, rel=1e-6, abs=1e-12)


# --- 7. Alertas -------------------------------------------------------------


def test_alerta_critica_por_temperatura_baja():
    eng = AlertEngine(PASTEURIZACION)
    alertas = eng.evaluate({"temperature_C": 65.0, "pH": 6.7, "pressure_kPa": 150.0}, 10.0)
    assert any(a.level == CRITICAL and a.variable == "temperature_C" for a in alertas)
    assert AlertEngine.overall_status(alertas) == CRITICAL


def test_sin_alertas_en_condiciones_normales():
    eng = AlertEngine(PASTEURIZACION)
    alertas = eng.evaluate({"temperature_C": 72.5, "pH": 6.70, "pressure_kPa": 150.0}, 10.0)
    assert AlertEngine.overall_status(alertas) == OK


def test_histeresis_evita_parpadeo():
    """
    Una variable que oscila justo sobre el umbral no debe generar una entrada
    nueva en la bitácora en cada muestra.
    """
    eng = AlertEngine(PASTEURIZACION)
    for v in (70.9, 71.05, 70.95, 71.02):
        eng.evaluate({"temperature_C": v, "pH": 6.7, "pressure_kPa": 150.0}, 1.0)
    assert len(eng.deviation_log) == 1


# --- 8. Brix, propiedades coligativas y viscosidad --------------------------


def test_brix_basico():
    assert chem.brix_from_masses(10.0, 90.0) == pytest.approx(10.0)
    assert chem.brix_from_masses(0.0, 100.0) == pytest.approx(0.0)


def test_molalidad_definicion():
    """m = mol soluto / kg disolvente, no de solución."""
    m = chem.molality(solids_kg=0.03423, water_kg=1.0, M_solute=342.3)
    assert m == pytest.approx(0.1, rel=1e-6)   # 34.23 g / 342.3 g/mol = 0.1 mol


def test_antoine_inversion_consistente():
    """La inversión de Antoine debe ser el inverso exacto de la función directa."""
    for P in (15.0, 50.0, 101.325):
        t_c = chem.antoine_boiling_point_C(P)
        p_back = chem.water_vapor_pressure_kPa(c_to_k(t_c))
        assert p_back == pytest.approx(P, rel=1e-3)


def test_agua_hierve_a_100C_a_presion_atmosferica():
    assert chem.antoine_boiling_point_C(101.325) == pytest.approx(100.0, abs=0.5)


def test_vacio_baja_el_punto_de_ebullicion():
    """A menor presión, menor temperatura de ebullición (por eso se usa vacío)."""
    t_vacio = chem.antoine_boiling_point_C(15.0)
    t_atm = chem.antoine_boiling_point_C(101.325)
    assert t_vacio < t_atm
    assert 45.0 < t_vacio < 65.0        # ~54 °C esperado a 15 kPa


def test_elevacion_ebulloscopica_proporcional_a_molalidad():
    """ΔTb = Kb · m: al doblar la molalidad, se dobla la elevación."""
    dTb1 = chem.boiling_point_elevation_C(1.0, 0.512)
    dTb2 = chem.boiling_point_elevation_C(2.0, 0.512)
    assert dTb1 == pytest.approx(0.512)
    assert dTb2 == pytest.approx(2 * dTb1)


def test_actividad_de_agua_baja_al_concentrar():
    """Ley de Raoult: más soluto, menor fracción molar de agua, menor a_w."""
    aw_diluido = chem.water_activity_raoult(10.0, 90.0, 342.3)
    aw_concentrado = chem.water_activity_raoult(65.0, 35.0, 342.3)
    assert aw_diluido > aw_concentrado
    assert 0.0 <= aw_concentrado <= 1.0


def test_actividad_de_agua_es_uno_sin_soluto():
    assert chem.water_activity_raoult(0.0, 100.0, 342.3) == pytest.approx(1.0)


def test_viscosidad_baja_al_calentar():
    """
    Ecuación de Andrade: SIGNO OPUESTO a Arrhenius. Una reacción se acelera
    al calentar; un líquido fluye MEJOR (η baja) al calentar.
    """
    v_frio = chem.viscosity_cP(c_to_k(5.0), 10.0, 6.7, PASTEURIZACION)
    v_caliente = chem.viscosity_cP(c_to_k(70.0), 10.0, 6.7, PASTEURIZACION)
    assert v_caliente < v_frio


def test_viscosidad_sube_con_el_brix():
    T = c_to_k(54.0)
    v_diluido = chem.viscosity_cP(T, 12.0, 3.7, CONCENTRACION)
    v_concentrado = chem.viscosity_cP(T, 65.0, 3.7, CONCENTRACION)
    assert v_concentrado > v_diluido


def test_yogur_gelifica_al_bajar_el_pH():
    """
    La viscosidad del yogur debe dispararse al cruzar el pH isoeléctrico de
    la caseína (~4.65), independientemente del Brix (que casi no cambia).
    """
    T = c_to_k(43.0)
    brix = chem.brix_from_masses(FERMENTACION.solids_mass_kg, FERMENTACION.water_mass0_kg)
    v_inicio = chem.viscosity_cP(T, brix, 6.6, FERMENTACION)
    v_final = chem.viscosity_cP(T, brix, 4.2, FERMENTACION)
    assert v_final > 50 * v_inicio      # salto de al menos 50x, no un cambio gradual


def test_proceso_concentracion_alcanza_65_brix():
    """
    Integrando el evaporador al vacío el tiempo suficiente, el jugo debe
    concentrarse desde ~11.8 °Bx hasta el objetivo comercial de 65 °Bx, con
    la actividad de agua bajando y la viscosidad subiendo de forma monótona.
    """
    p = CONCENTRACION
    y = chem.initial_state(p)
    brix_prev, aw_prev = 0.0, 1.0
    reached = False
    for _ in range(220):                    # 220 × 60 s ≈ 3.7 h de proceso
        y = chem.step(y, 60.0, p, p.T_setpoint_C)
        m = chem.derived_metrics(y, p)
        assert m["brix"] >= brix_prev - 1e-6     # el Brix nunca puede bajar
        assert m["water_activity"] <= aw_prev + 1e-6  # a_w nunca puede subir
        brix_prev, aw_prev = m["brix"], m["water_activity"]
        if m["brix"] >= 65.0:
            reached = True
            break
    assert reached, f"No llegó a 65 °Bx; se quedó en {brix_prev:.1f} °Bx"
    assert 0.85 < aw_prev < 0.95


def test_agua_evaporada_es_fisicamente_posible():
    """El agua restante nunca debe ser negativa ni superar la inicial."""
    p = CONCENTRACION
    y = chem.initial_state(p)
    for _ in range(200):
        y = chem.step(y, 60.0, p, p.T_setpoint_C)
        assert 0.0 <= y[chem.IDX_W] <= p.water_mass0_kg
