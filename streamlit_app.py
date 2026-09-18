"""
streamlit_app.py
================
RUTA RÁPIDA. Misma física, cero JavaScript. Útil para la sustentación si el
profesor quiere ver el modelo funcionando en dos minutos, o como plan B si
falla la red durante la presentación.

Ejecutar desde la raíz del proyecto:
    streamlit run streamlit_app.py

Requiere Streamlit >= 1.33 por el uso de `st.fragment(run_every=...)`, que
refresca sólo el bloque de gráficas en lugar de recargar toda la página.
"""

import sys
from pathlib import Path

# Permite importar los módulos del motor desde backend/ sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import PROFILES
from simulator import DigitalTwin

st.set_page_config(page_title="Gemelo digital de proceso", layout="wide")


# ---------------------------------------------------------------------------
# Estado de sesión: un único gemelo por pestaña del navegador
# ---------------------------------------------------------------------------

if "twin" not in st.session_state:
    st.session_state.twin = DigitalTwin("pasteurizacion")
    st.session_state.twin.start()

twin: DigitalTwin = st.session_state.twin


# ---------------------------------------------------------------------------
# Barra lateral: controles interactivos
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Consigna de operación")

    proc = st.selectbox(
        "Proceso",
        options=list(PROFILES.keys()),
        format_func=lambda k: PROFILES[k].label,
        index=list(PROFILES.keys()).index(twin.profile.kind),
    )
    if proc != twin.profile.kind:
        twin.reset(proc)
        st.rerun()

    t_sp = st.slider("Temperatura objetivo (°C)", 4.0, 95.0,
                     float(twin.setpoint_C), 0.5)
    ph0 = st.slider("pH inicial", 3.0, 9.0, float(twin.profile.pH0), 0.05)
    ca0 = st.slider("Concentración inicial [A] (mol/L)", 0.0, 0.5,
                    float(twin.profile.CA0), 0.005)
    speed = st.slider("Aceleración del tiempo (×)", 1, 300, int(twin.speed))

    twin.set_setpoint(temperature_C=t_sp, pH0=ph0, CA0=ca0, speed=float(speed))

    c1, c2 = st.columns(2)
    if c1.button("Iniciar" if not twin.is_running else "Pausar", use_container_width=True):
        twin.stop() if twin.is_running else twin.start()
        st.rerun()
    if c2.button("Nuevo lote", use_container_width=True):
        twin.reset()
        st.rerun()

    st.caption(
        "La temperatura actúa sobre k mediante Arrhenius. "
        "Sube el set point 10 °C y observa cómo se desploma la vida media."
    )


# ---------------------------------------------------------------------------
# Utilidades de presentación
# ---------------------------------------------------------------------------

STATUS_COLOR = {"OK": "#2c6b4a", "WARNING": "#c98f00", "CRITICAL": "#b3241c"}


def history_dataframe(frames) -> pd.DataFrame:
    """Aplana las tramas JSON a un DataFrame para graficar."""
    rows = []
    for f in frames:
        rows.append(
            {
                "t": f["t_process_s"],
                "T": f["readings"]["temperature_C"],
                "T_camisa": f["control"]["jacket_C"],
                "pH": f["readings"]["pH"],
                "P": f["readings"]["pressure_kPa"],
                "CA": f["readings"]["concentration_A"],
                "CB": f["readings"]["concentration_B"],
            }
        )
    return pd.DataFrame(rows)


def build_figure(df: pd.DataFrame, profile) -> go.Figure:
    """Rejilla 2×2: concentración, temperatura, pH y presión contra el tiempo."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Concentración vs tiempo", "Temperatura vs tiempo",
                        "pH vs tiempo", "Presión vs tiempo"),
        vertical_spacing=0.14, horizontal_spacing=0.09,
    )
    fig.add_trace(go.Scatter(x=df.t, y=df.CA, name="[A]", line=dict(color="#1f5f63")), 1, 1)
    fig.add_trace(go.Scatter(x=df.t, y=df.CB, name="[B]", line=dict(color="#7a5a12")), 1, 1)
    fig.add_trace(go.Scatter(x=df.t, y=df["T"], name="T producto",
                             line=dict(color="#2b3134")), 1, 2)
    fig.add_trace(go.Scatter(x=df.t, y=df.T_camisa, name="T camisa",
                             line=dict(color="#7a5a12", dash="dash")), 1, 2)
    fig.add_trace(go.Scatter(x=df.t, y=df.pH, name="pH", line=dict(color="#5f3a5e")), 2, 1)
    fig.add_trace(go.Scatter(x=df.t, y=df.P, name="Presión",
                             line=dict(color="#2b3134")), 2, 2)

    # Líneas de límite crítico sobre cada panel que tenga PCC definido.
    panels = {"temperature_C": (1, 2), "pH": (2, 1), "pressure_kPa": (2, 2)}
    for var, (r, c) in panels.items():
        lim = profile.limits.get(var)
        if not lim:
            continue
        for val in (lim.crit_low, lim.crit_high):
            fig.add_hline(y=val, line=dict(color="#b3241c", width=1, dash="dot"),
                          row=r, col=c)

    fig.update_xaxes(title_text="tiempo de proceso (s)")
    fig.update_layout(height=620, margin=dict(t=48, b=40, l=50, r=20),
                      showlegend=True, legend=dict(orientation="h", y=-0.12))
    return fig


# ---------------------------------------------------------------------------
# Bloque auto-refrescante: se redibuja cada segundo sin recargar la página
# ---------------------------------------------------------------------------

@st.fragment(run_every="1s")
def live_panel() -> None:
    snap = twin.snapshot()
    if "readings" not in snap:
        st.info("Arrancando el simulador…")
        return

    st.subheader(snap["process_label"])
    st.caption(f"{snap['batch_id']} · t = {snap['t_process_s']:.0f} s de proceso")

    r, m, s = snap["readings"], snap["metrics"], snap["state"]
    cols = st.columns(5)
    cols[0].metric("Temperatura", f"{r['temperature_C']:.2f} °C")
    cols[1].metric("pH", f"{r['pH']:.3f}")
    cols[2].metric("Presión", f"{r['pressure_kPa']:.1f} kPa")
    cols[3].metric("Conversión de A", f"{m['conversion']*100:.1f} %")
    cols[4].metric("Reducción log", f"{m['log_reduction']:.2f}")

    status = snap["status"]
    st.markdown(
        f"<div style='padding:8px 12px;border-left:5px solid {STATUS_COLOR[status]};"
        f"background:#f0f0ec'><b>Estado del lote: {status}</b></div>",
        unsafe_allow_html=True,
    )
    for a in snap["alerts"]:
        (st.error if a["level"] == "CRITICAL" else st.warning)(
            f"**{a['ccp_id']}** · {a['message']}\n\n{a['corrective_action']}"
        )

    frames = twin.get_history(300)
    if len(frames) >= 2:
        st.plotly_chart(build_figure(history_dataframe(frames), twin.profile),
                        use_container_width=True)

    with st.expander("Parámetros del modelo en este instante"):
        st.write(
            {
                "k(T) [1/s]": m["k_rate_s"],
                "Vida media t½ [s]": m["half_life_s"],
                "Letalidad F acumulada [s]": s["lethality_F_s"],
                "Recuento microbiano [UFC/mL]": s["microbial_count"],
                "T camisa [°C]": snap["control"]["jacket_C"],
            }
        )

    if twin.deviation_log():
        with st.expander("Bitácora de desviaciones (anexo HACCP)"):
            st.dataframe(pd.DataFrame(twin.deviation_log()), use_container_width=True)


live_panel()
