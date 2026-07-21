"""
Modelo Estocastico LGD-PEE: Riesgo de Quiebre Empresarial y Empleo
Simulacion Monte Carlo con difusion de saltos (jump-diffusion).
Calibrado con REEM/INEC 2022-2024: Kaplan-Meier, XGBoost y Buhlmann-Straub.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# CONFIGURACION DE LA PAGINA
# ==========================================
st.set_page_config(
    page_title="Simulacion LGD-PEE | Riesgo de Credito",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# PARAMETROS CALIBRADOS (CONSTANTES)
# ==========================================
# Riesgo anual de quiebre por provincia (XGBoost, AUC=0.7859)
RIESGO_PROVINCIA = {
    "Santo Domingo": 0.189,
    "Guayas": 0.180,
    "Pichincha": 0.171,
    "Azuay": 0.170,
}

# Ajuste sectorial de riesgo (Sector 2 - Minas y canteras - excluido)
MULTIPLICADOR_SECTOR = {
    "1: Agricultura, ganaderia, silvicultura y pesca": 0.94,
    "3: Industrias manufactureras": 0.89,
    "4: Comercio": 1.00,
    "5: Construccion": 0.97,
    "6: Servicios": 0.85,
}

# Ventas medias por sector en miles USD (Tabla 12 del estudio)
VENTAS_MEDIA_SECTOR = {
    "1: Agricultura, ganaderia, silvicultura y pesca": 2_212_848.0,
    "3: Industrias manufactureras": 2_093_575.0,
    "4: Comercio": 1_050_949.0,
    "5: Construccion": 556_417.6,
    "6: Servicios": 543_242.5,
}

# Credibilidad Buhlmann-Straub
VHM = 3553.11
EPV = 52308.76
Z = 0.998
MEDIA_GLOBAL = 6.17

# Estimador de credibilidad R_hat por sector (Tabla 12 del estudio).
# Es la estimacion Buhlmann del empleo medio sectorial, ya ponderada por Z.
R_HAT_SECTOR = {
    "1: Agricultura, ganaderia, silvicultura y pesca": 19.018179,
    "3: Industrias manufactureras": 10.072314,
    "4: Comercio": 4.459824,
    "5: Construccion": 6.022202,
    "6: Servicios": 5.870661,
}

# Anclas disponibles para el termino colectivo de la formula de credibilidad
ANCLA_GLOBAL = "Media global (6.17)"
ANCLA_SECTORIAL = "R_hat sectorial (Tabla 12)"

# Parametros de simulacion
N_ITERACIONES = 10_000
T_TRIMESTRES = 12

# Elasticidad del riesgo respecto al tamano (Ilustraciones 10-14: relacion inversa)
ELASTICIDAD_TAMANO = -0.15

# Volatilidad trimestral del empleo, expresada como fraccion del empleo vigente.
# NOTA METODOLOGICA: EPV=52308.76 es una varianza TRANSVERSAL entre empresas
# heterogeneas dentro de un sector, no una varianza temporal intra-empresa.
# Usarla como sigma de un paseo aleatorio produce sigma ~114 plazas/trimestre,
# lo que destruye por ruido a cualquier empresa de tamano realista. EPV se
# conserva donde corresponde (credibilidad Buhlmann) y la dinamica temporal se
# modela con volatilidad proporcional log-normal, configurable por el usuario.
VOL_TRIMESTRAL_DEFAULT = 0.12


# ==========================================
# MOTOR MONTE CARLO
# ==========================================
def calcular_multiplicador_tamano(ventas: float, ventas_media_sector: float) -> float:
    """
    Ajusta el riesgo segun el tamano relativo de la empresa dentro de su sector.

    El estudio evidencia relacion inversa entre volumen de ventas y probabilidad
    de cierre: las empresas grandes se concentran en zonas de menor p_riesgo.
    Se aplica una elasticidad negativa suave sobre el ratio de ventas, acotado
    entre 0.25x y 4x la media sectorial para evitar extrapolaciones absurdas.
    """
    if ventas_media_sector <= 0:
        return 1.0
    ratio = np.clip(ventas / ventas_media_sector, 0.25, 4.0)
    return float(ratio**ELASTICIDAD_TAMANO)


@st.cache_data
def simular_monte_carlo(
    riesgo_base_anual: float,
    mult_sector: float,
    mult_tamano: float,
    plazas_ini: float,
    p_shock: float,
    mult_stress: float,
    factor_castigo: float,
    vol_trimestral: float,
    ancla_colectiva: float,
    semilla: int,
) -> tuple:
    """
    Ejecuta la simulacion Monte Carlo con difusion de saltos de forma vectorizada.

    La semilla se recibe como argumento explicito para que el cache de Streamlit
    invalide correctamente cuando el usuario pide una nueva realizacion aleatoria.

    Retorna (empleo_hist, supervivencia_hist, vivas_final).
    """
    rng = np.random.default_rng(semilla)

    # a) Riesgo trimestral base a partir del anual ajustado
    h_anual_ajustado = np.clip(riesgo_base_anual * mult_sector * mult_tamano, 0.0, 0.999)
    h_q = 1.0 - (1.0 - h_anual_ajustado) ** (1 / 4)

    # Deriva del proceso log-normal: se corrige por -sigma^2/2 para que la
    # esperanza del multiplicador sea 1 (empleo sin tendencia sistematica).
    deriva = -0.5 * vol_trimestral**2

    # e) Severidad LGD: valor congelado para las empresas que quiebran.
    # El parentesis externo agrupa TODA la combinacion de credibilidad antes
    # de aplicar el factor de castigo. `ancla_colectiva` es el termino colectivo
    # de Buhlmann: media global (6.17) o R_hat del sector (Tabla 12).
    empleo_residual = (Z * plazas_ini + (1 - Z) * ancla_colectiva) * factor_castigo

    # Inicializacion
    empleo_hist = np.zeros((T_TRIMESTRES + 1, N_ITERACIONES), dtype=np.float64)
    empleo_actual = np.full(N_ITERACIONES, float(plazas_ini), dtype=np.float64)
    empleo_hist[0] = empleo_actual

    vivas = np.ones(N_ITERACIONES, dtype=bool)
    supervivencia_hist = np.zeros(T_TRIMESTRES + 1, dtype=np.float64)
    supervivencia_hist[0] = 1.0

    # UNICO BUCLE: sobre los 12 trimestres. Todo lo demas es vectorizado sobre N.
    for t in range(1, T_TRIMESTRES + 1):
        # b) Cisne Negro
        shock_t = rng.random(N_ITERACIONES) < p_shock
        h_dinamico = np.clip(h_q * np.where(shock_t, mult_stress, 1.0), 0.0, 1.0)

        # c) Quiebre (solo puede quebrar quien sigue viva)
        nuevas_quiebras = (rng.random(N_ITERACIONES) < h_dinamico) & vivas
        vivas = vivas & ~nuevas_quiebras

        # d) Fluctuacion del empleo de las supervivientes (log-normal multiplicativa)
        factor = np.exp(rng.normal(deriva, vol_trimestral, N_ITERACIONES))
        empleo_actual = np.where(vivas, empleo_actual * factor, empleo_actual)

        # Caida inmediata del 20% si hubo shock en el trimestre
        empleo_actual = np.where(vivas & shock_t, empleo_actual * 0.80, empleo_actual)

        # Piso de cero plazas
        empleo_actual = np.where(vivas, np.maximum(empleo_actual, 0.0), empleo_actual)

        # e) Severidad para las que quebraron en este trimestre.
        # Las que quebraron antes conservan su residual sin volver a fluctuar.
        empleo_actual = np.where(nuevas_quiebras, empleo_residual, empleo_actual)

        empleo_hist[t] = empleo_actual
        supervivencia_hist[t] = vivas.mean()

    return empleo_hist, supervivencia_hist, vivas


def _var_cvar(muestra: np.ndarray, plazas_ini: float) -> tuple:
    """
    VaR y CVaR al 95% sobre una muestra de empleo final, como perdida de plazas.

    El CVaR se obtiene ordenando y promediando exactamente el peor 5% de casos,
    en lugar de filtrar por <= percentil. Con masa puntual en el valor residual
    de las quebradas, el filtro por percentil capturaria mucho mas del 5%.
    """
    if muestra.size == 0:
        return float("nan"), float("nan"), float("nan")
    p5 = float(np.percentile(muestra, 5))
    n_cola = max(1, int(0.05 * muestra.size))
    peores = np.sort(muestra)[:n_cola]
    return plazas_ini - p5, plazas_ini - float(peores.mean()), p5


def calcular_metricas(
    empleo_hist: np.ndarray,
    supervivencia_hist: np.ndarray,
    vivas_final: np.ndarray,
    plazas_ini: float,
) -> dict:
    """
    Calcula metricas de riesgo en dos vistas complementarias.

    - Incondicional: sobre los 10.000 escenarios. Es la medida relevante para
      riesgo agregado, pero se satura cuando la probabilidad de quiebre supera
      el 5%, porque el percentil 5 cae dentro de la masa de empresas quebradas.
    - Condicional a supervivencia: solo sobre las empresas activas en T=12.
      Informa sobre la volatilidad del negocio en marcha y si discrimina entre
      escenarios de estres.
    """
    empleo_final = empleo_hist[-1]
    var_u, cvar_u, p5_u = _var_cvar(empleo_final, plazas_ini)

    supervivientes = empleo_final[vivas_final]
    var_c, cvar_c, p5_c = _var_cvar(supervivientes, plazas_ini)

    return {
        "tasa_supervivencia": float(supervivencia_hist[-1]),
        "var_incond": var_u,
        "cvar_incond": cvar_u,
        "p5_incond": p5_u,
        "var_cond": var_c,
        "cvar_cond": cvar_c,
        "p5_cond": p5_c,
        "n_supervivientes": int(vivas_final.sum()),
        "empleo_mediano_superv": (
            float(np.median(supervivientes)) if supervivientes.size else float("nan")
        ),
    }


# ==========================================
# VISUALIZACIONES (PLOTLY)
# ==========================================
def graficar_fan_chart(empleo_hist: np.ndarray) -> go.Figure:
    """Fan chart con mediana y banda de confianza del 90% (P5-P95)."""
    p5 = np.percentile(empleo_hist, 5, axis=1)
    p50 = np.percentile(empleo_hist, 50, axis=1)
    p95 = np.percentile(empleo_hist, 95, axis=1)
    x = np.arange(T_TRIMESTRES + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([p95, p5[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 204, 150, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Banda P5-P95",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p50,
            line=dict(color="rgb(0, 204, 150)", width=3),
            name="Mediana (P50)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Proyeccion Estocastica de Empleo (Fan Chart)",
        xaxis_title="Trimestres",
        yaxis_title="Plazas de Empleo",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def graficar_supervivencia(supervivencia_hist: np.ndarray) -> go.Figure:
    """Curva de supervivencia escalonada a lo largo de los 12 trimestres."""
    x = np.arange(T_TRIMESTRES + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=supervivencia_hist,
            mode="lines",
            line_shape="hv",
            line=dict(color="rgb(239, 85, 59)", width=3),
            name="Tasa de Supervivencia",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Curva de Supervivencia Empresarial",
        xaxis_title="Trimestres",
        yaxis_title="Proporcion de Empresas Activas",
        yaxis_range=[0, 1.05],
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def graficar_histograma(empleo_final: np.ndarray, p5_empleo: float) -> go.Figure:
    """Histograma de plazas retenidas en T=12 con el corte del VaR 95%."""
    fig = px.histogram(x=empleo_final, nbins=60, color_discrete_sequence=["#636EFA"])
    fig.add_vline(
        x=p5_empleo,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR 95%: {p5_empleo:.1f} plazas",
        annotation_position="top right",
        annotation_font=dict(color="red"),
    )
    fig.update_layout(
        template="plotly_dark",
        title="Distribucion de Plazas Retenidas en T=12",
        xaxis_title="Plazas de Empleo",
        yaxis_title="Frecuencia (N escenarios)",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ==========================================
# INTERFAZ
# ==========================================
def main():
    st.title("Modelo Estocastico LGD-PEE: Riesgo de Quiebre y Empleo")
    st.markdown(
        "Simulacion de Monte Carlo con difusion de saltos para modelar el riesgo "
        "de quiebre empresarial y su impacto en la generacion de plazas de empleo "
        "en Ecuador (2022-2024)."
    )

    with st.sidebar:
        st.header("Parametros del Modelo")

        provincia_sel = st.selectbox("Provincia", list(RIESGO_PROVINCIA.keys()))
        sector_sel = st.selectbox("Sector Economico", list(MULTIPLICADOR_SECTOR.keys()))

        ventas_anuales = st.number_input(
            "Ventas totales anuales (miles USD)",
            min_value=0,
            value=500_000,
            step=10_000,
        )
        plazas_ini = st.number_input(
            "Plazas de empleo actuales", min_value=1, value=20, step=1
        )

        st.divider()
        st.subheader("Escenarios de Estres")

        p_shock = (
            st.slider(
                "Prob. trimestral de Cisne Negro (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
            )
            / 100.0
        )
        mult_stress = st.slider(
            "Multiplicador de riesgo en shock",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.25,
        )
        factor_castigo = (
            st.slider(
                "Factor de castigo LGD (%)",
                min_value=0,
                max_value=50,
                value=10,
                step=5,
            )
            / 100.0
        )
        ancla_sel = st.radio(
            "Ancla colectiva de la formula LGD",
            [ANCLA_SECTORIAL, ANCLA_GLOBAL],
            help=(
                "Termino colectivo de Buhlmann-Straub. R_hat sectorial reproduce "
                "la logica del estudio (Tabla 12); la media global es el promedio "
                "no sectorizado de la poblacion completa."
            ),
        )
        vol_trimestral = (
            st.slider(
                "Volatilidad trimestral del empleo (%)",
                min_value=2,
                max_value=30,
                value=int(VOL_TRIMESTRAL_DEFAULT * 100),
                step=1,
                help=(
                    "Desviacion estandar log-normal del empleo por trimestre. "
                    "No se deriva de EPV: ver nota metodologica en el codigo."
                ),
            )
            / 100.0
        )

        st.divider()
        usar_semilla = st.checkbox("Fijar semilla (reproducibilidad)", value=True)
        if usar_semilla:
            semilla = int(
                st.number_input("Semilla", min_value=0, value=42, step=1)
            )
        else:
            # Semilla nueva en cada rerun: fuerza la invalidacion del cache
            semilla = int(np.random.default_rng().integers(0, 2**31 - 1))
            st.caption(f"Semilla aleatoria activa: {semilla}")

    # Composicion del riesgo
    riesgo_base = RIESGO_PROVINCIA[provincia_sel]
    mult_sector = MULTIPLICADOR_SECTOR[sector_sel]
    mult_tamano = calcular_multiplicador_tamano(
        ventas_anuales, VENTAS_MEDIA_SECTOR[sector_sel]
    )
    riesgo_efectivo = riesgo_base * mult_sector * mult_tamano
    ancla_colectiva = (
        R_HAT_SECTOR[sector_sel] if ancla_sel == ANCLA_SECTORIAL else MEDIA_GLOBAL
    )

    empleo_hist, supervivencia_hist, vivas_final = simular_monte_carlo(
        riesgo_base_anual=riesgo_base,
        mult_sector=mult_sector,
        mult_tamano=mult_tamano,
        plazas_ini=float(plazas_ini),
        p_shock=p_shock,
        mult_stress=mult_stress,
        factor_castigo=factor_castigo,
        vol_trimestral=vol_trimestral,
        ancla_colectiva=ancla_colectiva,
        semilla=semilla,
    )

    m = calcular_metricas(
        empleo_hist, supervivencia_hist, vivas_final, float(plazas_ini)
    )

    empleo_residual = (
        Z * float(plazas_ini) + (1 - Z) * ancla_colectiva
    ) * factor_castigo

    # Ancla no seleccionada: se calcula para la tabla comparativa del expander,
    # que evidencia el efecto marginal del termino colectivo cuando Z ~ 1.
    ancla_alterna = (
        MEDIA_GLOBAL if ancla_sel == ANCLA_SECTORIAL else R_HAT_SECTOR[sector_sel]
    )
    residual_alterno = (
        Z * float(plazas_ini) + (1 - Z) * ancla_alterna
    ) * factor_castigo

    st.caption(
        f"Riesgo anual efectivo: **{riesgo_efectivo:.2%}** "
        f"(base {riesgo_base:.1%} x sector {mult_sector:.2f} x tamano {mult_tamano:.3f})"
        f"  |  Ancla LGD: {ancla_sel} = {ancla_colectiva:.4f}  ->  "
        f"empleo residual tras quiebre = **{empleo_residual:.2f} plazas**"
    )

    st.subheader("Riesgo incondicional (sobre los 10.000 escenarios)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Tasa de Supervivencia (T=12)",
            value=f"{m['tasa_supervivencia']:.2%}",
            delta=f"{(m['tasa_supervivencia'] - 1.0) * 100:.2f} pp vs inicio",
        )
    with col2:
        st.metric(
            label="VaR 95% (perdida de empleo)",
            value=f"{max(0.0, m['var_incond']):.1f} plazas",
            delta=f"-{max(0.0, m['var_incond']):.1f} plazas",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            label="CVaR 95% (Expected Shortfall)",
            value=f"{max(0.0, m['cvar_incond']):.1f} plazas",
            delta=f"-{max(0.0, m['cvar_incond']):.1f} plazas",
            delta_color="inverse",
        )

    if m["tasa_supervivencia"] < 0.95:
        st.info(
            "Con una probabilidad de quiebre superior al 5%, el percentil 5 cae "
            "dentro de la masa de empresas quebradas: el VaR incondicional se "
            "satura en la perdida total y deja de discriminar entre escenarios. "
            "Las metricas condicionales de abajo si son sensibles a los sliders."
        )

    st.subheader("Riesgo condicional a supervivencia (empresas activas en T=12)")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric(
            label="Escenarios supervivientes",
            value=f"{m['n_supervivientes']:,}",
            delta=f"empleo mediano {m['empleo_mediano_superv']:.1f} plazas",
            delta_color="off",
        )
    with col5:
        st.metric(
            label="VaR 95% | superviviente",
            value=f"{m['var_cond']:.1f} plazas",
            delta=f"P5 = {m['p5_cond']:.1f} plazas",
            delta_color="off",
        )
    with col6:
        st.metric(
            label="CVaR 95% | superviviente",
            value=f"{m['cvar_cond']:.1f} plazas",
            delta_color="off",
        )

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(graficar_fan_chart(empleo_hist), use_container_width=True)
    with col_b:
        st.plotly_chart(
            graficar_supervivencia(supervivencia_hist), use_container_width=True
        )

    st.plotly_chart(
        graficar_histograma(empleo_hist[-1], m["p5_incond"]), use_container_width=True
    )

    delta_abs = abs(empleo_residual - residual_alterno)
    with st.expander("Nota sobre el factor de credibilidad Z"):
        st.markdown(
            f"""
El estudio calibra **Z = {Z}** sobre 700.147 empresas. Con un valor tan proximo
a 1, la formula de credibilidad pondera la experiencia individual al {Z:.1%} y
el termino colectivo solo al {1 - Z:.1%}. En consecuencia, **la eleccion del
ancla apenas altera el resultado**:

| Ancla | Valor | Empleo residual |
|---|---|---|
| {ancla_sel} (activa) | {ancla_colectiva:.4f} | {empleo_residual:.4f} plazas |
| Alternativa | {ancla_alterna:.4f} | {residual_alterno:.4f} plazas |

Diferencia: **{delta_abs:.4f} plazas** ({delta_abs / max(empleo_residual, 1e-9):.2%}).

No es un error de implementacion, sino la consecuencia directa de
K = EPV/VHM = {EPV / VHM:.2f} sobre una poblacion muy grande. El selector se
mantiene por transparencia metodologica y para permitir analisis de sensibilidad.
"""
        )


if __name__ == "__main__":
    main()
