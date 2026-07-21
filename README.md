# Modelo Estocástico LGD-PEE

Simulación de Monte Carlo con difusión de saltos (*jump-diffusion*) para modelar el riesgo de quiebre empresarial y su impacto en la generación de plazas de empleo en Ecuador, periodo 2022–2024.

La aplicación integra tres capas metodológicas:

| Capa | Método | Aporte al modelo |
|---|---|---|
| Supervivencia | Kaplan-Meier | Trayectorias de permanencia empresarial |
| Riesgo predictivo | XGBoost (AUC 0.7859) | Probabilidad de cierre por provincia y sector |
| Severidad | Credibilidad Bühlmann-Straub | Empleo residual tras el quiebre (LGD-PEE) |

---

## Instalación

```bash
git clone https://github.com/<usuario>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La app queda disponible en `http://localhost:8501`.

### Tests

```bash
pytest
```

68 pruebas cubren coherencia analítica, consistencia VaR/CVaR, rendimiento, dominio del empleo, reproducibilidad y monotonía frente al tamaño.

---

## Uso

El panel lateral controla siete parámetros:

| Control | Rango | Efecto |
|---|---|---|
| Provincia | 4 opciones | Riesgo base anual (17.0%–18.9%) |
| Sector económico | 5 opciones | Multiplicador de riesgo (0.85–1.00) |
| Ventas anuales | ≥ 0 kUSD | Ajuste por tamaño (elasticidad −0.15) |
| Plazas iniciales | ≥ 1 | Punto de partida del empleo |
| Prob. Cisne Negro | 0–10% trimestral | Frecuencia de shocks |
| Multiplicador de estrés | 1×–5× | Intensidad del shock sobre el riesgo |
| Factor de castigo LGD | 0–50% | Severidad de la pérdida al quebrar |
| Volatilidad trimestral | 2–30% | Dispersión del empleo en marcha |
| Ancla colectiva LGD | 2 opciones | Término colectivo de Bühlmann |

La salida son tres métricas incondicionales, tres condicionales a supervivencia, y tres gráficos: *fan chart* de empleo, curva de supervivencia e histograma de cola.

---

## Parámetros calibrados

Todos provienen del estudio y **no deben recalcularse en tiempo de ejecución**.

### Riesgo anual por provincia (XGBoost)

| Provincia | p(cierre) | Empresas |
|---|---|---|
| Santo Domingo de los Tsáchilas | 18.9% | 36 216 |
| Guayas | 18.0% | 260 191 |
| Pichincha | 17.1% | 329 121 |
| Azuay | 17.0% | 74 619 |

### Sectores

El **Sector 2 (Explotación de minas y canteras) está excluido**, siguiendo la sección 3.2.1 del estudio: supervivencia baja en 3 de 4 provincias y riesgo medio del 18.1%, el más alto de todos los sectores, lo que produciría estimaciones de credibilidad poco confiables.

| Sector | Multiplicador | R̂ (Tabla 12) | Ventas medias (kUSD) |
|---|---|---|---|
| 1 · Agricultura, ganadería, silvicultura y pesca | 0.94 | 19.018 | 2 212 848.0 |
| 3 · Industrias manufactureras | 0.89 | 10.072 | 2 093 575.0 |
| 4 · Comercio | 1.00 | 4.460 | 1 050 949.0 |
| 5 · Construcción | 0.97 | 6.022 | 556 417.6 |
| 6 · Servicios | 0.85 | 5.871 | 543 242.5 |

### Credibilidad Bühlmann-Straub

```
VHM = 3553.11      varianza de medias hipotéticas (entre sectores)
EPV = 52308.76     varianza esperada del proceso (dentro de sectores)
K   = EPV / VHM = 14.72
Z   = 0.998        factor de credibilidad
X̄   = 6.17         media global de plazas
```

---

## Motor de simulación

**N = 10 000** iteraciones, **T = 12** trimestres. Completamente vectorizado: un único bucle `for` sobre los trimestres, todo lo demás son operaciones NumPy sobre arrays de shape `(N,)`. Tiempo típico: **≈ 5 ms**.

Por cada trimestre *t*:

1. **Riesgo trimestral.** `h_q = 1 − (1 − h_anual)^(1/4)`, donde `h_anual = riesgo_provincia × mult_sector × mult_tamaño`.
2. **Cisne negro.** `shock ~ Bernoulli(p)`. Si ocurre, `h_dinámico = clip(h_q × mult_estrés, 0, 1)`.
3. **Quiebre.** `U(0,1) < h_dinámico` sobre las empresas aún vivas.
4. **Fluctuación del empleo.** Multiplicativa log-normal: `empleo ×= exp(N(−σ²/2, σ))`. Si hubo shock, caída adicional del 20%. Piso en cero.
5. **Severidad LGD.** Las que quiebran congelan su empleo en
   `(Z × plazas_iniciales + (1 − Z) × ancla_colectiva) × factor_castigo`.

---

## Decisiones metodológicas

Tres puntos donde la implementación se aparta de una lectura literal del estudio. Están documentados porque afectan la interpretación de los resultados.

### 1. EPV no se usa como volatilidad temporal

EPV = 52 308.76 es una varianza **transversal** entre empresas heterogéneas dentro de un sector — el propio estudio lo describe así: *"dentro de cada sector existen empresas que generan muchas plazas y otras que generan pocas"*.

Usarla como σ de un paseo aleatorio temporal implica **σ = √(EPV/4) ≈ 114 plazas por trimestre**. Aplicado a una empresa de 20 empleados, el ruido domina por completo el modelo de riesgo: el percentil 90 del empleo final alcanzaba 1 530 plazas partiendo de 20.

La dinámica temporal se modela con **volatilidad proporcional log-normal** (12% trimestral por defecto, ajustable 2–30%), con corrección de deriva −σ²/2 para que la esperanza del multiplicador sea 1. EPV se conserva donde sí corresponde: en la fórmula de credibilidad.

### 2. Doble vista de VaR y CVaR

Con ~18% de riesgo anual, alrededor del 45% de las empresas quiebran en 3 años. El percentil 5 cae entonces dentro de la masa puntual de empresas quebradas, y **el VaR incondicional se satura en la pérdida total**: mover los sliders no lo altera.

Es correcto matemáticamente pero inútil como diagnóstico. La app reporta ambas vistas:

- **Incondicional** — sobre los 10 000 escenarios. Es la medida relevante para riesgo agregado de cartera.
- **Condicional a supervivencia** — solo sobre empresas activas en T=12. Informa sobre la volatilidad del negocio en marcha y sí discrimina entre escenarios.

Verificado: el VaR condicional pasa de 10.6 a 13.5 plazas al subir la probabilidad de shock de 0% a 10%.

### 3. Ajuste de riesgo por tamaño

El estudio evidencia relación inversa entre volumen de ventas y probabilidad de cierre (Ilustraciones 10–14: *"el tamaño empresarial parece estar inversamente relacionado con la probabilidad de riesgo"*), pero no cuantifica una elasticidad.

Se aplica `mult = (ventas / ventas_media_sector)^(−0.15)`, acotado entre 0.25× y 4× la media sectorial. Sin esto, el input de ventas sería decorativo.

### Nota sobre Z = 0.998

Con Z tan próximo a 1, la fórmula de credibilidad pondera la experiencia individual al 99.8% y el término colectivo solo al 0.2%. **La elección del ancla (media global vs. R̂ sectorial) altera el empleo residual en menos del 0.2%.** No es un defecto de implementación sino la consecuencia de K = 14.72 sobre 700 147 empresas. El selector se mantiene por transparencia y para permitir análisis de sensibilidad.

---

## Estructura

```
.
├── app.py                  # Aplicación completa (motor + interfaz)
├── requirements.txt
├── pytest.ini
├── .streamlit/
│   └── config.toml         # Tema oscuro
├── tests/
│   └── test_motor.py       # 68 pruebas
└── docs/
    ├── AI_STUDIO.md        # Proceso de generación en Google AI Studio
    └── PROMPT.md           # Prompt íntegro utilizado
```

---

## Validación

| Criterio | Resultado |
|---|---|
| Supervivencia sin shocks ≈ (1−h)³ | Desviación máx. 0.74 pp (20 combinaciones) |
| CVaR ≥ VaR | 432 combinaciones sin excepción |
| Tiempo de simulación | ≈ 5 ms (objetivo < 2 s) |
| Empleo negativo | Mínimo global 0.0099 |
| Reproducibilidad por semilla | Exacta |
| Monotonía ventas → supervivencia | 47.9% → 62.8% |
| Deriva log-normal centrada | Media 19.93 partiendo de 20.0 |

---

## Fuente

Collaguazo Simbaña, I. J. y Rueda Rueda, B. E. (2026). *Tarificación del impacto de las empresas en la generación de plazas de empleo en el Ecuador en el periodo 2022–2024*. Universidad Central del Ecuador, Facultad de Ciencias Económicas, Carrera de Estadística. Tutor: Mat. Galo Vinicio Izquierdo Espinosa.

Datos: Registro Estadístico de Empresas (REEM), INEC — 700 147 empresas en Pichincha, Guayas, Azuay y Santo Domingo de los Tsáchilas.

Esta aplicación es una herramienta de simulación con fines analíticos y académicos. No constituye asesoría financiera ni de inversión.
