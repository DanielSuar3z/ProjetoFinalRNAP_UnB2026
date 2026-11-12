# Guía: `src/resbo/prior/crm.py` — el prior físico (conectividades CRM)

**Rol de una línea:** ajusta un **CRM (Capacitance-Resistance Model)** con los
históricos de producción de los baselines, y convierte las conectividades τ̂
recuperadas en una **puntuación escalar por candidato** que la adquisición del
BO usa como sesgo decreciente.

**Qué es un CRM (idea física):** sin simular el yacimiento, un productor "huele"
a sus inyectores: su producción se explica como suma ponderada de las tasas de
los inyectores. Los pesos τ_ij (0 a 1) son las **conectividades**: qué fracción
del fluido inyectado por i llega a j. Modelo de capacitancia (acumulación) +
resistencia (transmisibilidad) — de ahí el nombre. Referencias: Sayarpour 2009
(tesis canónica), You & Albertazzi (GCPRM, con constantes de tiempo).

**La ecuación que se ajusta (forma steady-state simplificada):**

```
q_j  ≈  Σ_i τ_ij · rate_i  +  q0_j
```

con controles CONSTANTES dentro de cada run (patrón BIOPSO), la "excitación"
del sistema la aportan **los distintos runs** (cada corrida tiene rates
distintos), no la variación temporal. Apilando runs → regresión lineal.

---

## `class CRMModel`

### `__init__(self, n_inj, n_prod, min_excitation=50.0)` — los estados

| Atributo | Shape | Qué es | Por qué ese nombre |
|---|---|---|---|
| `self.tau` | `(8,4)` | **Las conectividades estimadas** — el corazón del prior. Fila i = inyector, columna j = productor. `tau_ij` = fracción del aporte de i que va a j. Letra griega τ: notación estándar de la literatura CRM (los papers la escriben τ o J^T). | |
| `self.q0` | `(4,)` | Producción "natural" de cada pozo sin soporte de inyector (el término independiente de la regresión; físicamente, declinación primaria/depleción). `q0` = caudal en estado "0 soportado". | |
| `self.fit_r2` | `(4,)` | **R² por pozo**: cuánto de la varianza del histórico explica el CRM. Es el semáforo de identificabilidad propuesto para el gate S2 (pozo con R² bajo → su fila de τ no es fiable). | `fit` = ajuste, `r2` = coeficiente de determinación. |
| `self.n_runs_used` | int | Cuántas corridas se apilaron en el ajuste (diagnóstico mínimo: n ≥ n_prod+1). | |
| `self.min_excitation` | 50.0 sm3/d | Umbral de tasa para considerar "excitada" una arista. **Está declarado pero aún no se usa** — queda para el filtro de runs casi-apagados cuando lleguen los datos reales (honestidad: es deuda documentada, no código muerto). | |

### `fit(self, X, curves, late_frac=0.4) -> self` — el ajuste

**Entradas:**
- `X` — `(n_runs, 12)` controles de cada corrida (del `SimDatabase.curves`).
- `curves` — lista de `(4,130)` de producción por pozo/mes.
- `late_frac` — fracción final de la historia usada. El modelo steady-state
  solo es válido cuando el campo ya "estabilizó" (después del transitorio del
  lag); promediar TODO el histórico mezclaría régimen transitorio y mentiría.
  `0.4` = usar los últimos 40% de los meses (≈ meses 79-130).

**Paso 1 — destilar cada run a un par de números:**
```python
t0 = int(T * (1.0 - late_frac))
q_mean = c[:, t0:].mean(axis=1)        # (4,) producción media del late-time
rates.append(ctrl[: self.n_inj])
```
Cada run → `(rates_j = 8 números, q_promedio = 4 números)`. `q_mean`: media en
el tiempo (eje 1 de la curva = meses) → un escalar por pozo. El eje `axis=1`
promedia a lo largo de las columnas (meses), dejando un valor por fila (pozo).

**Paso 2 — apilar y regredir por productor:**
```python
A = np.hstack([R, np.ones((n, 1))])    # matriz de diseño (n, 9)
sol, *_ = np.linalg.lstsq(A, y, rcond=None)
tau_j = np.clip(sol[: self.n_inj], 0.0, 1.0)
q0_j = max(sol[self.n_inj], 0.0)
```
- `R` — `(n_runs, 8)`: filas = runs, columnas = rates de cada inyector. La
  "excitación" que necesita la regresión: si todos los runs tuvieran el mismo
  rate, R sería rank-1 y τ no se identificaría (por eso el LHS de la coleta).
- `Q` — `(n_runs, 4)` producciones medias.
- `A` — **matriz de diseño** (nombre estándar de regresión): las 8 columnas de
  rates + una columna de 1s para el término independiente (q0). Una regresión
  lineal multivariable = resolver `A·θ ≈ y` con θ = [τ_1j…τ_8j, q0_j].
- `np.linalg.lstsq` — mínimos cuadrados (solución de mínimos residuos
  cuadráticos). El docstring lo llama "NNLS simplificado": NNLS (Non-Negative
  Least Squares) es el método estricto con τ≥0; aquí se aproxima con lstsq +
  clip, más simple y suficiente como prior (anotado como simplificación).
- `clip(sol[:n_inj], 0, 1)` — impone la física: una conectividad no puede ser
  negativa (no hay "anti-flujo") ni mayor que 1 (no llega más de lo inyectado).
  El clip SESGA la solución (los extremos se pegan a 0/1), compromiso
  aceptado documentado.
- `max(sol[n_inj], 0)` — la producción natural tampoco puede ser negativa.
- Se ajusta **pozo por pozo** (una regresión por columna j) porque cada productor
  tiene su propio vector de conectividades y su propio R².

**Paso 3 — diagnóstico R²:**
```python
yhat = A @ np.concatenate([tau_j, [q0_j]])
ss_res = ((y - yhat)**2).sum()          # residuos (lo no explicado)
ss_tot = ((y - y.mean())**2).sum()       # varianza total
self.fit_r2[j] = 1 - ss_res/ss_tot
```
Definición clásica de R²: `1 − SS_residual/SS_total`. `yhat` = predicción (y
gorro = "y-hat", predicho). El `+1e-12` en ss_tot evita 0/0 si un pozo fuera
constantemente cero.

**Guarda:** `self.n_runs_used = n` y devuelve `self` (patrón *fluent*:
`CRMModel(...).fit(X, curves)` en una línea, como se usa en el toy).

### `score(self, controls) -> float` — el prior escalar

Convierte τ̂ en una opinión sobre UN candidato. Tres componentes
adimensionales, combinadas 0.45/0.40/0.15:

```python
support = rate @ self.tau        # (4,) soporte predicho por pozo
```
- Predicción CRM: cuánto líquido sostendrá cada productor con ESTAS tasas.

| Componente | Fórmula | Intuición física | Peso |
|---|---|---|---|
| `s` — soporte | `support.sum()/rate.sum()` | Fracción del agua inyectada que el CRM cree que LLEGA a productores (vs perderse). Alta = inyección eficaz. Como filas de τ suman 1, está en [0,1]. | 0.45 |
| `voidage` — balance de vacío | `1 − |1 − support.sum()/rate.sum()|` | Balance de masa: producción total ≈ inyección total (reservorio lleno: "voidage replacement" ~100%). Se penaliza cualquier desbalance (por arriba o abajo). En términos de `s`: `1−|1−s|` = `min(s, 2−s)` acotada. | 0.40 |
| `brk` — expectante de breakthrough | `1 − Σ_j p_j²` con `p = support/Σsupport` | Mide si el soporte está **concentrado** (un inyector dominando a un pozo → canal preferente → rompe agua temprano) o repartido. `Σp²` es el índice de concentración (tipo Gini/Herfindahl en economía); 1/n_prod = uniforme (bueno), 1 = un dominante (malo). `brk` = break(t)rough. | 0.15 |

**La suma ponderada `0.45·s + 0.40·voidage + 0.15·brk`** — los pesos son
juicio de ingeniería (soporte manda, balance casi tanto, concentración
complementa). **Ojo, riesgo conocido y documentado en CHANGELOG:** en el mock
este score premia sobre-inyectar (más soporte ⇒ más score) mientras el NPV
castiga el agua cara → el prior ENGAÑÓ al BO en el toy (593M vs 611M). La
corrección propuesta (pendiente S2): que el score incluya el costo del agua.

### `score_many(self, X) -> np.ndarray` — la versión vectorizada

Misma fórmula sobre `(n_cand, 12)` entero de un golpe (con `axis=1`/
`keepdims` para las sumas por candidato). Existe porque la adquisición puntúa
4000 candidatos por iteración y el bucle Python sería ~1000× más lento. Es
duuplicación deliberada `score`/`score_many`: la escalar es la que se lee y
testea; la vectorizada es la que corre. (Si se cambia una, se cambia la otra.)

---

## `class VoidagePrior` — el baseline de ablación que degenera

Responde una pregunta incómoda del protocolo: **si el prior fuera solo el
balance de voidage (sin CRM), ¿informaría algo?** La respuesta del código es
que NO: para un vector de control constante, el balance de masa global es
idénticamente ~1 para cualquier candidato ⇒ score constante ⇒ z-score de una
constante = 0 para todos. Por eso sus métodos devuelven `0.0`/`zeros`:
documenta matemáticamente que **el voidage solo es un caso degenerado sin
poder de guiado**, y que el prior útil del proyecto es el CRM completo. Existe
como clase para que el protocolo de ablación ("prior=voidage") sea ejecutable,
no teórico.

---

## Cómo se conecta con el resto

1. `toy_experiment.py` paso 2: `db.curves(["mfo","gwo","pso","foa"])` →
   `CRMModel.fit` → se printea `corr(τ̂, τ_true)` (el test de
   identificabilidad).
2. `BayesOptimizer(prior_model=crm)`: su `score_many(cand)` entra en la
   adquisición mezclada con EI ([bo_acquisition.md](bo_acquisition.md)).
