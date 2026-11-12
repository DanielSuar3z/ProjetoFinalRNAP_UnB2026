# Guía: `src/resbo/bo/loop.py` — `BayesOptimizer`, el laço central

**Rol de una línea:** orquesta el ciclo completo de BO batch: DoE inicial →
entrenar surrogate → puntuar pool → elegir batch diverso → simular → repetir.
Es el "brazo BO" y el "brazo BO+prior" a la vez (según `prior_model`).

**Regla de oro del proyecto (en el docstring):** el prior entra SOLO por la
adquisición; **nunca sustituye simulaciones**. Toda evaluación de fitness es
una simulación real registrada en la BD.

---

## `class BayesOptimizer`

### `__init__(self, protocol, simulator, db, prior_model=None, n_models=5, batch=None, iters=None, pool=4000, n_init=5, w0=None, tau_decay=None, epochs=300)`

| Atributo | Default | Qué es | Por qué ese nombre |
|---|---|---|---|
| `self.p` | — | el `Protocol` (guardado con nombre corto para no chocar con `prior`... y por pereza sana). |
| `self.sim` | — | simulador (mock/OPM). |
| `self.db` | — | `SimDatabase` compartida con los baselines. |
| `self.prior` | `None` | **El CRM ya ajustado o None.** Este argumento ES la diferencia `bo_prior` vs `bo`: con CRM → prior activo; `None` → EI puro. No hay dos clases. |
| `self.lb/ub` | del protocolo | caja de búsqueda (12,). |
| `self.batch` | `budget.bo.batch` (5) | `q` candidatos elegidos POR iteración (batch). Se llama `batch` no `q` para ser explícito; `q` es la letra del paper. Batch>1 = BO paralelo (MC2 puede lanzar ≤7 a la vez). |
| `self.iters` | `budget.bo.iters` (10) | nº de ciclos surrogate→selección→simulación. |
| `self.pool` | 4000 | nº de candidatos aleatorios que se generan por iteración para puntuar. `pool` = "charca" donde pescar los mejores. 4000 es el equilibrio: suficiente densidad en 12D sin coste de predicción excesivo (80 predicciones MC × 4000 por iter). |
| `self.n_init` | 5 | puntos LHS iniciales ANTES del laço (arrancar con 0 datos es imposible: el surrogate no puede entrenar). |
| `self.w0` | `prior.w0` (0.5) | peso inicial del prior (override del YAML si se pasa). |
| `self.tau_decay` | `prior.decay_obs` (30) | constante de decaimiento en nº de sims. |
| `self.n_models` | 5 | miembros del ensemble BNN. |
| `self.epochs` | 300 | epochs por red por re-entrenamiento. |
| `self.rng` | RNG(0) | re-sembrado en `run(seed)`. |
| `self.history` | `[]` | lista `(n_sims, best_npv)` — la curva best-so-far del brazo (alimenta AUC y figuras). Se puebla SIMULACIÓN a simulación, no por iter. |

### `run(self, algo_tag, seed, base_algos=("mfo","gwo","pso","foa")) -> float`

**Argumentos:**
- `algo_tag` — etiqueta que quedará en la BD (`"bo"` o `"bo_prior"`). Lo pasa
  el script: la clase no conoce los nombres de los brazos.
- `base_algos` — qué brazos "gratis" se poolean para entrenar el surrogate.
  Esta línea es la materialización del argumento de ResBO: **el BO no repite
  el trabajo de los baselines, lo hereda.**

**Fase 1 — estado inicial:**
```python
X_obs, y_obs, _ = self.db.controls_npv(list(base_algos))
init = self._lhs(self.n_init)
for x in init:
    out = self.sim.evaluate(x)
    self.db.add(algo_tag, seed, -1, x, ...)
```
- `X_obs, y_obs` — el pool heredado (baselines) sobre el que entrenará el
  surrogate. `_obs` = observed (observado).
- Los `n_init` DoE propios se registran con `iteration=-1` (convención BD:
  "antes del laço").
- `n_sims = self.n_init` — contador de sims PROPIAS (el denominador moral del
  brazo y la `n_obs` del decaimiento).

**Fase 2 — el laço (por iteración):**
```python
model = self._train_surrogate(list(base_algos) + [algo_tag])
```
- **Se re-entrena el ensemble CADA iter** con pool+historial del brazo. Con
  ~200 sims y 300 epochs es barato en el mock; en MC2 será el cuello de botella
  (candidato a fine-tuning).
- Se le pasa `base_algos + [algo_tag]`: el BO entrena también con SUS propias
  sims acumuladas (no solo las heredadas).

```python
cand = self._lhs(self.pool)
mu, sigma = model.predict(cand)
f_best = float(y_obs.max())
w = prior_weight(n_sims, self.w0, self.tau_decay) if self.prior else 0.0
prior_s = self.prior.score_many(cand) if self.prior else None
s = acquisition(mu, sigma, f_best, prior_s, w)
```
- El quinteto del cerebro: candidatos LHS → predicción (μ,σ) → umbral f_best →
  peso del prior vivo (cuenta `n_sims` propias, no el pool) → score CRM →
  ranking combinado. Ver [bo_acquisition.md](bo_acquisition.md).

```python
batch = self._batch_select(cand, s, self.batch, X_obs)
for x in batch:
    out = self.sim.evaluate(x); self.db.add(...); n_sims += 1
    self.history.append((n_sims, float(y_obs.max())))
```
- Se simula el batch (en MC2: ≤7 en paralelo) y se registra cada fila.
- `history` guarda `(nº sims propias, mejor NPV global)` tras CADA simulación.

`return float(y_obs.max())` — el mejor NPV alcanzado (pool incluido).

### `_lhs(self, n) -> (n,12)` — Latin Hypercube Sampling

```python
perm = self.rng.permutation(n)
X[:, j] = (perm + self.rng.random(n)) / n
X = lb + X*(ub-lb)
```
Truco estándar de LHS: en cada dimensión se permutan las n franjas `[i/n,(i+1)/n)`
y se tira un punto uniforme dentro de la franja asignada (`perm` da la franja,
`+random` el jitter). Garantiza que cada franja de cada dimensión tenga EXACTA-
MENTE un candidato (cobertura 1D perfecta, lo que el uniforme puro no asegura).
Es la fuente del DoE inicial y del pool — y la razón de que el CRM identifique
bien (excitación maximal).

### `_train_surrogate(self, algo_tags) -> BnnEnsemble | None`

Enruta `db.controls_npv(algo_tags)` → BNN nuevo (seed aleatoria del rng del
brazo) con `fit_norm_stats` + `fit`. Devuelve `None` si hay <6 datos (el laço
lo convierte en RuntimeError: es una invariante, no una condición normal).

### `_batch_select(self, cand, scores, q, X_obs) -> (q,12)` — diversidad

```python
order = np.argsort(-scores)              # candidatos por score descendente
for i in order:
    c = cand[i]
    if sel:
        dist = norm((S - c)/rng_span, axis=1).min()   # distancia normalizada
        if dist < 0.04: continue                      # demasiado parecido → descarta
    sel.append(c)
```
- Toma el top-q por score PERO con un radio de exclusión mutua de 0.04 en la
  caja normalizada: sin esto, EI/CRM suelen tener un pico ancho y los q=5
  elegidos caerían casi encima unos de otros → batch redundante (5 sims para
  aprender 1 región). `rng_span` = r**a**nge (ojo: `rng` aquí es "range", no
  el generador aleatorio — colisión de nombre local, consciente).
- `X_obs` entra a la firma pero hoy NO se usa en la distancia (solo entre
  miembros del batch); se dejó como interfaz para excluir también candidatos ya
  observados (deuda menor anotada).
- El `while len(sel) < q: append(cand[order[randint]])` es relleno: si con el
  radio de exclusión no se llenó el batch (pool muy concentrado), se completan
  con candidatos aleatorios del ranking (nunca se devuelve batch corto).

---

## Cómo se conecta con el resto

- **Entra:** `protocol`, `simulator`, `db` (compartida), `prior_model` (CRM
  ajustado por el script).
- **Sale:** filas `algo="bo"`/`"bo_prior"` en la BD + `history` para figuras.
- Dos instancias gemelas del mismo class = los dos brazos BO de la ablación.
