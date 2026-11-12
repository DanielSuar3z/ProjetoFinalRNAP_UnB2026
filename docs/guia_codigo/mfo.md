# Guía: `src/resbo/mfo.py` — Moth-Flame Optimizer (polillas y llamas)

**Rol de una línea:** metaheurística de enjambre (Mirjalili 2015) donde cada
**polilla** es un candidato de control y cada evaluación es **una simulación**
(compra de fitness patrón BIOPSO).

**Etimología de la metáfora:** MFO = **M**oth-**F**lame **O**ptimizer. Las
polillas vuelven en espiral hacia las llamas; aquí las "llamas" son los
mejores candidatos de la población anterior y la espiral es la ecuación de
movimiento. Nombre de la clase = siglas del paper.

---

## `class MFO`

### `__init__(self, protocol, simulator, db, seed=0)`

| Atributo | Qué es | Por qué ese nombre |
|---|---|---|
| `self.lb`, `self.ub` | Bordes de la caja (12,), robados al `Protocol`. | **l**ower/**u**pper **b**ound. |
| `self.n` | Tamaño de población (nº de polillas). | `n` = número, lectura corta estándar. Se lee de `protocol["budget"]["metaheuristics"]["pop"]`: **los presupuestos viven en el YAML, no en el código**. |
| `self.iters` | Iteraciones (rondas de vuelo). | `iters` = iterations. |
| `self.sim` | El simulador (mock u OPM): quien pone el fitness. | `sim` = simulator. |
| `self.db` | La base de datos donde cada evaluación deja su fila. | |
| `self.seed` | Semilla del RNG de ESTA corrida. Dos seeds = dos corridas independientes del mismo algoritmo (las "reps" del artículo). | |

### `run(self) -> float` — el laço, línea a línea

```python
rng = np.random.default_rng(self.seed)
```
- `rng`: generador aleatorio **moderno** de numpy (API `default_rng`, no el
  `np.random.seed` global legado). Local al método: dos MFO con la misma seed
  producen exactamente la misma trayectoria sin contaminarse entre sí.

```python
d = len(self.lb)                    # d = dimensión = 12
X = self.lb + rng.random((self.n, d)) * (self.ub - self.lb)
```
- `d` — dimensión del espacio de búsqueda (convención de optimización).
- `X` — matriz de población `(n, 12)`: fila i = posición (vector de control) de
  la polilla i. `X` = convención de "matriz de candidatos".
- `lb + rand·(ub−lb)` — el muestreo uniforme canónico en la caja.

```python
F = np.array([self._eval(x, 0) for x in X])
```
- `F` — fitness de cada polilla `(n,)`. **La evaluación inicial también gasta
  simulaciones** (por eso MFO/GWO gastan `pop·(iters+1)` sims y PSO/FOA solo
  `pop·iters`: estos últimos evalúan dentro del laço).

```python
best_fit = float(F.max())
```
- Mejor histórico. `float(...)` para devolver escalar Python limpio.

**El laço de iteraciones:**

```python
order = np.argsort(F)               # indices de peor a mejor
flames = X[order]                   # "llamas" ordenadas: flames[-1] = mejor
```
- `order` — permutación que ordenaría F ascendente (`argsort`).
- `flames` — población reordenada por aptitud. En MFO los candidatos son a la
  vez polillas y llamas; la convención del paper: llama 0 = peor.

```python
a = -1 - it / max(self.iters - 1, 1)   # decae -1 → -2
```
- `a` — coeficiente de la ecuación espiral. En el paper original va de −1 a −2
  linealmente; controla cuán "apretada" es la espiral (más negativo = polilla
  más cerca de la llama). El `max(...,1)` evita división por 0 con 1 iteración.

```python
j = int(np.floor((self.n - 1) * (rng.random() ** 2)))
flame = flames[min(j, self.n - 1)]
```
- Selección de la llama objetivo. `rng.random()**2` **sesga la elección hacia
  el principio del índice** → como `flames` está ordenado de peor a mejor…
  ¡ojo!, aquí el sesgo va hacia las llamas de índice bajo (peores); se eligió
  así deliberadamente para mantener exploración (¡en MFO canónico el sesgo es
  hacia la última llama!). Decisión registrada: versión local exploratoria; el
  paper usa `random` sin exponenciar y flames del final. **Este es un punto
  revisable del código** — estudiarlo en S2.
- `int(np.floor(...))` — convierte el float en índice entero.

```python
D = np.abs(flame - X[i])            # distancia polilla-llama
t = a * rng.random()                # parámetro de la espiral
Xn[i] = D * np.exp(t) * np.cos(2*pi*t) + flame
```
- La **ecuación espiral logarítmica** del paper: la polilla gira alrededor de
  la llama con radio que decae exponencialmente (`D·e^t`) mientras oscila
  (`cos 2πt`). `D` = distance; `t` = parámetro continuo de la espiral.

```python
Xn = np.clip(Xn, self.lb, self.ub)
F = np.array([self._eval(x, it + 1) for x in Xn])
```
- `clip` — recorta cada coordenada a la caja (fuera de la caja el candidato no
  tiene sentido físico).
- Evaluación de la nueva población: **aquí se pagan las simulaciones** (n por
  ronda). `it + 1` porque la inicial fue la "iteración 0".

`return best_fit` — el mejor NPV visto en toda la corrida (no solo el de la
última generación: es "best-so-far").

### `_eval(self, x, it) -> float` (método privado)

El **único punto de contacto algoritmo↔mundo**: simula, inserta la fila en la
BD con etiqueta `"mfo"`, devuelve el NPV. El guion bajo = "usar solo dentro de
la clase". Que `db.add` esté aquí y no en el simulador es lo que garantiza que
no exista ninguna evaluación fantasma.

---

## Cómo se conecta con el resto

`toy_experiment.py` instancia `MFO(proto, MockSimulator, db, seed)` por seed y
llama `run()`. Sus filas (`algo="mfo"`) alimentan la BD → CRM y surrogate.
