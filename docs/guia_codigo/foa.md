# Guía: `src/resbo/foa.py` — Fruit Fly Optimization Algorithm (réplica del repo BIOPSO)

**Rol de una línea:** algoritmo de la **mosca de la fruta** (Pan 2012), el
baseline más simple de la familia BIOPSO: un solo "centro de olfato" y
elitismo puro. Réplica exacta de `storms_unb/foa/foa_wellcontrol.py`
(referencia en `docs/ref_biopso/foa_wellcontrol.py`).

**Etimología de la metáfora (del paper de Pan):** la mosca de la fruta tiene un
olfato mucho mejor que su vista: vuela al azar siguiendo el **olor** (aquí, el
NPV medido en cada punto); cuando encuentra comida, usa la **vista** para
guiar a toda la manada hacia ese punto exacto. Los pasos 1–7 del paper están
comentados en el código con números de paso.

---

## `class FOA`

### `__init__(self, protocol, simulator, db, seed=0)`

`self.n` (flies), `self.iters` (rondas de vuelo), `lb/ub`, `sim`, `db`, `seed`
— igual que la familia (ver [mfo.md](mfo.md)). Plus del algoritmo:

| Atributo | Default | Qué es |
|---|---|---|
| `self.step` | 0.1 | **Radio de vuelo**: cada mosca se ubica a `center ± U(0,1)·step·ancho_de_caja` por coordenada. El paper no fija valor (depende del problema); el repo lo expone como fracción de caja (`foaStep=0.1`). YAML: `protocol["foa"]["step"]`. |

### `run(self) -> float` — los pasos del paper

```python
center = 0.5 * (self.lb + self.ub)
gbest, gval = center.copy(), -np.inf
```
- **Paso 1 — centro de osmóforos.** En el repo arranca en el "control base del
  caso" (histórico del Egg); aquí es el centro de la caja (mismo rol de ancla
  que la partícula 0 de PSO). `center` = la posición desde la que la manada
  "huele". `gbest/gval` = mejor histórico (misma semántica que PSO).

Cada ronda:

```python
perturb = (rng.random((self.n, d)) * 2.0 - 1.0) * self.step * span
fly = center + perturb
fly[0, :] = center
fly = np.clip(fly, self.lb, self.ub)
```
- **Paso 2 — búsqueda por olfato.** `(rand·2−1)` = uniforme en ±1 →
  perturbación simétrica de radio `step·span`.
- `fly[0] = center` — **la mosca 0 se evalúa EXACTAMENTE en el centro**. No es
  decorativo: es lo que hace al algoritmo elitista sin gastar una simulación
  extra fuera de la ronda — como el centro coincide con `gbest` (ver paso 7),
  la mosca 0 siempre vuelve a medir el mejor punto conocido y por eso el
  best-so-far por ronda **nunca puede bajar** (es lo que verifica
  `test_foa_elitism_and_budget`).
- Clip final a la caja.

```python
val = np.empty(self.n)
for i in range(self.n):
    val[i] = self._eval(fly[i], it)
```
- **Pasos 3-4 — concentración de olor.** En el paper de Pan,
  `Smell_i = Función(1/Dist_i)` con una distancia a un "origen" que solo tiene
  sentido en su ejemplo financiero. La adaptación estándar (y la del repo) es
  **medir el fitness directamente en la posición**: 1 mosca = 1 simulación.
  Documentado como "diferencia consciente nº1" en el archivo.

```python
j = int(np.argmax(val))
if val[j] > gval:
    gval, gbest = float(val[j]), fly[j].copy()
    center = gbest.copy()
```
- **Pasos 5-7 — el mejor olor y el vuelo de la manada.** El centro SOLO se
  mueve si la ronda superó al mejor histórico; si no, la manada vuelve a
  volar alrededor del mismo punto. Eso es **elitismo puro**: sin parámetros
  extra, sin costo en sims.
- `.copy()` otra vez: evitar vistas de numpy sobre la matriz `fly` que la
  siguiente ronda reutiliza.

`return float(gval)`.

### `_eval(self, x, it)`

Contacto único con el simulador; etiqueta `"foa"` en la BD. Presupuesto:
`pop·iters` simulaciones (evaluación dentro del lazo, como PSO).

---

## Qué hace (y qué no) a FOA — para related work

- **No tiene** memoria individual (vs PSO), jerarquía de líderes (vs GWO), ni
  escape/reinicio (vs HGSO). Un solo centro, sin estados por mosca: el más
  simple de los cuatro.
- Por eso convergencia lenta y fácil de atrapar (en el toy es el peor brazo,
  ~547M vs ~585M de MFO): la perturbación de radio fijo `0.1·span` alrededor de
  un punto es una búsqueda local con ruido.
- El docstring del repo lo señala candidato natural para **extensión memética**
  (refinamiento local por gradiente StoSAG) — línea futura interesante para
  el grupo.
