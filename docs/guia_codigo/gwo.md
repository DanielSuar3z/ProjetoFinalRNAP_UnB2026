# Guía: `src/resbo/gwo.py` — Grey Wolf Optimizer (manada de lobos)

**Rol de una línea:** metaheurística (Mirjalili et al. 2014) donde la manada
caza guiada por sus **tres líderes** (alfa/beta/delta); 1 lobo = 1 simulación.

**Etimología de la metáfora:** GWO = **G**rey **W**olf **O**ptimizer. En la
manada real hay jerarquía: α lidera, β y δ le asisten, ω (el resto) obedece.
En el algoritmo esa jerarquía se traduce en "los tres mejores candidatos
actúan como guías del movimiento de todos los demás".

---

## `class GWO`

### `__init__`

Idéntico en estructura y nombres a `MFO.__init__` (ver [mfo.md](mfo.md)):
`lb/ub` (caja), `n` (población), `iters`, `sim`, `db`, `seed`. La familia de
baselines comparte contrato a propósito: `(protocol, simulator, db, seed)` +
`run()` — se pueden intercambiar en el toy con cambiar una letra.

### `run(self) -> float`

**Inicialización** — igual que MFO: `X` uniforme en la caja `(n,12)`, fitness
`F` pagando n simulaciones (iteración 0).

```python
alpha, beta, delta, fa, fb, fd = self._leaders(X, F)
```
- `alpha`, `beta`, `delta` — posiciones de los 3 mejores candidatos. Nombres
  griegos = rangos de la jerarquía del paper (α > β > δ > ω).
- `fa, fb, fd` — sus fitness ("f" de fitness + letra del rango). Se guardan
  aparte para no re-ordenar la población cada vez.
- `best = fa` — best-so-far de la corrida.

**Laço de caza:**

```python
a = 2.0 - 2.0 * it / max(self.iters - 1, 1)     # decae 2 → 0
```
- `a` — el coeficiente de convergencia del paper. Controla la Amplitud del
  movimiento: con `a=2` los lobos exploran lejos (coeficiente A puede valer
  ±2), con `a=0` soloexplotan alrededor de los líderes. Exploración →
  explotación es la "curva de temperatura" del algoritmo.

```python
Xn[i] = self._hunt(X[i], (alpha, beta, delta), a, rng)
Xn = np.clip(Xn, self.lb, self.ub)
F = np.array([self._eval(x, it + 1) for x in Xn])
```
- Cada lobo se mueve una sola vez por ronda (hacia una mezcla de sus 3
  líderes), se recorta a la caja, y se paga su simulación.

```python
alpha, beta, delta, fa, fb, fd = self._leaders(X, F)
best = max(best, fa)
```
- La jerarquía **se reeligue cada ronda** desde cero con la población actual
  (el paper lo permite: la jerarquía refleja la aptitud de la ronda).
- `best` sigue siendo best-so-far histórico aunque α de esta ronda sea peor.

### `@staticmethod _hunt(x, leaders, a, rng)`

La **ecuación de caza** del paper, aplicada tres veces (una por líder) y
promediada:

```python
for leader in leaders:
    A = 2 * a * rng.random(len(x)) - a     # A ∈ [-a, +a]
    C = 2 * rng.random(len(x))             # C ∈ [0, 2]
    D = np.abs(C * leader - x)             # distancia "percibida"
    acc += leader - A * D
return acc / 3.0
```

| Variable | Qué es | Por qué ese nombre/valor |
|---|---|---|
| `A` | Coeficiente de **distancia**. Con \|A\|>1 el lobo salta más lejos del líder (explora), con \|A\|<1 se acerca (explota). Como A tiene componente aleatoria, la caza es estocástica. | `A` = nombre del paper ("coefficient A"). |
| `C` | Coeficiente de **Aleatoriedad**: "el lobo no ve al líder exactamente donde está" (C=2 sobreestima, C=0 lo subestima). Introducir ruido en la percepción es lo que evita convergencia prematura determinista. | `C` = nombre del paper. |
| `D` | Distancia lobo→líder percibida, `D = |C·líder − x|`. | `D` = distance, ecuación (3.4) del paper. |
| `leader - A*D` | Posición estimada del lobo **respecto a ese líder** (ecuación 3.5). | |
| `acc / 3.0` | Los tres "votos" (α, β, δ) se promedian: el lobo va a donde los tres líderes le dicen a la vez. | `acc` = accumulator. |

Es `@staticmethod` porque no toca nada del objeto: es pura función matemática
(x, líderes, a, rng) → nueva posición.

### `@staticmethod _leaders(X, F)`

```python
order = np.argsort(F)[::-1]          # índices de MAYOR a menor fitness
i0, i1, i2 = order[0], order[1], order[2]
return X[i0].copy(), X[i1].copy(), X[i2].copy(), F[i0], F[i1], F[i2]
```
- El `[::-1]` invierte el argsort (ascendente → descendente; maximizamos NPV).
- `.copy()` en las posiciones: si no, el retorno sería una **vista** de `X` y
  al mover los lobos se "moverían" también los líderes guardados (bug clásico
  de numpy con vistas).
- Exige `n ≥ 3` poblacion (garantizado por presupuesto: pop=10).

### `_eval(self, x, it)`

Igual que en MFO pero etiqueta `"gwo"` en la BD. Un punto de contacto único
con el simulador.

---

## Comparación rápida con MFO (para el artículo)

| | MFO | GWO |
|---|---|---|
| Guía del movimiento | 1 llama elegida al azar (sesgado) | 3 líderes fijos (top-3) |
| Mecánica | espiral logarítmica | ecuación de caza A/C |
| Cuántos "jefes" siguen los individuos | 1 por individuo/ronda | 3, promediados |
| Simulaciones por corrida | `pop·(iters+1)` | `pop·(iters+1)` |

Ambos gastan la misma presupuesto; si en los resultados uno converge más
rápido, la diferencia es **de dinámica de búsqueda**, no de cómputo.
