# Guía: `src/resbo/bo/acquisition.py` — dónde se decide a qué se simula

**Rol de una línea:** convierte la `(μ, σ)` del surrogate (+ el score del CRM)
en un **ranking de candidatos**: cuál es el próximo punto que vale la pena
simular. Es el cerebro de la adquisición y el sitio exacto donde vive la
hipótesis del proyecto (prior físico guiando la búsqueda).

---

## `def _norm_cdf(x)`

Función de distribución acumulada de la normal estándar, Φ(x), construida con
la función de error:

```python
Φ(x) = ½·(1 + erf(x/√2))
```

- `erf` de `math` es escalar → `np.vectorize(erf)` la "eleva" a arrays. Feo
  pero sin dependencias (no se usa scipy para poder correr con solo numpy;
  scipy es opcional en requirements).
- `cdf` = **c**umulative **d**istribution **f**unction.

---

## `def expected_improvement(mu, sigma, f_best, xi=0.0) -> (n,)`

**EI — Expected Improvement** (Jones 1998), la adquisición clásica de BO:
la MEJORA ESPERADA sobre el mejor valor actual, integrando la densidad normal
de la predicción.

```python
sigma = np.maximum(sigma, 1e-12)          # σ > 0 numérico
imp   = mu - f_best - xi                  # mejora "esperada" (media)
z     = imp / sigma                       # en unidades de σ
ei    = imp*Φ(z) + sigma*φ(z)             # φ = pdf normal estándar
return np.maximum(ei, 0.0)
```

| Argumento | Qué es y por qué se llama así |
|---|---|
| `mu` | `(n,)` μ(x): predicción media del surrogate por candidato. |
| `sigma` | `(n,)` σ(x): su incertidumbre. El `maximum(·,1e-12)` evita 0/0 donde la σ colapsó (p. ej. en puntos ya observados). |
| `f_best` | escalar: el mejor NPV visto hasta ahora. EI es "cuánto espero superar `f_best`". |
| `xi` (ξ) | margen de exploración: exigir superar `f_best+xi` hace la adquisición más audaz. Default 0 (neutro). |

**Lectura de la fórmula:** un candidato puntúa alto si (a) `mu ≫ f_best`
(explotación: predice buen NPV) **o** (b) `sigma` enorme (exploración: nadie
ha mirado ahí). `φ(z) = e^{−z²/2}/√(2π)` es la densidad; el producto
`imp·Φ(z)` domina cuando se espera mejorar, `σ·φ(z)` pondera la incertidumbre.
EI ≥ 0 siempre: un candidato no puede "esperar empeorar" (se puede no
simularlo, no "ganar por negativo").

---

## `def _rank01(v) -> (n,)`

```python
order = np.argsort(np.argsort(v))    # rango (0 al menor … n-1 al mayor)
return order / (n-1)                 # normalizado a [0,1]
```
Argsort doble = "el rango de cada elemento": argsort da posiciones ordenadas;
argsort de eso, el rango por posición original. Divide entre `n−1` ⇒ el mejor
=1.0, el peor =0.0.

**Por qué rangos y no valores:** decisión documentada en CHANGELOG (2026-11-06).
El EI del ensemble MC-dropout tiene PICOS enormes (un punto con σ alta genera
EI 10⁸ veces mayor que el resto). En z-score de valores, cualquier peso `w`
del prior quedaría swampeado (brazos con/sin prior daban resultados IDÉNTICOS:
el bug visible). Con rangos, ambas señales viven en [0,1] y `w` controla la
mezcla de forma interpretable (w=0.5 = "el prior vale media señal EI").

---

## `def acquisition(mu, sigma, f_best, prior_scores=None, w_prior=0.0, xi=0.0)`

La mezcla final:

```python
s = _rank01(ei)
if prior_scores is not None and w_prior > 0:
    s = s + w_prior * _rank01(prior_scores)
```

- `prior_scores` — `(n,)` scores del CRM para los mismos candidatos (de
  `CRMModel.score_many`).
- `w_prior` — el peso actual `w(n)` (lo calcula el laço con `prior_weight`).
- **`bo` sin prior** llega aquí con `prior_scores=None` (o `w=0`) ⇒ la
  adquisición es EI puro. **`bo_prior`** llega con ambos ⇒ ranking = rango_EI
  + w·rango_prior. Un solo código, dos brazos.
- El ranking resultante `s` solo importa por su ORDEN (el laço toma el top-q),
  así que sumar rangos es una fusión de rankings tipo "orda".

---

## `def prior_weight(n_obs, w0, tau_decay) -> float`

```python
w = w0 * exp(-n_obs / tau_decay)
```

Decaimiento exponencial estilo **DynMeanBO** (ICLR 2026): el prior pesa `w0`
al inicio (data-scarce, donde el surrogate no sabe nada) y se apaga
asintóticamente cuando el surrogate ya aprendió.

| Parámetro | Default (protocol.yaml) | Qué es |
|---|---|---|
| `w0` | 0.5 | Peso inicial del prior (medio: ni manda ni es decorativo). |
| `tau_decay` (τ_decaimiento) | 30 observaciones | Constante de tiempo: tras ~30 sims propias el peso cayó a `w0/e ≈ 0.18`; tras 60, a 0.07. Está en **sims**, no iteraciones: la "edad" del BO es cuánto ha simulado. |
| `n_obs` | — | ⚠️ Ojo de diseño: el laço le pasa `n_sims` = simulaciones PROPIAS del brazo BO, **no** el pool total de la BD. Los baselines son conocimiento de otros; el prior debe estar vivo mientras el BO mismo es data-scarce (CHANGELOG 2026-11-06, bug que mataba el prior desde el iter 0). |

Nota: esta `tau` de decaimiento **no** es la `tau` de conectividad del CRM —
colisión de letra griega, sin relación física.

---

## Cómo se conecta con el resto

Solo `BayesOptimizer.run()` llama `acquisition(...)`; `prior_weight` se llama
en el laço. `expected_improvement` se exporta por si se quieren curvas de EI
para las figuras del artículo.
