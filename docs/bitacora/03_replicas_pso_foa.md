# Paso 03 — PSO y FOA como réplicas exactas del repo BIOPSO

**Fecha:** 2026-11-07 · **Estado:** ✔ cerrado

## 1. Por qué aparecen PSO y FOA

Los baselines del artículo deben ser los algoritmos con los que se genera la
coleta REAL en el lab. Al revisar BIOPSO (vía API de GitHub con credencial
store, porque MC2 ya no respondía por SSH) se confirmó que existen:

- `src/ChronosUNB/storms_unb/pso/pso_wellcontrol.py`
- `src/ChronosUNB/storms_unb/foa/foa_wellcontrol.py`

Y con ellos un principio que se volvió regla del proyecto: **las versiones
locales replican las FÓRMULAS EXACTAS del repo, no las versiones canónicas de
los libros** (p. ej. el PSO de EvoloPy con decay de inercia NO es lo que corre
en BIOPSO). Razón: si el dataset de entrenamiento y las baselines del artículo
vienen del código del repo, una réplica "canónica" local sería un algoritmo
DISTINTO comparando contra otra cosa.

## 2. Qué replica exactamente cada archivo local

### `src/resbo/pso.py` ↔ `pso_wellcontrol.py`

| Propiedad del repo | Implementación local |
|---|---|
| `w = psoInertia = 0.7` **constante** (sin decay) | `self.w` fijo, leído de `protocol.yaml` sección `pso:` |
| `c1 = c2 = 1.5` (psoCognitive/psoSocial) | ídem |
| partícula 0 anclada al control base del caso | `pos[0] = self.base` (desde el paso 08, `protocol.base_controls`; antes centro de caja) |
| velocidad inicial ±5% del ancho de caja | `(rng.random(...)−0.5)·span·0.1` |
| **sin** clip de velocidad (no hay Vmax); solo clip de posición a la caja | igual |
| fitness = NPV medido, maximizar directo (sin truco `costo = −NPV`) | igual |
| gbest derivado de los pbest por comparación de NPV | igual |
| evaluación dentro del lazo: `pop·iters` simulaciones (1 partícula = 1 sim) | igual, y el test lo exige por conteo |

### `src/resbo/foa.py` ↔ `foa_wellcontrol.py` (Fruit Fly Optimization, Pan 2012)

| Propiedad del repo | Implementación local |
|---|---|
| un único centro de osmóforos | `center` único, movido a la mejor mosca |
| mosca 0 evaluada EXACTAMENTE en el centro | elitismo puro: el centro solo cambia si una mosca lo supera (monotonidad garantizada, testeada) |
| perturbación `±foaStep × ancho de caja` por coordenada | `foa.step: 0.1` (default del repo `foaFlies=30` → overridden a 15 en la coleta) |

Los hiperparámetros quedaron **expuestos en `protocol.yaml`** (secciones
`pso:`/`foa:`) con defaults igualitos al repo — así el día que el config real
del lab cambie, la réplica local se actualiza sin tocar código.

## 3. Tests semánticos (no solo numéricos)

- `test_pso_matches_repo_semantics`: parámetros w/c1/c2 correctos, ANCLA en el
  control base (partícula 0 de la iteración 0 == `base_controls`), y conteo
  exacto `pop·iters` de evaluaciones.
- `test_foa_elitism_and_budget`: la mejor fitness es **monótona** (elitismo), y
  el conteo de simulaciones es el presupuestado.

## 4. Nota de honestidad metodológica (para el artigo)

Con PSO/FOA integrados, los 4 brazos comparten presupuesto pero NO comparten
la misma "forma" de gastarlo: PSO/FOA evalúan dentro del lazo (`pop·iters`),
MFO/GWO evalúan la inicial además (`pop·(iters+1)` en el local). Esta asimetría
de ±pop simulaciones está documentada en tests y notas; en la coleta real de
BIOPSO todos los configs S1 son 15×6 = 90, así que la comparación del artículo
usa la contabilidad del repo.

## ¿Qué quedó abierto?

- Nada técnico. Pendiente: confirmar contra la rama main local del clone de
  BIOPSO cuando MC2 vuelva (se hizo en el paso 06: confirmado, el clone local
  contiene ambos archivos tal como se consultaron).
