# Paso 02 — Núcleo del pipeline local sobre el MockSimulator (ResBO v0.1.0)

**Fecha:** 2026-11-06 · **Commit:** `782c5a2` · **Estado:** ✔ cerrado (superpuesto parcialmente por el paso 08)

## 1. El problema de desarrollo sin MC2

Sin cluster no hay simulaciones, y sin simulaciones no hay forma de depurar un
lazo BO (que es secuencial, con surrogate, prior y estadística). La decisión
estructurante: **escribir TODO el pipeline contra un contrato de simulador y
proveer un juguete local** que corre en milisegundos:

```python
out = sim.evaluate(controls)     # controls: (12,) = [rate×8 inyectores, bhp×4 productores]
out["npv"]        -> float                       # el fitness
out["prod_curve"] -> (n_prod, horizon)           # producción por pozo y mes
```

`MockSimulator` (`src/resbo/simulator.py`) implementa física de juguete con la
estructura correcta del problema:

- **τ_true (8×4) oculta**: conectividades verdaderas que solo el mock conoce;
  cada inyector reparte 1.0 entre sus 4 productores (normalización por fila).
- Presión por productor: `c·dp/dt = soporte_j − q_j`, con
  `soporte_j = Σᵢ τᵢⱼ·rateᵢ·(1−exp(−t/lag))` (retardo del soporte).
  `q = PI·(p − bhp)` (ley del pozo).
- **Fracción de agua** que crece con el volumen inyectado *dirigido* acumulado
  (breakthrough): inyectar de más acelera agua, y el agua cuesta.
- NPV con descuento mensual.

Que un **CRM pueda recuperar τ_true** desde las curvas es la prueba de que el
juguete conserva la propiedad que hace interesting el problema real.

## 2. Los módulos y qué papel juega cada uno

| Módulo | Rol (detalle en `docs/guia_codigo/`) |
|---|---|
| `configs/protocol.yaml` | Única fuente de verdad de caja/economía/presupuestos/prior. NINGÚN número experimental vive en código. |
| `database.py` | BD de simulaciones (`simulations.csv`): cada fila = (algoritmo, seed, iteración, agente, controles, NPV). |
| `mfo.py`, `gwo.py` | Baselines con el patrón BIOPSO: 1 individuo = 1 simulación, controles constantes por vida de campo, fitness = NPV directo (maximizar). |
| `prior/crm.py` | CRM estático: mínimos cuadrados sobre las curvas → τ̂ᵢⱼ (8×4). |
| `models/bnn.py` | Ensemble de MLPs numpy con MC-dropout: media + σ epistémica para EI. |
| `bo/acquisition.py`, `bo/loop.py` | EI + mezcla con prior decaíble; lazo batch (pool LHS → puntuar → top-q sims "físicas"). |
| `analysis/metrics.py` | AUC de convergência, sims-al-umbral, **Mann-Whitney U** y **rank-biserial** (tamaño de efecto). |
| `scripts/toy_experiment.py` | Corre los brazos sobre el mock, guarda BD + figura + tabla. |

## 3. Decisiones de diseño que nacieron aquí (y sobrevivieron)

1. **El decaimiento del prior cuenta las sims PROPIAS del lazo BO**, no las
   observaciones pooled de los baselines. Motivo (lectura DynMeanBO): los
   baselines son conocimiento de OTROS algoritmos; el prior debe vivir mientras
   EL BO está data-scarce. Efecto colateral: con el bug anterior (contar el
   dataset completo) el prior estaba muerto desde el iter 0.
2. **EI y prior se combinan por RANGOS [0,1], no por z-score.** Con z-score,
   los picos del EI del ensemble swampeaban al prior (brazos con y sin prior
   daban resultados idénticos); con rangos, `w` controla la mezcla de forma
   interpretable.
3. Fixes del mismo día: curvas como `(n_prod, horizon)` (coherente con
   `database.curves()` y el reshape del CRM — antes `(T, n_prod)` rompía el
   CRM), y `database.curves()` devolvía solo el primer registro.

## 4. El hallazgo fundacional: el prior puede ser ENGAÑOSO

En la demo toy (2 seeds, mock de la época, economía de 2 términos):

```
bo_prior  media ≈ 593M      bo (sin prior)  media ≈ 611M
Mann-Whitney U(bo_prior vs bo): r_rb = −1.00  (el prior PERÓ always)
```

Diagnóstico físico: el prior CRM premia **soporte** (más inyección ↔ más
conectividad hacia el pozo), pero el mock castiga el **costo de manejar agua** →
el prior empuja sistemáticamente a sobreinyectar. Es la confirmación local del
riesgo central de todo prior-informed BO, y produjo tres reglas permanentes:

- (a) validar el prior contra baselines ANTES de confiarle peso;
- (b) el decaimiento `w(n)` no es ornamento, es el seguro;
- (c) el brazo de **ablación "BO sin prior" es obligatorio** en el diseño
  experimental (sin él no se puede atribuir la ganancia a nada).

## ¿Qué quedó abierto?

- PSO y FOA faltaban como brazos (la coleta BIOPSO los tiene) → paso 03.
- Publicar todo en el repo público del curso → paso 04.
- El mock y la caja eran "tipo Egg" genéricos, no el caso real → estalla en el paso 08.
