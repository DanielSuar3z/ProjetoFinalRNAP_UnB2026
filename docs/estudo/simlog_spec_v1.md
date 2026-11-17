# SPEC simlog v1 — contrato de registro por simulación (BIOPSO ↔ ResBO)

**Estado:** CONGELADO. Cualquier cambio de formato ⇒ `simlog_spec_v2.md` + aviso a ResBO.
**Fecha:** 2026-11-16 · **Autores:** ResBO (consumidor) · **Implementador:** BIOPSO (productor).
**Verificado contra:** clon local de BIOPSO (branch actual, archivo por archivo, 2026-11-16).

---

## 0. Para qué existe

El motor BIOPSO hoy guarda solo el resultado final por corrida (`results.csv`: 1 NPV) y
las curvas de la solución ganadora (`curvas.guardar_curvas`). El **histórico
`(controles → NPV)` de cada simulación interna se pierde** al terminar la corrida.
ResBO necesita ese histórico para:

1. entrenar el surrogate BNN → miles de filas `(controles_12d, NPV)` = **`simlog.csv`** (§1),
2. ajustar el prior CRM → curvas de producción/inyección **de cada simulación** = **`sim_curves.npz`** (§2).

Sin estos dos archivos no hay entrenamiento posible (ver diagnóstico en el
CHANGELOG de ResBO, 2026-11-16).

**Principio de diseño:** el parche es PURAMENTE ADITIVO. No cambia ninguna decisión
del algoritmo, ni el formato de ningún artefacto existente (§7 lista de no-regresión).

---

## 1. `simlog.csv` — una fila por simulación

**Ubicación:** `<work_dir>/simlog.csv` — en la RAÍZ del work_dir de la corrida
(`run_generic.py` crea `work_dir = <base>/run_XX`). Sobrevive a todo: `cleanTemp()`
solo borra directorios con "SPAN" en el nombre y `run_generic.py` solo borra `tmp/`
(verificado en el código). **NUNCA** dentro de un SPAN ni de `tmp/`.

**Cabecera EXACTA y congelada** (orden de columnas literal, separador `,`):

```
iteration,agent,inje_1,inje_2,inje_3,inje_4,inje_5,inje_6,inje_7,inje_8,prod_1,prod_2,prod_3,prod_4,npv
```

| Columna | Semántica |
|---|---|
| `iteration` | contador de iteración del algoritmo, **1-based**, el MISMO valor que recibe `log_convergence(work_dir, it, ...)`. |
| `agent` | índice 0-based del agente dentro de la ronda (partícula/wolf/moth/fly/gas n.º k, orden de evaluación). |
| `inje_1..8` | los 8 valores de tasa de inyección evaluados, **tal cual se asignaron** a `real.constraints.well_controls['inje_values'][0,:]` (unidades del deck: sm³/d métrico, caja [0, 150]). |
| `prod_1..4` | los 4 valores de BHP evaluados (`prod_values[0,:]`, bar, caja [200, 395]). |
| `npv` | fitness de ESA simulación: `NPVHandler.fieldNPVcum[-1]` — literalmente el mismo escalar que el algoritmo usó para elegir (definición idéntica a la fila de `npv_vals`). |

**Reglas de escritura:**

- **Fila por simulación realmente ejecutada** (agente evaluado = fila insertada).
- Orden de filas: (iteración, agente) — el mismo orden en que el algoritmo evaluó.
- Números con precisión completa (`repr(float)`, round-trip exacto).
- `iteration` debe casar 1:1 con las filas de `convergence.csv` del mismo work_dir.
- **Conteo esperado: `agents × iterations` para los 5 algoritmos** (PSO/GWO/MFO/FOA/HGSO
  evalúan TODOS dentro del lazo — verificado en el código del repo: no hay evaluación
  inicial fuera del lazo, a diferencia de EvoloPy).

**Semántica de re-ejecución:** al INICIO de `run()` (después de `real.span_data = []`)
se TRUNCA el log (`reset_simlog(work_dir)`): si un `run_XX` se re-ejecuta tras un
corte, el simlog no puede mezclar filas viejas y nuevas.

---

## 2. `sim_curves.npz` — curvas de CADA simulación

**Ubicación:** `<work_dir>/sim_curves.npz` (misma raíz, mismas garantías de supervivencia).

**Claves y shapes congeladas** (`np.savez_compressed`):

| Clave | Shape | dtype | Contenido |
|---|---|---|---|
| `wpr` | `(n_sims, n_prod, n_t)` | float64 | producción de agua por productor y paso, de `handler.data.wellRawData['WPR'][índice del pozo]` |
| `wir` | `(n_sims, n_inj, n_t)` | float64 | inyección de agua por inyector y paso, de `wellRawData['WIR']` |
| `time_days` | `(n_t,)` | float64 | `handler.data.timeSteps` (días de simulación) |
| `sim_index` | `(n_sims,)` | int64 | `0..n_sims-1` — casa 1:1 con las filas de `simlog.csv` |
| `producer_wells` | `(n_prod,)` | unicode | nombres de los pozos productores en el orden de las filas de `wpr` |
| `injector_wells` | `(n_inj,)` | unicode | ídem inyectores para `wir` |

**Invariantes (el consumidor las va a exigir):**

1. **Orden plano (ronda, agente) idéntico al de `simlog.csv`**: la fila i del npz es la
   simulación de la fila i del CSV. Esto se cumple automáticamente porque ambos salen
   del mismo recorrido de `real.span_data` (ver §3.2) — es LA invariante del contrato.
2. Para Egg + `predictionWindow: 130`: `n_t == 130`, `n_prod == 4`, `n_inj == 8`.
3. Orden de pozos determinista: **nombres ordenados alfabéticamente** (no depender del
   orden interno `__wellOrd`), y los nombres se guardan en el npz. El consumidor
   (ResBO `loader_biopso.py`) valida `n_prod==4`/`n_inj==8` y hace fail con error si
   el deck no coincide.
4. `n_sims == nº de filas de simlog.csv` (misma corrida).
5. Si `WPR`/`WIR` no están pedidos en el SUMMARY del deck, `wellRawData` ya los rellena
   con ceros (comportamiento actual de `SimulationData`, verificado); el smoke test §6
   exige que NO sean todo ceros → si fallara, falta la keyword en el SUMMARY del caso.

---

## 3. Implementación de referencia (copiar literal)

### 3.1 Archivo nuevo: `src/ChronosUNB/storms_unb/core/simlog.py`

```python
# -*- coding: utf-8 -*-
"""
simlog.py

Registro por simulacion (contrato simlog_spec_v1, proyecto ResBO).

- log_sim_round(): agrega N filas (1 por agente evaluado) a <work_dir>/simlog.csv.
- save_sim_curves(): extrae las curvas WPR/WIR de TODAS las simulaciones de la
  corrida (real.span_data) a <work_dir>/sim_curves.npz.
- reset_simlog(): trunca el log al arrancar la corrida (re-ejecuciones seguras).

Aditivo: no toca decisiones del algoritmo ni otros artefactos.
Formatos congelados por docs/simlog_spec_v1.md -- NO cambiar sin versionar.
"""

import csv
from pathlib import Path

import numpy as np

HEADER = (['iteration', 'agent']
          + ['inje_{}'.format(i) for i in range(1, 9)]
          + ['prod_{}'.format(i) for i in range(1, 5)]
          + ['npv'])


def _path(work_dir):
    return Path(work_dir) / 'simlog.csv'


def reset_simlog(work_dir):
    _path(work_dir).unlink(missing_ok=True)


def log_sim_round(work_dir, iteration, pos, npv_vals):
    """Una fila por (agente, ronda). pos: (n_agents, n_inje + n_prod);
    npv_vals: (n_agents,) -- ambos en el orden de evaluacion de la ronda."""
    path = _path(work_dir)
    is_new = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(HEADER)
        for k, (x, v) in enumerate(zip(np.atleast_2d(pos),
                                       np.asarray(npv_vals))):
            writer.writerow([iteration, k]
                            + [repr(float(c)) for c in x]
                            + [repr(float(v))])


def save_sim_curves(real, work_dir):
    """Extrae WPR/WIR de todas las simulaciones (orden plano ronda-agente,
    identico al orden de filas de simlog.csv). Devuelve False si no hay datos."""
    handlers = [h for round_data in real.span_data for h in round_data]
    if not handlers:
        return False
    ref = handlers[0].data
    prod = sorted(w for w, t in zip(ref.wellNames, ref.wellTypes)
                  if t == 'PRODUCER')
    inj = sorted(w for w, t in zip(ref.wellNames, ref.wellTypes)
                 if t == 'INJECTOR')

    def curves(handler, var, wells):
        idx = {w: i for i, w in enumerate(handler.data.wellNames)}
        raw = handler.data.wellRawData[var]
        return np.vstack([np.asarray(raw[idx[w]], dtype='float64')
                          for w in wells])

    wpr = np.stack([curves(h, 'WPR', prod) for h in handlers])
    wir = np.stack([curves(h, 'WIR', inj) for h in handlers])
    np.savez_compressed(
        Path(work_dir) / 'sim_curves.npz',
        wpr=wpr,
        wir=wir,
        time_days=np.asarray(ref.timeSteps, dtype='float64'),
        sim_index=np.arange(len(handlers), dtype='int64'),
        producer_wells=np.array(prod),
        injector_wells=np.array(inj),
    )
    return True
```

### 3.2 Por qué el orden del npz casa con el simlog (verificado en el código)

`analyzeRealMNPV()` hace `real.span_data.append(n_agents*[None])` por ronda y
`update_current_data()` rellena `span_data[-1][span-1-offset]` en orden de agente;
recorrer `span_data` plano da exactamente (ronda 1 agente 0..k, ronda 2 …) = el
orden de `log_sim_round` por rondas. **No agregar ni reordenar handlers entre la
última `log_sim_round` y `save_sim_curves`.**

### 3.3 Puntos de inserción (5 algoritmos × 2 llamadas + 1 reset)

En CADA uno de estos archivos añadir el import (junta al de `convergence`, ya existe
uno idéntico):

```python
from ...storms_unb.core.simlog import log_sim_round, save_sim_curves, reset_simlog
```

**Llamada A — dentro del lazo**, justo DESPUÉS de que `npv_vals` queda definido y
ANTES de cualquier actualización de posiciones (el print `*** ... POSITIONS ***` es
el landmark; la matriz de posiciones de ese print ES la evaluada):

| Archivo | La matriz evaluada se llama | Insertar tras la línea |
|---|---|---|
| `pso/pso_wellcontrol.py` | `positions` | `data.fieldNPVcum[-1] for data in real.span_data[-1]` (~l.168) |
| `gwo/gwo_wellcontrol.py` | `wolf_pos` | ídem (~l.291) |
| `mfo/mfo_wellcontrol.py` | `moth_pos` | ídem (~l.254) |
| `foa/foa_wellcontrol.py` | `fly_pos` | ídem (~l.262) |
| `hgso/hgso_wellcontrol.py` | `positions` | ídem (~l.282) |

```python
            log_sim_round(self._work_dir, it, <matriz>, npv_vals)
```

(`it` es el contador 1-based del `for it in range(1, n_iterations + 1)` — ya existe
en los 5; `self._work_dir` ya existe en los 5.)

**Llamada B — al inicio y al final del ciclo de vida de la corrida, en `run()`
de cada algoritmo:**

```python
        # justo despues de la linea `real.span_data = []` del run():
        reset_simlog(self._work_dir)

        # justo antes de la linea `self.pickleProgress()` del run():
        try:
            save_sim_curves(real, self._work_dir)
        except Exception as e:
            print('aviso: no se pudieron guardar las curvas por simulacion ({})'.format(e))
```

El `try/except` es obligatorio: un fallo del logging NUNCA debe tumbar una corrida
de 10 h en MC2 (misma política que `curvas.guardar_curvas` en `run_generic.py`).

---

## 4. Algoritmo `random` (DoE uniforme) — para la coleta CRM (T-B5)

**No usar `direct/direct_run.py`:** es `ReactiveControl` (control múltiple/schedule)
y rompe el contrato de control constante de esta spec. Se crea un algoritmo nuevo
mínimo, copia del esqueleto de FOA:

- `src/ChronosUNB/storms_unb/random_search/__init__.py` (vacío)
- `src/ChronosUNB/storms_unb/random_search/random_wellcontrol.py`, clase
  `RandomSearchWellControl`, clonando `foa_wellcontrol.py` y cambiando SOLO:
  1. lectura de config: `n_agents = _cfg.get('randomAgents', 15)`,
     `n_iterations = _cfg.get('randomIterations', 4)`;
  2. en el lazo, reemplazar la generación alrededor del centro por uniforme puro:
     ```python
     fly_pos = lo + rng.uniform(size=(n_agents, nwell)) * (hi - lo)
     ```
     **sin ancla del agente 0 al control base y sin mover ningún centro** (es DoE,
     no búsqueda); eliminar la lógica de `center`.
  3. conservar: `log_convergence` (monitoreo en vivo de la interfaz), el par
     `reset_simlog`/`log_sim_round`/`save_sim_curves`, y el esqueleto `run()`
     (`pickleProgress`, `cleanTemp`, `optimum_data`).
- Registro en `interface/run_generic.py`:
  ```python
  from ChronosUNB.storms_unb.random_search.random_wellcontrol import RandomSearchWellControl
  ALGOS = {..., "random": RandomSearchWellControl}
  ```
- Config nueva `configs/egg_random_doe.yaml`: copia de `configs/egg_foa_protocol.yaml`
  con `randomAgents: 15` / `randomIterations: 4` (**= 60 sims por corrida**, caja
  `0–150 / 200–395` heredada; mismo `NPVParams` y `producerGrupcon`).

---

## 5. Artefactos y layout de la coleta S1 (T-B5)

Comando por brazo (WSL/MC2, desde la raíz del repo; **máx. 7 procesos**, sesión SSH
viva con `nohup ... & wait`, nunca `disown`; todo bajo `/mnt/c`):

```bash
python interface/run_generic.py --algo <mfo|gwo|pso|foa|random> \
  --config configs/egg_<algo>_protocol.yaml        # egg_random_doe.yaml para random
  --n-runs 5 \
  --csv  experiments/coleta_s1/<algo>/results.csv \
  --workdir /mnt/c/DAGO/coleta_s1/<algo> \
  --max-processes 7
```

Entregable esperado por corrida `run_XX/` (todo en la raíz del work_dir):

```
simlog.csv  sim_curves.npz  convergence.csv  results.csv(al CSV por brazo, fuera)
well_curves.csv  field_curves.csv  curves_meta.json  <algo>_progress.pkl
```

Y la cosecha local para ResBO (`experiments/coleta_s1/`) es la copia de estos
`run_XX/` completos. **Un `run_XX` sin `simlog.csv` o sin `sim_curves.npz` se
considera corrida fallida** (se re-lanza; `already_done_runs` lo vuelve a tomar).

---

## 6. Smoke test de aceptación (T-B3) — criterios binarios

Correr los smoketests existentes + crear `configs/egg_foa_smoketest.yaml`
(copia del de PSO con `foaFlies: 3` / `foaIterations: 2`):

| Algoritmo | Config | Filas esperadas simlog | npz |
|---|---|---|---|
| PSO | `egg_pso_smoketest.yaml` (4×1) | **4** | wpr (4,4,n_t), wir (4,8,n_t) |
| MFO | `egg_mfo_smoketest.yaml` (4×3) | **12** | (12,4,n_t) / (12,8,n_t) |
| GWO | `egg_gwo_smoketest.yaml` (3×2) | **6** | (6,4,n_t) / (6,8,n_t) |
| FOA | nuevo smoketest (3×2) | **6** | (6,4,n_t) / (6,8,n_t) |

Checks (script descriptivo basta; sugerido `interface/check_simlog.py`):

1. **Conteo**: filas de `simlog.csv` == agents×iters; `iteration` ∈ {1..iters};
   `agent` ∈ {0..agents−1} completos por ronda.
2. **Consistencia de orden**: `np.max` corriente por iteración del simlog == cada
   fila de `convergence.csv` (misma iteración, mismo gbest). Esto prueba que la
   matriz logueada fue la evaluada.
3. **Consistencia final**: mejor NPV del simlog (última ronda) == `npv_final` de
   `results.csv` para ese run_id, tolerancia relativa 1e-6 (con `spans: 1` ambos
   salen del mismo `fieldNPVcum[-1]`).
4. **npz**: shapes de §2 con `n_sims == filas del simlog`; `sim_index == arange`;
   `producer_wells`/`injector_wells` de tamaño 4/8; **`wpr` no todo-ceros y `wir`
   no todo-ceros** (si falla → faltan WWPR/W…IR en el SUMMARY del deck).
5. **Re-ejecución**: borrar la última fila de `results.csv` y relanzar el mismo
   run_id ⇒ el `simlog.csv` final vuelve a tener exactamente agents×iters filas
   (prueba que `reset_simlog` funciona).
6. **No-regresión**: `results.csv`, `convergence.csv`, `well_curves.csv` y el pkl
   del smoketest se generan igual que antes del parche.

---

## 7. Lista de no-regresión (NO tocar)

- `results.csv`, `convergence.csv`, `curvas.guardar_curvas`, `*_progress.pkl`:
  comportamiento y formato intactos (el simlog convive con ellos).
- Cualquier constante/semilla/ecuación de los 5 algoritmos (el logging no debe
  cambiar NI UN NPV respecto de una corrida previa — check 3 lo garantiza).
- `real.span_controls` neutro (50.0), `cleanTemp`, `pickleProgress`.

## 8. Consumidor del contrato (lado ResBO, para contexto del implementador)

`resbo/src/resbo/loader_biopso.py::load_run(work_dir, ...)` (T-R2) leerá:
`simlog.csv` → filas `SimRecord(algo, seed, iteration, controls[12], npv)` +
`prod_curve` del npz casada por `sim_index`; y `scripts/fit_crm.py` usará `wpr`
de la ventana tardía. Cualquier desviación de esta spec rompe el loader con error
explícito (no silencioso). Versionar cambios aquí y en el CHANGELOG de ResBO.
