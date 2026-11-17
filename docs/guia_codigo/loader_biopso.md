# `src/resbo/loader_biopso.py` — lector estricto de artefactos BIOPSO

**Qué es**: la frontera entre el motor del equipo y ResBO. Convierte run_XX/ de
BIOPSO en SimRecords de nuestra SimDatabase. Es la implementación de referencia
del contrato `docs/estudo/simlog_spec_v1.md`.

**Por qué existe**: el plan separa "el motor escribe, ResBO lee". Este archivo
es donde ese contrato se cobra: si BIOPSO escribe algo que no cumple la spec,
el loader **falla ruidosamente** (SpecViolation) en vez de tragar datos
basura silenciosamente.

## Conceptos

### `SpecViolation(RuntimeError)`
Cualquier desacuerdo con la spec: cabecera distinta, shape imposible, orden
roto, wpr todo-ceros. La regla de diseño: **nunca advertir, siempre explotar**.

### `SPEC_HEADER` / `NPZ_KEYS`
Las 15 columnas del simlog.csv y las 6 claves del sim_curves.npz, copiadas
literalmente de la spec. Si la spec sube de versión, este es el primer sitio
que cambia.

## Las dos rutas de `load_run`

### Ruta A — `simlog.csv` + `sim_curves.npz` (corridas con parche)
```
_read_simlog:  cabecera exacta -> [(iteracion, agente, controles(12), npv)]
_read_npz:     wpr (n_sims, n_prod, T) + validaciones de consistencia
por fila i:    db.add(algo, seed, iter_i, ctrl_i, npv_i, prod_curve=wpr[i])
```
El invariante que se valida: `sim_index == arange(n_sims)` — el orden de filas
del simlog ES el orden del npz (planos (ronda, agente) de `real.span_data`).

### Ruta B — rescate (corridas viejas SIN parche)
De esas corridas solo sobreviven las curvas de la MEJOR política
(`well_curves.csv`), el NPV (`results.csv`, que vive en el PADRE del run en la
convención BIOPSO) y —con esfuerzo del harvest— los controles ganadores del
`*_progress.pkl` en el sidecar `best_controls.json`.
Por eso produce **1 fila por corrida** con `iteration = -1` (marca de rescate).
Sin sidecar ni argumento `controls`, el loader explica exactamente qué falta.

## `load_harvest(root, db)`
Recorre `<root>/<algo>/run_XX/` y etiqueta por padre conocido
(`mfo/gwo/pso/foa/hgso/random/stosag/bo/bo_prior`); si el padre no es un tag,
usa el nombre del propio run.

## `make_synthetic_run(out_dir, sim, protocol, ...)`
Fixture: genera runs que **cumplen la spec al 100%** evaluando con el
MockSimulator real. `rescue=True` emite el formato viejo (well_curves.csv +
results.csv con el run_id derivado del nombre de carpeta + best_controls.json).
Usos: los 3 tests nuevos de `tests/test_core.py` y el `--self-check` de
`scripts/validate_loader.py`. Con esto el pipeline quedó probado HOY, sin
esperar ni a MC2 ni al parche del equipo.
