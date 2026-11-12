# Guía: `src/resbo/database.py` — la memoria del proyecto

**Rol de una línea:** guarda cada simulación como una fila `(controles, NPV,
curvas)` en un CSV versionable, y ofrece "vistas" listas para ML: `(X, y)` para
el BNN y `(X, curvas)` para el CRM.

**Por qué existe:** el proyecto entero gira alrededor de una base de datos. El
BNN entrena con ella, el CRM se ajusta con ella, los brazos la pueblan y el
artículo se escribe con ella. Regla heredada de BIOPSO: **formato en disco
versionable y sin binarios** (CSV plano), para que git diff funcione y un
resultado de MC2 sea auditable línea a línea.

---

## Formato en disco

`<dir>/simulations.csv`, una fila por simulación:

| Columna | Contenido |
|---|---|
| `sim_id` | entero único en toda la base |
| `algo` | etiqueta del brazo: `mfo`, `gwo`, `pso`, `foa`, `bo`, `bo_prior` |
| `seed` | semilla de la corrida (repetición nº `seed` del brazo) |
| `iteration` | nº de iteración del algoritmo dentro de la corrida (`-1` = DoE inicial del BO) |
| `controls` | los 12 floats separados por `;` (el `;` interior evita chocar con la coma del CSV) |
| `npv_usd` | fitness de esa simulación |
| `prod_curve` | `n_prod*horizon` floats separados por `;`, o `;` solo si no hay curva |

Usar `;` como separador *dentro* de las celdas es la solución barata a "matriz
dentro de CSV" sin meter JSON. El formato es idempotente: `save()` → `load()`
reproduce la base exacta.

---

## `@dataclass class SimRecord`

Un registro = **una simulación**. `@dataclass` genera `__init__`/`__repr__` a
partir de los tipos declarados: es un "struct" de Python.

| Campo | Tipo/shape | Qué es y por qué se llama así |
|---|---|---|
| `sim_id` | int | Identificador secuencial de la simulación. `sim` = simulation. |
| `algo` | str | Algoritmo que la generó. Es la clave de filtrado de toda la BD. |
| `seed` | int | Semilla → identifica la "réplica" del brazo. Dos seeds = dos corridas independientes del mismo algoritmo. |
| `iteration` | int | En qué iteración del algoritmo se evaluó. `-1` se reserva para el DoE inicial del BO (antes del laço). |
| `controls` | `(12,)` float | El vector candidato evaluado `[rates×8, bhp×4]`. |
| `npv` | float | Fitness medido (VPL en USD). **El único escalar objetivo.** |
| `prod_curve` | `(n_prod*horizon,)` o `None` | Curva de producción **aplanada** (ravel). Se guarda plana para el CSV; se redimensiona a `(n_prod, horizon)` al leerla con `curves()`. `None` = simulación sin curva (todas las nuestras la traen). |

---

## `class SimDatabase`

La colección en memoria + persistencia. Piensa en ella como una tabla SQL en
miniatura: `add` (insert), `save` (commit), `load` (conectar),
`controls_npv`/`curves` (SELECT con vista).

### `__init__(self, path)`

| Atributo | Qué es |
|---|---|
| `self.path` | Carpeta donde vive la base (p. ej. `experiments/toy/`). `os.makedirs(..., exist_ok=True)` la crea si no existe. |
| `self.sim_file` | Ruta completa a `simulations.csv`. |
| `self.records` | `list[SimRecord]` — la tabla entera en RAM. Con ~10⁴ sims es trivialmente pequeña. |
| `self._next_id` | Contador para asignar `sim_id` sin colisiones. El guion bajo indica "interno"; al cargar se recalcula como `max(sim_id)+1`. |

### `add(algo, seed, iteration, controls, npv, prod_curve=None) -> SimRecord`

- Inserta en memoria (NO escribe disco). El disco solo se toca en `save()`.
- `np.asarray(controls, float)` — fuerza ndarray float aunque le pase listas.
- `np.asarray(...).ravel()` en la curva — aplanado canónico (ver `SimRecord`).
- Devuelve el registro creado (por si el llamador quiere su `sim_id`).
- Cada algoritmo llama aquí desde su `_eval()`: **agente evaluado = fila
  insertada**, garantizando que "toda evaluación es una simulación real
  registrada".

### `save()`

- Escribe TODO el historial a `simulations.csv` (sobrescribe). `newline=""` es
  el idiom de csv.writer para no duplicar saltos de línea en Windows.
- `map(repr, ...)` en vez de `str`: `repr` de un float de Python conserva
  todos los dígitos (round-trip exacto), `str` en versiones viejas los
  recortaba. El `;` vacío en `prod_curve` marca "sin curva" (no se puede
  dejar la celda vacía: ambiguo con curva de longitud 0).

### `@classmethod load(path) -> SimDatabase`

- Reconstruye la base desde CSV. Si el archivo no existe, devuelve base vacía
  (comportamiento "conectar a BD nueva" en vez de error).
- `header = next(rd)` — consume y descarta la cabecera (su orden está asumido).
- `_next_id = max(_next_id, sim_id+1)` — garantiza que un `add` posterior no
  reutilice un id existente.

### `controls_npv(algos=None) -> (X, y, recs)`

La vista para el **BNN**:

- `X` — `(n, 12)`: la matriz de diseño (cada fila, un candidato). `X` es la
  convención universal de numpy para la matriz de entradas.
- `y` — `(n,)`: los NPV. `y` = vector objetivo.
- `recs` — los `SimRecord` filtrados (metadatos: quien quiera seed/iteración).
- `algos` — lista de etiquetas a conservar; `None` = todas. Los brazos BO
  llaman `controls_npv(["mfo","gwo","pso","foa","bo"])` para entrenar con
  toda la historia.
- Ojo: con 0 filas devuelve `X` de shape `(0,0)` — los llamadores chequean
  `len(X)` antes.

### `curves(algos=None, horizon=130) -> (X, curves)`

La vista para el **CRM**:

- `curves` es una *lista* (no matriz) de arrays `(n_prod, horizon)`, uno por
  run. Lista porque preserva la estructura por pozo sin inventar un tensor 3D.
- `n_prod = len(prod_curve) / horizon` — la curva aplanada se "unde-ravel"
  dividiendo por el horizonte. El parámetro `horizon` se pasó explícito
  porque `SimRecord` no guarda la forma original (lección del bug del
  2026-11-06 documentado en CHANGELOG).
- Filtra registros con `prod_curve is None` — el CRM no puede usarlos.

---

## Cómo se conecta con el resto

- **Productores de filas:** `mfo/gwo/pso/foa._eval()` y `BayesOptimizer.run()`.
- **Consumidores:** `BayesOptimizer._train_surrogate` (vía `controls_npv`) y
  `toy_experiment.py` paso 2 (vía `curves` → `CRMModel.fit`).
- El paso final del toy (`db.save()`) es lo que hace que `experiments/toy/`
  sea re-ponible sin volver a correr nada.
