# Guía de código — ResBO (diccionario de variables, funciones y clases)

Una nota `.md` por cada archivo de código del proyecto. Cada nota explica, para
cada variable, función y clase: **qué es, para qué sirve, cómo se lee su nombre
(etimología) y por qué existe ahí**. Convenciones compartidas:

- **Formas (shape)**: se escriben como `(filas, columnas)` de numpy.
- **Convención de curvas del proyecto**: toda curva de producción se guarda como
  `(n_prod, horizon)` — pozos en filas, meses en columnas. (Bug histórico cazado
  el 2026-11-06: el mock las devolvía `(T, n_prod)` y rompía el CRM; ver CHANGELOG.)
- **Vector de controles** (12 dimensiones para Egg):
  `controls = [rate_1..rate_8 (sm3/d), bhp_1..bhp_4 (bar)]`
  — los primeros 8 son inyectores, los últimos 4 productores. Ese orden es
  sagrado: todo el código lo asume (`controls[:n_inj]` = rates, `controls[n_inj:]` = BHP).

## Mapa de módulos y orden de lectura sugerido

| # | Nota | Archivo documentado | Rol |
|---|------|---------------------|-----|
| 1 | [protocol_yaml.md](protocol_yaml.md) | `configs/protocol.yaml` | Única fuente de verdad de la caja, economía y presupuestos |
| 2 | [config.md](config.md) | `src/resbo/config.py` | Loader del YAML → objeto `Protocol` |
| 3 | [database.md](database.md) | `src/resbo/database.py` | `SimRecord`/`SimDatabase`: la memoria del proyecto |
| 4 | [simulator.md](simulator.md) | `src/resbo/simulator.py` | Contrato de simulador + `MockSimulator` + stub OPM |
| 5 | [mfo.md](mfo.md) | `src/resbo/mfo.py` | Moth-Flame Optimizer (polillas) |
| 6 | [gwo.md](gwo.md) | `src/resbo/gwo.py` | Grey Wolf Optimizer (lobos) |
| 7 | [pso.md](pso.md) | `src/resbo/pso.py` | PSO — réplica del repo BIOPSO |
| 8 | [foa.md](foa.md) | `src/resbo/foa.py` | FOA — mosca de la fruta, réplica del repo BIOPSO |
| 9 | [prior_crm.md](prior_crm.md) | `src/resbo/prior/crm.py` | CRM: ajuste de conectividades τ_ij y score del prior |
| 10 | [models_bnn.md](models_bnn.md) | `src/resbo/models/bnn.py` | Surrogate: ensemble de MLPs + MC-dropout |
| 11 | [bo_acquisition.md](bo_acquisition.md) | `src/resbo/bo/acquisition.py` | EI + mezcla con prior decaimiento |
| 12 | [bo_loop.md](bo_loop.md) | `src/resbo/bo/loop.py` | `BayesOptimizer`: el laço central |
| 13 | [analysis_metrics.md](analysis_metrics.md) | `src/resbo/analysis/metrics.py` | AUC, Mann-Whitney, rank-biserial |
| 14 | [toy_experiment.md](toy_experiment.md) | `scripts/toy_experiment.py` | Demo end-to-end de los 6 brazos |
| 15 | [test_core.md](test_core.md) | `tests/test_core.py` | Tests del núcleo (ejecutables sin pytest) |
| 16 | [loader_biopso.md](loader_biopso.md) | `src/resbo/loader_biopso.py` | Lector estricto del contrato simlog_spec_v1 + fixtures sintéticos |
| 17 | [scripts_train_y_validate.md](scripts_train_y_validate.md) | `scripts/train_bnn.py`, `scripts/validate_loader.py` | Entrenamiento standalone del BNN (gate NRMSE) + verificador de contrato |
| 18 | [scripts_fit_crm.md](scripts_fit_crm.md) | `scripts/fit_crm.py`, contracto `crm_v1.npz` (`prior/crm.py`) | Ajuste del prior CRM con gate R² por pozo (T-R4); tau con máscara de pozos |
| 19 | [scripts_export_deploy.md](scripts_export_deploy.md) | `scripts/export_deploy.py`, contracto `resbo_data_v1` | Empaquetado del deploy del brazo BO para BIOPSO (T-R5): experiencia + prior + surrogate con atestación de gates y manifest sha256 |

## Los archivos `__init__.py` (no tienen nota propia)

Los cinco `__init__.py` del paquete (`resbo/`, `prior/`, `models/`, `bo/`,
`analysis/`) solo **re-exportan nombres** para que se pueda escribir
`from resbo.bo import BayesOptimizer` en vez de `from resbo.bo.loop import ...`.
`resbo/__init__.py` además guarda `__version__ = "0.1.0"` (número de versión del
paquete, convención semántica major.minor.patch).

## Glosario transversal (aparece en varias notas)

| Símbolo / nombre | Se lee | Qué es |
|---|---|---|
| `NPV` | "en-pi-vi" | Net Present Value (VPL en español): suma descontada de (ingreso petróleo − costo líquido)·día. **Es la única función objetivo**; maximizar. |
| `τ` / `tau` | "tau" | Letra griega que el CRM usa para las **conectividades** inyector→productor. `tau_true` = la del mock (secreta); `tau` del CRM = su estimación. |
| `BHP` | "bi-aitch-pi" | Bottom Hole Pressure: presión de flujo de fondo del productor (bar). Menor BHP → más producción. |
| `rate` | — | Tasa de inyeccion de un inyector (sm3/d). |
| `lb`, `ub` | — | lower/upper bound: bordes de la caja de busqueda, vector de 12. |
| `pop` / `iters` | — | Tamano de poblacion y nº de iteraciones del brazo metaheuristico. |
| `seed` | — | Semilla del generador aleatorio: misma seed → misma corrida (reproducibilidad). |
| `brazo` / "arm" | — | Cada variante experimental que se compara (mfo, gwo, pso, foa, bo_prior, bo). |
| `algo` | — | Etiqueta de la columna de la BD que identifica el brazo que generó cada simulacion. |
| `EI` | — | Expected Improvement: funcion de adquisicion clasica de BO. |
| `LHS` | — | Latin Hypercube Sampling: DoE uniforme de 1 punto/cada franja de cada dimension. |
| `w(n)` | — | Peso del prior sobre la adquisicion, decae con las sims propias del BO (DynMeanBO). |
| `R²` | "r cuadrado" | Coeficiente de determinacion: cuanto de la varianza explica el ajuste (aqui, por pozo). |

> Como estudiar: abre la nota del modulo y, al lado, el codigo con la nota en
> otra ventana. Las notas estan ordenadas como el codigo (arriba→abajo).
