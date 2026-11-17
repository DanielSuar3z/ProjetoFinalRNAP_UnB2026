# `scripts/train_bnn.py` y `scripts/validate_loader.py` — el taller de datos

Dos scripts de comando (no librerías): entrenan y verifican fuera del lazo de
optimización, sobre cualquier cosecha de simulaciones.

## `train_bnn.py` (T-R3)

**Flujo**: `--db <SimDatabase>` o `--harvest <arbol BIOPSO>` → split 80/20 con
`--split-seed` fija → entrena `BnnEnsemble` → mide → guarda artefactos → sale
con código 2 si el gate falla (integrable en CI/scripts).

### El gate: `NRMSE_val < 5% del rango`
`nrmse = rmse / (max(y) − min(y))` con el rango del dataset COMPLETO. Es la
materialización del "MAPE < 5%" del cronograma para una magnitud que puede ser
negativa (el VPL): un error relativo a |y| individual castigaría mal los
casi-ceros, el rango normaliza contra la escala real del problema.
El JSON guarda `nrmse_rango_def` para que nadie reinterprete el número.

### Métricas de sanidad extra
- `r2` en val — calidad bruta.
- `cobertura_2sigma_val` — fracción de validación dentro de μ±2σ del ensemble.
  Debe rondar 0.9–1.0; si se dispara, las σ están mal calibradas y el EI del
  brazo BO mentirá.

### Artefactos (`experiments/models/<tag>.*`)
- `<tag>.npz` — pesos + estadísticos de normalización (`BnnEnsemble.save_npz`,
  sin pickle). Este archivo es el "modelo entrenado" que el brazo BO de BIOPSO
  (T-R5) cargará junto al prior CRM.
- `<tag>_metrics.json` — números reproducibles (hiper, semillas, gate).
- `<tag>_scatter.png` — pred vs real en val, para el informe.

### Por qué `--algos`
El dataset puede mezclar etiquetas (`mfo,gwo,random` para el DoE inicial, o
todo el harvest). Entrenar sobre el subconjunto correcto es parte del protocolo.

## `validate_loader.py` (verificación de T-R2)

- `<ruta>` — valida un run_XX/ o un árbol de cosecha completo con el loader
  estricto y imprime conteos/shapes/rangos por etiqueta. Salida != 0 si algo
  viola la spec. Es el "smoke test de contrato" que el equipo puede correr
  tras aplicar el parche simlog.
- `--self-check` — genera fixtures sintéticos (ruta A, ruta B, cabecera
  corrupta, wpr todo-ceros) y exige que los válidos pasen y los inválidos
  exploten. No necesita datos externos.

## Validación actual (2026-11-17)

Fixtures sintéticos (253 sims, 5 etiquetas + 1 rescate): gate PASS con
NRMSE_val 1.7% (mock). Base toy real (236 sims de la demo): NRMSE_val 1.1%.
Ambos son MockSimulator — suavizado; el gate se pondrá serio con datos Flow.
