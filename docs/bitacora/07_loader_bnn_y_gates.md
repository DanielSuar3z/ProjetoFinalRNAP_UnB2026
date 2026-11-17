# Paso 07 — Loader estricto, persistencia BNN (`bnn_v1.npz`) y gates de calidad (T-R2/T-R3)

**Fecha:** 2026-11-17 · **Commit:** `15cfdbd` · **Estado:** ✔ cerrado

## 1. Principio del loader: desconfiar hasta que el contrato se pruebe

`src/resbo/loader_biopso.py` lee work_dirs de BIOPSO y los convierte en la base
de datos local. No es un lector tolerantemente "flexible": es un **verificador
de contrato que falla duro** (`SpecViolation`) ante:

- cabecera de `simlog.csv` que no coincide byte a byte con `simlog_spec_v1`;
- shapes mal formados (controles que no suman 12, npz con dimensiones
  inconsistentes con las filas);
- `wpr` todo-ceros en el npz (síntoma típico de una corrida muerta o de un
  guarda `try/except` del paso 06 tragándose un error);
- orden de filas que no respeta el invariante (ronda, agente).

Razón de ser: el dataset alimenta un BNN que a su vez guía simulaciones de
7 minutos. Un desalineamiento silencioso controles↔NPV no "ensucia" el
resultado: lo **invierte** con apariencia de normalidad. Mejor un raises a las
10 de la mañana que 30 horas de pared gastadas en un dataset podrido.

### Dos rutas de ingesta

| Ruta | Qué lee | Para qué sirve |
|---|---|---|
| **A (completa)** | `simlog.csv` + `sim_curves.npz` del work_dir | Los 90 puntos por corrida de la coleta S1 (controles + NPV + históricos por pozo → entrenamiento BNN **y** ajuste CRM). |
| **B (rescate)** | `well_curves.csv` del ganador + `results.csv` del directorio PADRE (layout real de BIOPSO) + sidecar `best_controls.json` | Las corridas históricas (`harvest_2026_11`): 1 fila/corrida con `iteration = -1` como marca de origen. |

### El fixture que cierra el ciclo sin MC2

`make_synthetic_run(...)` genera un work_dir sintético que CUMPLE la spec
(usando el MockSimulator real para producir NPVs y curvas). Así el loader, el
entrenador y el verificador se prueban de punta a punta en local: cuando
lleguen los datos reales, la única variable que cambia es la fuente.

## 2. Persistencia del BNN: contrato `bnn_v1.npz`, sin pickle

`BnnEnsemble.save_npz()/load_npz()` serializa pesos, hiperparámetros y
normalizadores en un `.npz` con schema versionado (`bnn_v1`). **Nada de
`pickle`**: (a) pickle ejecuta código al cargar — inaceptable para un archivo
que viaja entre máquinas del lab; (b) el brazo BO correrá del LADO BIOPSO
(paso 09, T-R5) en un entorno distinto al de entrenamiento, y solo un formato
neutro garantiza que el modelo sobreviva al viaje.

## 3. `train_bnn.py` y el gate de calidad

Pipeline: carga BD → split 80/20 de semilla FIJA → entrena ensemble (media +
MC-dropout) → métricas por split (RMSE, **NRMSE normalizada por el rango del
dataset**, R², **cobertura 2σ** de los intervalos) → **gate: `NRMSE_val < 5%`
del rango de NPV; si falla, exit code 2** ("no lanzar el brazo BO con este
modelo").

- El gate materializa el "MAPE < 5%" del cronograma, adaptado a la realidad de
  que el NPV del campo es firmado (puede cruzar cero; el rango es el
  denominador honesto).
- La cobertura 2σ valida que la INCERTIDUMBRE es real (sin ella, el EI es
  decoración): ~0.95 esperado.
- `validate_loader.py --self-check`: 4 auto-pruebas (ruta A OK, ruta B OK,
  cabecera corrupta RECHAZADA, wpr todo-ceros RECHAZADO).

## 4. Validación empírica del día (con los datos de entonces)

- Cosecha sintética estilo harvest: **253 sims, gate PASS (NRMSE val 1.7%)**.
- Base toy del mock de la época: **236 sims, PASS (1.1%)**.
- Tests: **11/11** (4 nuevos: roundtrip del loader, rescate, rechazos,
  roundtrip save/load BNN).
- Notas de la guía de código 16–17 escritas el mismo día.

⚠ Nota: esos NRMSE corresponden al mock PRE-paso 08. Con la economía real, el
toy volvió a fallar el gate (ver paso 08) — y eso, paradójicamente, es buena
señal del pipeline.

## ¿Qué quedó abierto?

- La caja/economía del mock y de `protocol.yaml` NO coincidían con el caso
  EGG_TEST que la coleta está corriendo → T-R1, el paso 08.
- El ajuste CRM sobre curvas reales (T-R4) necesita el `sim_curves.npz` de la
  coleta transferido → paso 09.
