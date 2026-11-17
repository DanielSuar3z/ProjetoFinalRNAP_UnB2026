# Paso 05 — Inspección del BIOPSO real y congelación del contrato `simlog_spec_v1`

**Fecha:** 2026-11-17 · **Estado:** ✔ cerrado · **Artefacto:** `docs/estudo/simlog_spec_v1.md` (source of truth, con copia en `BIOPSO/docs/`)

## 1. La pregunta que detonó todo

El pipeline local (pasos 02–04) funciona con su propia base de datos
(`simulations.csv` escrita por el propio código). Pero el EXPERIMENTO real son
las corridas de BIOPSO en MC2. Antes de escribir una sola línea de ingesta "de
verdad", había que responder con el código delante: **¿qué guarda HOY BIOPSO
por cada simulación?**

## 2. Lo que encontró la inspección (y por qué no alcanza)

Revisión del clone local de BIOPSO (`src/ChronosUNB/storms_unb/…`, los 5
algoritmos de control de pozos + `interface/run_generic.py`):

1. **`results.csv` solo guarda el NPV escalar por corrida** (y metadatos del
   run) — no el vetor de controles de cada individuo.
2. Las **curvas de producción** (`well_curves.csv`) se guardan **solo del
   ganador** de la corrida.
3. El único lugar donde existieron los controles completos (el `.pkl` del
   objeto de control) **vive en MC2** y parte ya no existe.

Con eso, el dataset de entrenamiento del BNN sería de ~1 fila por corrida
(nada) en lugar de ~90 (la población × iteraciones). **Conclusión: hay que
instrumentar BIOPSO para que registre por simulación, y congelar ANTES un
contrato de datos**, porque dos repos separados (BIOPSO que escribe, ResBO que
lee) no pueden acordar formatos "a palabra".

## 3. El contrato `simlog_spec_v1` (lo que quedó congelado)

### a) Un `simlog.csv` por corrida, 15 columnas, una fila por simulación

```
iteration,agent,inje_1..inje_8,prod_1..prod_4,npv
```

y un `sim_curves.npz` con `wpr, wir, time_days, sim_index` + nombres de pozos
(históricos por pozo y paso de tiempo, para el CRM). Ambos en el `work_dir` de
la corrida. NPV = `NPVHandler.fieldNPVcum[-1]` — la definición del framework,
no una reimplementación.

### b) El invariante de orden

BIOPSO acumula los controles en `real.span_data`; las filas del log deben
respetar el **orden plano (ronda, agente)** EXACTO de esa matriz, sin reordenar
nada entre el último `log_sim_round` y `save_sim_curves`. Es invariante (no
recomendación): si se viola, controles y NPV se desalinean y el dataset queda
silenciosamente corrupto — el peor modo de fallar.

### c) El DoE aleatorio necesita un ALGORITMO NUEVO (`random`)

Para entrenar se requiere cobertura uniforme de la caja (Latin/uniform), pero
el `direct_run` existente usa `ReactiveControl` (controles que cambian con el
tiempo) → **violaría la premisa de controles constantes** del protocolo y del
CRM. Decisión: algoritmo `random` nuevo (esqueleto de FOA sin ancla: población
uniforme en la caja, sin inteligencia).

### d) Ruta de rescate para las 44 corridas viejas

No se puede recuperar lo no guardado: las corridas históricas aportan **1 fila
por corrida** (mejor NPV de `results.csv` + controles del ganador vía sidecar
`best_controls.json`). Es datos parciales CONSCIENTEMENTE etiquetados
(`iteration = -1`), no datos inventados.

### e) Smokes de verificación del contrato

PSO 4×1 = 4 filas, MFO 4×3 = 12, GWO 3×2 = 6, FOA smoketest por crear. Un
verificador binario (`check_simlog.py`) valida el contrato sobre un work_dir.

## 4. Por qué "congelado" y no "borrador"

Porque el paso siguiente (escribir el instrumentado en BIOPSO Y el loader en
ResBO) se haría EN PARALELO sobre dos repos distintos. Sin spec congelada,
cada lado inventaría su formato y la integración fallaría en el peor momento
(con la coleta de 1.920 sims ya corriendo). La spec vive en el repo público
(`docs/estudo/simlog_spec_v1.md`) con copia en BIOPSO, incluye la implementación
de referencia `core/simlog.py` y los checks 1–6 que definen "aceptable".

## ¿Qué quedó abierto?

- Implementar el contrato del lado BIOPSO (T-B1–T-B5) → paso 06.
- Implementar el loader estricto + la persistencia del BNN del lado ResBO
  (T-R2/T-R3) → paso 07.
- El mock local seguía con caja/economía genéricas → paso 08.
