# Paso 06 — Lado BIOPSO: instrumentación simlog, DoE `random`, runbook de coleta y rescate (T-B1–T-B5)

**Fecha:** 2026-11-17 · **Commits en BIOPSO:** `eb10786` (motor) y `79b76fb` (rescate) · **Estado:** ✔ cerrado y verificado en el clone local

## 1. T-B1/T-B2 — Instrumentar los 5 algoritmos SIN tocar su lógica

`src/ChronosUNB/storms_unb/core/simlog.py` es la implementación de referencia
del contrato: `log_sim_round(...)` (una ronda de la población → filas del
simlog), `save_sim_curves(...)` (curvas por pozo al npz) y `reset_sim_log()`.

Inserción en `pso_wellcontrol.py`, `gwo_wellcontrol.py`, `mfo_wellcontrol.py`,
`foa_wellcontrol.py` y `hgso`: **puramente aditiva, ~8 líneas por archivo, con
guarda `try/except`**. Diseño deliberado: si el logging falla en medio de una
corrida de 90 simulaciones de 7 minutos, la corrida NO muere — el experimento
de 30 horas de pared vale más que dos columnas. El verificador (T-B3) detecta
el hueco después.

## 2. T-B4 — El algoritmo `random` para el DoE

`RandomSearchWellControl`: esqueleto de FOA sin ancla ni elitismo — población
uniforme en la caja (15 agentes × 4 iteraciones = 60 sims), registrada por el
mismo simlog. Entra en `ALGOS` de `run_generic.py` como `--algo random`, y de
paso se portatiliza la interfaz (`BIOPSO_SRC`/`--case-file` apuntan al repo).
Config: `configs/egg_random_doe.yaml`.

## 3. T-B3 — Verificador y pruebas

- `interface/check_simlog.py`: **verificador binario** (exit 0/1) de los checks
  1–4 del spec sobre un `work_dir` — cabecera exacta, conteo de filas =
  agents×iters, orden plano (ronda,agente), npz consistente con las filas.
- `tests/test_simlog.py`: 8 pruebas, todas pasan.
- Configs de smoke: `egg_foa_smoketest.yaml` (3×2=6) completa la batería
  PSO 4×1=4 / GWO 3×2=6 / MFO 4×3=12.

## 4. T-B5 — El runbook de la coleta (`docs/COLECTA_S1.md`)

Receta completa para el PC nuevo, en orden:

1. `git pull` + `test_simlog.py` (TODO OK).
2. **Smokes**: 1 corrida de cada smoketest, verificar filas exactas con
   `check_simlog.py` (la tabla de números esperados está en el runbook).
3. **Coleta S1**: 5 seeds × {MFO, GWO, PSO, FOA} con `egg_*_protocol.yaml`
   (15×6 = 90 sims/corrida) + 2 corridas DoE `egg_random_doe.yaml` (60 sims).
   **20×90 + 2×60 = 1.920 simulaciones**, ≈30–35 h de pared a 7 procesos.
   Un brazo por sesión con `nohup … & wait` (nunca disown), `--max-processes 7`.
4. **Cosecha**: loop que corre `check_simlog.py` sobre los **22 runs** y copia
   los artefactos aprobados (checks 5/6 del runbook = integridad contra
   `results.csv` y contra el conteo del config).

Los configs del S1 real quedaron así fijados como "la moneda" del experimento:
caja 0–150 sm³/d / 200–395 bar, `NPVParams` 45/6/0/2/0/0.08, ventana 130
meses, controles base 79.5/395. Este hecho es el que obliga al paso 08.

## 5. El rescate `harvest_2026_11` (commit `79b76fb`)

Manifiesto de lo rescatable de las corridas históricas en MC2: **42 resultados
+ 2 `.pkl` que aún conservan curvas (18 sims)**. Los 9 jobs bajo `/home/dago`
ya NO existen — se reprodujo el riesgo H-07 (datos que solo viven en un path
personal y desaparecen con el usuario). Justifica post-mortem la política del
contrato: registrar por simulación, desde el algoritmo, al repositorio de
experimentos.

## ¿Qué quedó abierto?

- Transferir `coleta_s1` + `harvest_2026_11` al entorno local (cuello actual,
  ver paso 09).
- El loader que consume esos artefactos del lado ResBO → paso 07.
