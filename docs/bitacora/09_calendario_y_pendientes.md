# Paso 09 — Re-ancoraje a la fecha real de entrega (09/12/2026) y cola de tareas

**Fecha:** 2026-11-17 · **Estado:** ⏳ en curso (este archivo se actualiza a cada avance)

## 1. Por qué se re-ancora

El cronograma original (S1–S6, ~6 semanas) suponía la entrega "al final de
S6". El ejecutor fijó la **fecha real: 09/12/2026** (miércoles). Desde hoy
quedan **~3 semanas**, y el estado real del proyecto no es "S1 empieza": es
"S1 terminada de hecho". Re-ancorar no es reescribir el plan: es mapear las
tareas y gates vigentes a fechas concretas (`Cronograma_ResBO_Jaime.md`,
§"Calendario al 09/12").

## 2. Lo que ya está cerrado al 17/11 (y por qué S1 no consume más semanas)

| Hito | Evidencia |
|---|---|
| Protocolo congelado y REALINEADO al EGG_TEST | `configs/protocol.yaml` (paso 08) |
| Repo público funcionando | `DanielSuar3z/ProjetoFinalRNAP_UnB2026` (paso 04) |
| Contrato de datos congelado | `simlog_spec_v1` (paso 05) |
| BIOPSO instrumentado + verificador + runbook | commits `eb10786`/`79b76fb` (paso 06) |
| **Coleta S1 lanzada** | `COLECTA_S1.md`: 22 runs, 1.920 sims, ≈30–35 h de pared (PC nuevo) |
| Ingesta + gates del lado ResBO | loader + `train_bnn.py` + 11/11 tests (pasos 07–08) |

El presupuesto de la tabla del cronograma se actualizó a la coleta REAL
(4 algos × 5 seeds × 90 + 2 DoE × 60 = 1.920 sims, ya en curso) en lugar de
los 3.600 estimados de MFO+GWO a 30 agentes.

## 3. Calendario al 09/12 (síntesis; el detalle vive en el cronograma)

| Ventana | Fase | Entregable |
|---|---|---|
| 17–19/11 | cierre S1 | **transferencia** `coleta_s1` + `harvest_2026_11` al local; `check_simlog.py` sobre todo lo recibido |
| 20–23/11 | S2 | **T-R4** `scripts/fit_crm.py` (τᵢⱼ 8×4, R² por pozo ≥ 0.7) + **`bnn_v1.npz` real** pasando el gate NRMSE < 5% ⛔ |
| 24–28/11 | S3 | **T-R5** `resbo_wellcontrol.py`: BO como Algorithm de BIOPSO consumiendo `bnn_v1.npz` + `crm_v1.npz`; smokes; lanzamiento de los 2 brazos BO (N=20, contingencia 16→12) |
| 29/11–01/12 | S3→S4 | análisis de los 6 brazos (4 baselines de la coleta + BO+prior + BO-solo): VPL, AUC, sims-al-umbral, Mann-Whitney + rank-biserial; figuras |
| 02–06/12 | S4+S5 | artigo IEEE 6–8 pág. + relatório 2 pág. uso de IA + repo citation-ready |
| 07–09/12 | S6 | **ENTREGA 09/12** a aprender3 + 4 slides + ensayos; oral en la aula subsequente |

Recortes explícitos por la compresión (documentados, no silenciosos): barrido
de sensibilidad del decaimiento (3.4) → opcional post-gate S3; corrida extra de
robustez (5.4) → fuera. Perillas de contingencia en orden: n_seeds BO
20→16→12; lo último que se toca son figuras y ablación.

## 4. Cola de tareas (orden ineludible)

1. **[CUELLO] Transferencia de datos** MC2/PC-nuevo → local: `experiments/coleta_s1/`
   y `harvest_2026_11/`. Sin esto, T-R4 y `bnn_v1` no empiezan. (MC2 SSH caído
   desde 07/11; la coleta se corrió en el PC nuevo según el runbook — falta
   confirmar con el ejecutor la ruta exacta de origen.)
2. **T-R4** `scripts/fit_crm.py`: ajuste CRM sobre los `sim_curves.npz` de la
   coleta, gate por pozo R² ≥ 0.7, con la caveat ya documentada: en el mock
   `prod_curve` es tasa de LÍQUIDO, en el npz real es `wpr` → el fit real usa
   las series reales (consistencia CRM↔datos, no CRM↔mock).
3. **`bnn_v1` real**: `train_bnn.py` sobre la base de la coleta (1.920 sims +
   rescate) → gate NRMSE < 5%. El toy ya demostró que el gate muerde.
4. **T-R5** `resbo_wellcontrol.py`: el brazo BO como algoritmo de BIOPSO
   (`--algo resbo` en `run_generic.py`), cargando `bnn_v1.npz` + `crm_v1.npz`,
   misma caja/presupuesto que la coleta; correr en MC2 los brazos BO+prior y
   BO-solo (la ablación) con las reglas de cómputo (7 procesos, nohup+wait).
5. **Análisis + artigo (T-R6)**: pipeline estadístico sobre los 6 brazos,
   figuras, artigo IEEE, relatório de IA, repo final, slides.
6. **Hygiene/pendingites de gobernanza**:
   - rotación del `GH_TOKEN` con acceso al repo privado (pendiente desde S1);
   - **sync de la proposta** (`.md/.html/.pdf/.docx`): todavía dice "MFO e GWO"
     → debe decir MFO/GWO/PSO/FOA (4 brazos, formulação actual);
   - **decisión S2 (gate GCPRM-lite)** que solo el ejecutor puede confirmar:
     actualizar el CRM a variantes de tiempo por arista SOLO si
     Δcorr(score,NPV) > 0.1 **y** R² ≥ 0.7 sobre los datos reales.

## 5. Reglas que siguen vigentes (heredadas, sin excepción)

- Cambios → fila en `CHANGELOG.md`; pasos → archivo aquí.
- Nada del repo privado del lab al repo público (ni una línea).
- Seeds y configs versionados; `results.csv` al repo, `runs/` no.
- Gates S2/S3 innegociables: el artigo puede ser más chico, no puede tener
  resultados sin base de datos congelada detrás.
