# Bitácora de trabajo — ResBO (paso a paso, explicado a detalle)

Este directorio es el **relato cronológico** del proyecto: qué hicimos en cada
paso, POR QUÉ, qué descubrimos y qué decidió cada hallazgo. Nació de la
petición explícita del ejecutor (2026-11-17): *"escribir un .md con cada uno de
los pasos que estamos haciendo, explicados a detalle para entender lo que
estamos haciendo"*.

## Cómo se diferencia de los otros documentos de registro

| Documento | Formato | Pregunta que responde |
|---|---|---|
| `CHANGELOG.md` (raíz) | tabla de filas | ¿QUÉ se decidió y CUÁNDO? (registro compacto, regla heredada de BIOPSO) |
| `docs/guia_codigo/` (18 notas) | una nota por archivo | ¿CÓMO funciona este archivo, línea por línea? |
| `docs/estudo/` | notas de estudio | ¿QUÉ dice la literatura sobre este tema? |
| **`docs/bitacora/` (esto)** | narrativa por paso | ¿POR QUÉ llegamos al estado actual y qué aprendimos en el camino? |

Si quieres ENTENDER el proyecto (no solo operarlo), lee esta bitácora en orden.

## Los pasos

| # | Paso | Fecha | Estado |
|---|---|---|---|
| [01](01_contexto_y_anclaje.md) | Contexto, ancla bibliográfica y adopción del proyecto ResBO | oct–nov 2026 | ✔ |
| [02](02_nucleo_resbo_sobre_mock.md) | Núcleo del pipeline local sobre el MockSimulator (v0.1.0) | 2026-11-06 | ✔ |
| [03](03_replicas_pso_foa.md) | PSO y FOA como réplicas exactas del repo BIOPSO | 2026-11-07 | ✔ |
| [04](04_repo_publico.md) | Repo público `ProjetoFinalRNAP_UnB2026` y reglas de publicación | 2026-11-16 | ✔ |
| [05](05_inspeccion_biopso_contrato.md) | Inspección del BIOPSO real y congelación del contrato `simlog_spec_v1` | 2026-11-17 | ✔ |
| [06](06_lado_biopso_t_b1_a_b5.md) | Lado BIOPSO: instrumentación simlog, DoE `random`, runbook de coleta, rescate | 2026-11-17 | ✔ |
| [07](07_loader_bnn_y_gates.md) | Loader estricto, persistencia BNN (`bnn_v1.npz`) y gates de calidad | 2026-11-17 | ✔ |
| [08](08_t_r1_realineacion_egg_test.md) | T-R1: realineación del protocolo local al caso EGG_TEST real | 2026-11-17 | ✔ |
| [09](09_calendario_y_pendientes.md) | Re-ancoraje a la fecha real de entrega (09/12/2026) y cola de tareas | 2026-11-17 | en curso |
| [10](10_t_r4_crm_offline.md) | T-R4 preparado offline: script `fit_crm.py`, contracto `crm_v1.npz` y gate por pozo | 2026-11-17 | ✔ |
| [11](11_t_r5_brazo_bo_biopso.md) | T-R5: el brazo BO cruza la frontera — BIOPSO corre la Optimización Bayesiana | 2026-11-17 | ✔ |
| [12](12_t_r6_agregado_seis_brazos.md) | T-R6 (parte 1): la máquina de los números — agregado de los 6 brazos para el artigo | 2026-11-17 | ✔ |
| [13](13_decisiones_anexo_gates_datos.md) | Decisiones del ejecutor: anexo de alcance en la proposta, gate S2 confirmado y plan de datos | 2026-11-17 | ✔ |
| [14](14_rescate_mc2_y_reanclaje_s1.md) | Rescate `harvest_2026_11` verificado en local; descubrimiento: S1 no había corrido — re-ancoraje del calendario | 2026-11-17 | ✔ |

## Convenciones de la bitácora

- Cada archivo termina con una sección **"¿Qué quedó abierto?"**: ahí vive el
  hilo del siguiente paso.
- Los números citados (NPVs, NRMSE, p-valores) son los REALES de las corridas
  de la fecha; si un paso posterior los invalida, se anota arriba del archivo
  ("⚠ superado por paso NN") y no se borra: el registro incluye los errores.
- Nombres de tareas: `T-Rn` = tareas del repo ResBO, `T-Bn` = tareas del lado
  BIOPSO. Las dos numeraciones son independientes.
