# ResBO — Otimização Bayesiana Informada por Conhecimento de Reservatório

Projeto Individual 2 — Redes Neurais e Aprendizado Profundo, 2026/2 (Prof. Díbio / UnB).
Autor: Jaime Daniel Beltrán Suárez (matr. 262122467).

**Idea central** (ver `Definicao_Projeto_RNAP_JaimeDaniel_ResBO.md`): sustituir parte de las
simulaciones físicas custosas (OPM Flow, Egg Model) por un **surrogate neural bayesiano (BNN)**
cuya función de adquisición es guiada por un **prior de domínio físico = conectividades
injetor–produtor de un modelo capacitância-resistência (CRM)** ajustado con los históricos de
producción de las propias corridas de los baselines (MFO/GWO/PSO/FOA del repo BIOPSO) — sin
simulaciones adicionales. El voidage balance es el caso degenerado (escalar) del prior.

Repo público del curso: **`DanielSuar3z/ProjetoFinalRNAP_UnB2026`** (este directorio).

## Estructura

```
resbo/
├── configs/protocol.yaml      # caja de controles y presupuesto, IDENTICA en los 4 brazos
├── src/resbo/
│   ├── config.py              # loader del protocolo
│   ├── database.py            # ExperimentRecord / RunRecord: (controles, VPL, curvas) → CSV/JSON
│   ├── simulator.py           # Interfaz Simulator: MockSimulator (juguetes, local) y OpmSimulator (MC2)
│   ├── mfo.py / gwo.py        # metaheurísticas (patrón BIOPSO: agente = 1 simulación)
│   ├── pso.py / foa.py        # PSO y FOA fieles a pso/foa_wellcontrol.py del repo BIOPSO
│   ├── prior/crm.py           # ajuste CRM (τ_ij 8×4) + score de prior + voidage como degenerado
│   ├── models/bnn.py          # ensemble de MLPs con MC-dropout (numpy puro) → μ(x), σ_epistémica(x)
│   ├── bo/acquisition.py      # EI/UCB + término de prior con decaimiento (estilo DynMeanBO)
│   ├── bo/loop.py             # laço BO batch (q candidatos por iteración)
│   └── analysis/              # métricas (AUC, sims-al-umbral) + Mann-Whitney/rank-biserial
├── experiments/               # resultados versionados (results.csv) + BOARD.md de jobs MC2
├── scripts/toy_experiment.py  # DEMO end-to-end con MockSimulator (sin MC2, sin OPM)
├── scripts/train_bnn.py       # entrena BNN desde una BD (gate NRMSE_val < 5%) → bnn_*.npz
├── scripts/fit_crm.py         # ajusta el prior CRM (gate R² por pozo) → crm_*.npz (--self-check sin datos)
├── scripts/validate_loader.py # verificador del contrato simlog (--self-check sin datos)
├── docs/estudo/               # notas de estudio + contrato simlog_spec_v1 (source of truth)
├── docs/guia_codigo/          # una nota por archivo del repo, explicada a detalle
├── docs/bitacora/             # relato cronológico paso a paso (por qué estamos donde estamos)
└── CHANGELOG.md               # bitácora de decisiones (regla heredada de BIOPSO)
```

## Correr la demo (sin MC2)

```bash
cd resbo
python scripts/toy_experiment.py          # pipeline completo sobre MockSimulator (~1 min)
```

Genera `experiments/toy/` con results.csv de los brazos (MFO, GWO, PSO, FOA, BO+prior, BO),
figuras de convergência y la tabla Mann-Whitney. El MockSimulator imita la estructura del Egg
(8 inyectores, 4 productores, 130 meses, conectividades τ_true) para que el CRM y el BNN tengan
algo real que aprender: si el pipeline funciona aquí, el único cambio para MC2 es inyectar
`OpmSimulator`.

## Integración con BIOPSO/MC2 (estado 2026-11-17)

1. ✔ Contrato de datos congelado: `docs/estudo/simlog_spec_v1.md` (simlog.csv de
   15 columnas + sim_curves.npz por corrida, orden plano (ronda,agente)).
2. ✔ Lado BIOPSO: instrumentado en los 5 algoritmos + algoritmo `random` para el
   DoE + verificador (`BIOPSO` commits `eb10786`/`79b76fb`); coleta S1 = 1.920 sims.
3. ✔ Ingesta local: `src/resbo/loader_biopso.py` (ruta A completa / ruta B rescate),
   `scripts/train_bnn.py` con gate, persistencia `bnn_v1.npz` (sin pickle).
4. ⏳ Pendiente: transferir `coleta_s1` + `harvest_2026_11`, ajustar CRM real
   (`scripts/fit_crm.py`), entrenar `bnn_v1` y correr el brazo BO como Algorithm
   de BIOPSO (`resbo_wellcontrol.py`) en MC2.
5. Reglas MC2: máx 7 sims simultáneas, sesión SSH conectada (nohup + wait), nunca disown.
