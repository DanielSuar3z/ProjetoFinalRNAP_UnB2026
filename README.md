# ResBO — Otimização Bayesiana Informada por Conhecimento de Reservatório

Projeto Individual 2 — Redes Neurais e Aprendizado Profundo, 2026/2 (Prof. Díbio / UnB).
Autor: Jaime Daniel Beltrán Suárez (matr. 262122467).

**Idea central** (ver `Definicao_Projeto_RNAP_JaimeDaniel_ResBO.md`): sustituir parte de las
simulaciones físicas custosas (OPM Flow, Egg Model) por un **surrogate neural bayesiano (BNN)**
cuya función de adquisición es guiada por un **prior de domínio físico = conectividades
injetor–produtor de un modelo capacitância-resistência (CRM)** ajustado con los históricos de
producción de las propias corridas de los baselines (MFO/GWO/PSO/FOA del repo BIOPSO) — sin
simulaciones adicionales. El voidage balance es el caso degenerado (escalar) del prior.

Futuro repo público del curso: `ProjetoFinalRNAP_UnB2026`.

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
├── docs/estudo/               # notas de estudio dirigido (ver cronograma)
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

## Para MC2 (tarea 1.3+ del cronograma)

1. Copiar `src/resbo/` al repo BIOPSO (o crear el repo del curso y traer los algoritmos adaptados).
2. Implementar `OpmSimulator` como envoltorio del motor headless (`run_generic.py` / spawns de
   `mpirun -n 1 flow`) y registrar por sim: controles, VPL y curva de producción por poço.
3. Reglas MC2: máx 7 sims simultáneas, sesión SSH conectada (nohup + wait), nunca disown.
