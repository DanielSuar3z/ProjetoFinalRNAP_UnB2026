# `analysis/compare.py` + `scripts/compare_arms.py` — agregado de los 6 brazos (S5)

**Aparece en:** T-R6 (análisis y artigo). Consumidor: el artigo IEEE y los
slides — esta es la nota del código que produce LOS NÚMEROS del resultado.

## Para qué existe

Al final del experimento hay seis carpetas de brazo, cada una con la
`results.csv` que deja el harness de BIOPSO (`run_generic.py`):

```
coleta_s1/mfo/  coleta_s1/gwo/  coleta_s1/pso/  coleta_s1/foa/   (5 seeds, 15x6)
s3/bo_prior/    s3/bo_solo/                                        (N=20, 10+5x16)
```

El script las lee TODAS con el mismo código y produce los cuatro artefactos
del artigo: `arms_summary.csv` (descriptivos por brazo), `pairwise.csv`
(Mann–Whitney + r_rb), `arms_table.md` (las dos tablas en markdown listas
para pegar) y `arms_boxplot.png` (figura, si hay matplotlib).

## Las cuatro políticas honestas (cada una con test)

| política | por qué |
|---|---|
| `run_id` duplicado → **gana la última fila** | un re-intento tras corrida fallida se registra al final; el harness normaliza borrando la fila vieja, pero el analista no confía: el código lo garantiza |
| **AUC en simulaciones, no iteraciones** | la geometría de rondas es distinta por brazo (15×6 clásico vs 10+5×16 del BO): comparar en iteraciones regalaría 1.5× presupuesto al clásico. El conteo de filas del `simlog.csv` (función `_cum_sims_por_iteracion`) ES la moneda común, y cubre las dos geometrias (ronda 0 del DoE incluida) sin hardcodear agents/n_init |
| run de **rescate** (sin simlog) → NPV entra a la estadística, AUC = NaN y se cuenta en `runs_sin_curvas` | un dato rescatado vale para la comparación de NPV final; inventar su curva de convergencia no. Que la tabla HABLE de los runs sin curvas es parte del reporting |
| `convergence.csv` menciona una ronda que el simlog no registró → **historia descartada** (no mezcla) | artefactos medio-corruptos de una corrida con aviso de disco: mejor NaN visible que curva inventada |

La estadística es la MISMA función que validó la demo toy
(`analysis/metrics.py`: Mann–Whitney bilateral con empates promediados,
aproximación normal; rank-biserial = wins-x − wins-y sobre nx·ny). Con
n < 8 por brazo la aproximación normal es generosa: `pairwise.csv` trae
columna `aviso_n_bajo` y el CLI lo imprime — el artigo reporta el tamaño de
efecto (r_rb) como estadístico principal, el p como acompañante.

## Comparaciones: `--ref` canónico o todos contra todos

`--ref bo_prior` (default) produce las comparaciones que el projeto
pregunta: BO+prior vs cada ablación y vs cada baseline. Si el brazo ref no
está entre los pasados (p.ej. análisis intermedio con 3 brazos), cae a
todos-contra-todos sin que nadie tenga que recordar la regla.

## Uso

```bash
python scripts/compare_arms.py \
    --arm mfo=experiments/coleta_s1/mfo \
    --arm gwo=experiments/coleta_s1/gwo \
    --arm pso=experiments/coleta_s1/pso \
    --arm foa=experiments/coleta_s1/foa \
    --arm bo_prior=experiments/s3/bo_prior \
    --arm bo_solo=experiments/s3/bo_solo \
    --ref bo_prior [--threshold 1.6e7]
```

`--threshold` activa la columna `sims_al_umbral_mediana`
(`metrics.sims_to_threshold` por run, mediana por brazo): con qué sims
alcanzó cada brazo el mejor NPV de referencia. Sin el flag, la columna no
existe (no se reportan métricas a medias).

Sin brazos por CLI corre `--self-check`: fixture con verdad conocida
(bo_prior +2e6 sobre foa, geometría resbo 4+2×3=10 sims conviviendo con la
clásica 3×2, un rescate, un duplicado) que verifica dirección del efecto,
p pequeño, AUC a 10 vs 6 sims y los cuatro artefactos escritos. Como en
`fit_crm`/`train_bnn`: el pipeline se valida hoy contra fixture; mañana al
llegar los datos de MC2 solo cambian las rutas de `--arm`.
