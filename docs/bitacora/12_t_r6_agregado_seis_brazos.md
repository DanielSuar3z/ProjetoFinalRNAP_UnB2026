# Paso 12 — T-R6 (parte 1): la máquina de los números — agregado de los 6 brazos

**Fecha:** 2026-11-17 · **Estado:** ✔ cerrado en código (13/13 tests, self-check
verde); la tabla real se produce el día que `compare_arms.py` apunte a los
árboles de la coleta + S3

## 1. La decisión que gobierna el paso

Con T-R5 cerrado, la cola crítica quedó reducida a lo que solo el acceso
puede desbloquear (datos + MC2). Quedaba una pregunta incómoda: cuando las
seis `results.csv` existan, ¿quién convierte 6×~20 NPVs en LAS tablas del
artículo? Si ese código se escribe el 05/12 con presión, se escribe mal. La
decisión del día: construir hoy el agregado completo (T-R6/S5), auto-validado
contra fixture con verdad conocida, en la misma filosofía de `fit_crm` y
`train_bnn` — mañana solo cambian las rutas de `--arm`.

## 2. Qué se construyó

- `src/resbo/analysis/compare.py` — el módulo: `load_arm` (directorito de
  brazo o `results.csv` directo), `resumen_brazo` (n, media, mediana, sd,
  min, max, AUC, runs sin curvas), `pairwise` (Mann–Whitney + rank-biserial
  del propio `analysis/metrics.py`, sin reimplementar) y `run_compare`
  (escribe `arms_summary.csv`, `pairwise.csv`, `arms_table.md` y
  `arms_boxplot.png`).
- `scripts/compare_arms.py` — CLI con `--arm nombre=ruta` repetible,
  `--ref bo_prior` (comparaciones canónicas: el brazo objetivo contra cada
  ablación y cada baseline; si el ref no está, todos contra todos),
  `--threshold` opcional y `--self-check`.

## 3. Las cuatro políticas honestas (y por qué cada una es un test)

**3.1 `run_id` duplicado → gana la última fila.** El harness normaliza una
re-ejecución borrando la fila vieja, pero un borrado a mano que salió mal
deja dos filas. El analista no puede permitirse creer en el proceso: el
último registro gana, y un fixture lo verifica (el valor del re-intento,
9.9e7, está en el brazo).

**3.2 La AUC se integra en SIMULACIONES, no en iteraciones.** Esta es la
trampa más fina del experimento: `convergence.csv` indexa por iteración, y
las iteraciones NO valen lo mismo en cada brazo — el clásico quema 15 sims
por iteración, el BO 5 (+10 del DoE inicial). Una AUC "por iteración"
regalaría 1.5× presupuesto aparente a MFO/GWO/PSO/FOA.
`_cum_sims_por_iteracion` lee el `simlog.csv` de cada run y cuenta filas: el
conteo ES la moneda, cubre la geometría clásica y la del Anexo A (ronda 0)
sin hardcodear nada. El fixture monta las dos geometrias a la vez y exige
AUC a 10 sims (BO) vs 6 (clásico).

**3.3 El run de rescate entra a la estadística pero no a la curva.** Los
runs del harvest sin `simlog` (ruta B, 1 fila por sidecar) tienen NPV final
confiable pero ninguna historia de convergencia. Excluirlos castigaría dos
veces al dato (ni NPV ni curva); inventarles una curva sería fraude.
Entran al descriptivo y a Mann–Whitney; su AUC es NaN y la tabla lleva una
columna `runs_sin_curvas` que los hace VISIBLES. El artículo puede decir
"n=20, 1 sin curvas" sin que nadie tenga que reconstruir la historia.

**3.4 Artefactos inconsistentes → NaN, no mezcla.** Si un `convergence.csv`
menciona una iteración que el `simlog` del mismo run no registró (disco,
aviso de logging a mitad de corrida), la historia SE DESCARTA entera. Un NaN
visible en la tabla es honesto; una curva cosida de dos orígenes no.

## 4. La estadística no se reescribió

`pairwise` llama a las MISMAS `mannwhitney`/`rank_biserial` que validaron la
demo toy (paso 09): U bilateral con empates promediados, aproximación
normal. Con n por debajo de 8 la aproximación es generosa con el p — por
eso `pairwise.csv` trae columna `aviso_n_bajo` y el CLI la imprime.
Doctrina del projeto para la tabla: **r_rb como estadístico principal, p
como acompañante** (con N=20 por brazo y contingencia 16→12, el tamaño de
efecto es lo que sobrevive a la discusión).

## 5. Verificación de la sesión

- `python scripts/compare_arms.py --self-check`: bo_prior sintético +2e6
  sobre foa → `r_rb=+1.000, p=3.9e-03`, geometría resbo AUC a 10 sims,
  clásico a 6, rescate contado, duplicado resuelto, cuatro artefactos
  escritos (el boxplot salió con matplotlib local).
- `tests/test_core.py`: **13/13** — la prueba nueva (`test_compare_arms_selfcheck`)
  además del self-check verifica por API que la columna del umbral aparece
  SOLO con `--threshold` (nada de métricas a medias en la tabla).

## 6. Efecto en el calendario

S5 ya no es un bloque de trabajo "después de S3": el día que los árboles
lleguen de MC2, la tabla del artigo sale de un comando:

```
python scripts/compare_arms.py --arm mfo=... --arm gwo=... --arm pso=... \
    --arm foa=... --arm bo_prior=... --arm bo_solo=... --ref bo_prior
```

Lo que sigue debiendo T-R6: la sync de los docs de propuesta (todavía dicen
"MFO e GWO" donde el experimento real corre cuatro baselines) — es edición
de un documento presentado al profesor y se decide con el ejecutor, no en
silencio; y el esqueleto del artigo, que empieza cuando haya números.
