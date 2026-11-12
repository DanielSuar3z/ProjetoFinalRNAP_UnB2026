# Guía: `scripts/toy_experiment.py` — la demo end-to-end

**Rol de una línea:** ejecuta el pipeline COMPLETO del proyecto sobre el mock
en ~20 s: los 4 baselines → CRM → brazos BO → estadística → figura.
Es a la vez: smoke test del pipeline, generador de `experiments/toy/` y el
"laboratorio" donde se detectan los bugs de diseño (los tres hallazgos del
CHANGELOG nacieron aquí).

**Uso:** `cd resbo && python scripts/toy_experiment.py [--quick]`

---

## Cabecera del script (líneas 13–35)

```python
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
```
- `HERE`/`ROOT` — carpeta del script (`scripts/`) y raíz del proyecto
  (`resbo/`). Se calculan desde `__file__` (nunca desde el cwd) para que
  `python scripts/...` funcione lanzado desde donde sea.
- `sys.path.insert(0, .../src)` — el "truco" para importar `resbo` sin
  instalar el paquete (`pip install -e .` sería la alternativa formal).
- `# noqa` en los imports — instrucción al linter: "este import parece no
  usado pero sí se usa" (los usa `main()` más abajo; el linter miraba solo el
  orden de lectura).

---

## `def main()`

### Configuración

```python
proto = Protocol.load(os.path.join(ROOT, "configs", "protocol.yaml"))
if args.quick:
    proto.d["budget"]["metaheuristics"] = {"pop": 6, "iters": 3}
    proto.d["budget"]["bo"] = {"batch": 3, "iters": 4, "pool": 800}
    proto.d["n_seeds"] = 2
```
- `--quick` muta el dict interno del protocolo (`proto.d`) a presupuestos
  mínimos: ~20 s en vez de ~2 min. Solo presup/semillas: la física y la caja
  NO cambian.
- `outdir = experiments/toy/` — siempre se sobreescribe (la demo es
  re-generable; los resultados serían los de MC2 y van en otra carpeta).

### Estado compartido

```python
db = SimDatabase(outdir)
arms: dict[str, list[float]] = {"mfo": [], "gwo": [], "pso": [], "foa": [],
                                "bo_prior": [], "bo": []}
histories: dict[str, list] = {k: [] for k in arms}
```
- `db` — LA base de datos, compartida por todos los brazos (esta compartición
  es el experimento: BO hereda datos de los baselines).
- `arms` — por brazo, la lista de "best-NPV final" de cada seed. Es lo que
  alimenta la tabla de resultados y el Mann-Whitney. `arms` = brazos
  experimentales (traducción literal de la terminología de ensayos clínicos
  que usan los papers de benchmarking).
- `histories` — las curvas `(n_sims, best)` por seed (para futuras figuras de
  AUC; hoy la figura se pinta directo desde `db`).

### Paso 1 — los baselines

```python
for seed in range(proto.n_seeds):
    for algo, Klass in (("mfo", MFO), ("gwo", GWO), ("pso", PSO), ("foa", FOA)):
        sim = MockSimulator(proto, seed=42)      # misma "verdad" física
        best = Klass(proto, sim, db, seed=seed).run()
        arms[algo].append(best)
```
- `seed` exterior — la repetición del experimento (N seeds = N corridas).
- `Klass` — el constructor de clase en la tupla: la misma línea instancia y
  ejecuta los cuatro gracias al contrato común `(proto, sim, db, seed)`.
- **`MockSimulator(seed=42)` SIEMPRE** — crucial: la semilla del MOCK define el
  yacimiento (τ_true). Todos los brazos y seeds compiten sobre EL MISMO campo;
  la semilla del algoritmo (`seed=` del Klass) solo mueve sus aleatoriedades.
  Si se cambiara la seed del mock por seed, cada brazo optimizaría un campo
  distinto y la comparación no valdría nada.

### Paso 2 — el CRM

```python
Xc, curves = db.curves(base_algos)
crm = CRMModel(proto.n_inj, proto.n_prod).fit(Xc, curves)
corr = _corr_tau(crm.tau, sim.tau_true)
print(..., f"corr(tau_fit,tau_true)={corr:.3f}")
```
- Se ajusta SOLO con los baselines (el BO aún no corrió): es lo que habría
  disponible en S1 real antes de lanzar el BO.
- `_corr_tau` — helper: correlación de Pearson entre las 32 entradas de τ̂ y
  τ_true aplanadas. **Es un privilegio del mock** (con OPM no se conoce
  τ_true; allí el diagnóstico será el R² por pozo). En el toy dio 1.000 porque
  el mock es lineal-estacionario de manual.

### Paso 3 — los brazos BO

```python
for tag, use_prior in (("bo_prior", True), ("bo", False)):
    for seed in range(proto.n_seeds):
        bo = BayesOptimizer(proto, MockSimulator(proto, seed=42), db,
                            prior_model=crm if use_prior else None)
        bo.run(algo_tag=tag, seed=seed, base_algos=base_algos)
```
- **Las 3 líneas que definen la ablación**: mismo objeto, única diferencia
  `prior_model=crm` vs `None`. Ver discusión larga en la nota anterior.
- Tras cada run, se extraen las filas del brazo de esa seed
  (`[r for r in db.records if r.algo == tag and r.seed == seed]`) y se guarda
  el max en `arms`.

### Paso 4 — comparación y salida

```python
bp = np.array(arms["bo_prior"]); b0 = np.array(arms["bo"])
U, p = mannwhitney(bp, b0); rb = rank_biserial(bp, b0)
```
- La comparación ESTRELLA del proyecto (prior vs sin prior). Los baselines se
  printean como referencia pero no entran al test (no es un benchmark de
  algoritmos puros: ver la advertencia de presupuesto en el README raíz).
- `bp/b0` — abreviaturas bo_prior/bo (locales al main, vida corta).
- `db.save()` — el CSV queda en disco. Luego: tabla resumen + figura.

---

## Helpers privados del script

### `_plot_convergence(outdir, db)`

Pinta el best-so-far medio de cada brazo vs nº de sims (`convergence.png`):
- `matplotlib.use("Agg")` — backend sin pantalla (servidores/sin display).
  Import envuelto en `try/ImportError`: sin matplotlib el script sigue y no
  pinta (la demo no DEBE depender de librerías de figuras).
- Por brazo: agrupa NPVs por seed en `curves` (dict seed→lista),
  les aplica `np.maximum.accumulate` (best-so-far), les rellena el final con
  su último valor (padding: seeds cortas no hunden la media), y promedia.
- `colors` dict — paleta fija por brazo (mismo código de color en todas las
  figuras del proyecto). `/1e6` → NPV en millones USD (eje legible).

### `_corr_tau(tau_fit, tau_true)`

Pearson plano (`np.corrcoef(...)[0,1]`) sobre las matrices aplanadas. Nota:
`corrcoef` de dos arrays devuelve la matriz de correlación 2×2; `[0,1]` = la
entrada cruzada.

### `_write_summary(outdir, arms)`

CSV chiquito (`arms_summary.csv`: brazo, nº seeds, media/max/min) para leer la
tabla sin abrir notebooks.

---

## Cómo se conecta con el resto

Es el único consumidor de TODO a la vez — por eso es también el test de
integración no escrito. Cuando exista `run_generic.py`-style para MC2, este
script quedará como demo didáctica y nacerá `scripts/exp_coleta_s1.py` con la
misma estructura (brazos → CRM → brazos → análisis) pero `OpmSimulator` y
budgets reales.
