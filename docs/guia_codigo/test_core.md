# Guía: `tests/test_core.py` — los tests del núcleo

**Rol de una línea:** 7 pruebas rápidas (<10 s) que protegen las invariantes
del pipeline sin necesidad de MC2 ni pytest (`python -m pytest tests -q` o el
mini-runner que usamos en consola).

**Filosofía:** cada test protege una decisión de diseño o un bug ya cazado.
Los tests no prueban "que el algoritmo sea bueno" (eso es el experimento),
prueban que el código significa lo que dice.

---

## Cabecera

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "protocol.yaml")
```
Igual que el script: ruta a `src/` para importar `resbo` sin instalar, y ruta
canónica al YAML (`CFG`).

## `def _proto() -> Protocol`

**Fixture** (preparador) compartido por todos los tests: carga el protocolo
real pero con presupuesto MINI (`pop=5, iters=2, n_seeds=1`) para que cada
test corra en instantes. Mutar `p.d` es el mismo truco que `--quick` del toy.

---

## Los tests, uno a uno

### `test_bounds_shape()`
**Invariante:** la caja tiene 12 dimensiones y `lb < ub` componente a
componente. Si alguien rompe el orden rates/BHP o pone un borde al revés,
esto revienta primero (barato y temprano).

### `test_mock_npv_finite()`
**Invariante:** el mock no produce NaN/Inf en el centro de la caja y devuelve
la curva con la convención `(n_prod, horizon)`.
- `np.isfinite(out["npv"])` — cazó de rebote el bug del `cumsum` sin `axis=0`
  (que reventaba con broadcast); quedó como test de forma.
- El `shape == (4, 130)` es el test de la convención que el bug del 06/11
  violaba. El comentario "punto central de la caja" recuerda que antes se
  pasaba un linspace mal formado (test mismo estaba roto).

### `test_crm_recovers_connectivity_trend()`
**La prueba de que el prior puede existir.** Genera 40 runs con rates
aleatorios MUY variados (uniforme en la caja = excitación máxima), ajusta el
CRM y exige `corr(τ̂, τ_true) > 0.4`.
- 40 runs ≥ `n_prod+1` (mínimo matemático) con holgura para que lstsq esté
  bien condicionado.
- El umbral 0.4 es conservador (el mock lineal da ~1.0): el test falla solo si
  el CRM está estructuralmente roto, no si tiene ruido. El string del assert
  "CRM no recuperó estructura (r=...)" fue calibrado: con curvas Orientation
  equivocadas (el bug de formato), la corr caía a −0.03 → el test cazó ese bug
  real.
- Nota: 0.4 y no 0.9 porque cuando lleguen datos REALES con ruido se pedirá
  otro umbral; este test es sobre el mock.

### `test_mannwhitney_shift()`
**Invariante del test estadístico:** dos normales claramente desplazadas
(media 10 vs 0, σ=1, n=20) deben dar `p < 0.05` y `r_rb > 0.5`.
Es un test del TEST: si el código de ranks/erfc tuviera un error, aquí
aparecería (con muestras casi separadas del todo es casi imposible fallar).

### `test_pso_matches_repo_semantics()`
**Prueba de fidelidad al repo BIOPSO** (código en `docs/ref_biopso/`):
- `pso.w, c1, c2 == (0.7, 1.5, 1.5)` — los defaults del repo, no de EvoloPy.
- `best == y.max()` — maximiza NPV directo (sin el truco costo=−NPV).
- `len(y) == pop·iters` — evalúa dentro del lazo (a diferencia de MFO/GWO que
  pagan la ronda inicial: `pop·(iters+1)`).
- Alguna fila de `iteration==0` está EXACTAMENTE en el centro de la caja — la
  ancla de la partícula 0 ("control base" del repo).
Usa `tempfile.TemporaryDirectory()` para una BD descartable (los tests no
ensucian `experiments/`).

### `test_foa_elitism_and_budget()`
**Prueba de las dos firmas de FOA:**
- `foa.step == 0.1` — default del repo.
- `len(y) == pop·iters` y `best == y.max()`.
- **El elitismo:** agrupa NPVs por iteración, toma el máximo por ronda y exige
  que la secuencia de máximos por ronda sea NO-DECRECIENTE
  (`np.all(np.diff(bsf) >= -1e-9)`). Esto solo se cumple porque la mosca 0 se
  evalúa en el centro que coincide con `gbest` — si alguien "arregla" FOA
  quitando la mosca 0, el test avisa.
- El `-1e-9` (en vez de 0) es tolerancia float: "no decreciente hasta error
  numérico".

### `test_auc_monotone()`
Test de humo de la métrica: una history trivialmente creciente da AUC > 0.
(No verifica valores exactos: la fórmula ya se revisó a mano; el test
principal de AUC es la propia figura del toy.)

---

## Cómo correrlos

```bash
cd resbo
python -m pytest tests -q      # si hay pytest
# o el runner inline que usamos en consola:
python - << 'EOF'
import sys, traceback; sys.path[:0] = ["tests", "src"]
import test_core as T
for n in [d for d in dir(T) if d.startswith("test_")]:
    try: getattr(T, n)(); print("PASS", n)
    except Exception: print("FAIL", n); traceback.print_exc()
EOF
```

Convenio de nombres: toda función `test_*` es detectada automáticamente por
pytest; las que no lo son (`_proto`) son helpers.
