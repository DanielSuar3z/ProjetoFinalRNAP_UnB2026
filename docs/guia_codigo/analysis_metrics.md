# Guía: `src/resbo/analysis/metrics.py` — cómo se mide "quién ganó"

**Rol de una línea:** métricas de convergencia (best-so-far, AUC, sims-al-
umbral) y los dos tests estadísticos del protocolo (Mann-Whitney U y tamaño de
efecto rank-biserial), todo sin scipy.

**El marco:** el protocolo del artículo compara 4+2 brazos con **N=20 seeds**
cada uno. Con muestras no paramétricas (los NPV no son normales) la comparación
regla es: Mann-Whitney para el p-valor + rank-biserial para el tamaño del
efecto (un p-valor sin efecto no dice nada en ingeniería).

---

## Métricas de convergencia

### `best_so_far(npvs) -> (n,)`

```python
np.maximum.accumulate(npvs)
```
Convierte la secuencia de NPV "en orden de llegada" en la curva de **máximo
corriente**: posición k = mejor NPV tras k simulaciones. Monótona no-decreciente
por construcción. Es la curva que se grafica y de la que derivan las otras dos.
`np.maximum.accumulate` = "running max" (fórmula del acumulador tipo cumsum con
max en vez de +).

### `auc(history, n_total=None) -> float`

- `history` — la lista `(n_sims, best_npv)` que deja `BayesOptimizer`.
- **Área bajo la curva best-so-far normalizada** = promedio temporal del
  best-so-far: `(Σ v_k·Δn_k) / n_total`. Un brazo que converge PRONTO acumula
  más área que otro que llega al mismo final tarde. `auc` = area under curve
  (adaptada: aquí el dominio es el nº de sims, no el eje x de una función).
- `step = np.diff(np.concatenate([[0], n]))` — los incrementos entre registros
  (las sims que aporta cada punto de `history`; en el toy es 1 a 1, pero la
  fórmula aguanta batches).

### `sims_to_threshold(history, threshold) -> float`

Primera simulación en la que best-so-far ≥ `threshold`. Si nunca lo alcanza,
devuelve `n_total+1` (penalización "peor que todos"): métrica tipo
"tiempo-hasta-objetivo", la más comprensible para un lector de ingeniería.

---

## Tests estadísticos (sin scipy)

### `mannwhitney(x, y) -> (U, p)`

Test **U de Mann-Whitney** bilateral con aproximación normal. Pasos del código:

```python
allv = np.concatenate([x, y])
order = allv.argsort(); ranks[order] = np.arange(1, n+1)
```
1. **Rankeo conjunto**: se ordenan las dos muestras juntas y cada valor recibe
   su rango 1..N (la esencia no-paramétrica: solo importan ORDENES, no valores
   — por eso resiste outliers y distribuciones raras).
```python
uniq, inv = np.unique(allv, return_inverse=True)
if len(uniq) < len(allv):   # hay empates
    for k in range(len(uniq)): ranks[inv == k] = ranks[inv == k].mean()
```
2. **Empates al promedio**: valores iguales reciben el rango medio (corrección
   estándar; con datos simulados "exactamente iguales" que sí ocurren).
```python
rx = ranks[:len(x)].sum()
U1 = rx - len(x)*(len(x)+1)/2
U2 = len(x)*len(y) - U1
U = min(U1, U2)
```
3. **Estadístico U**: `U1` = cuántas veces un valor de x supera a uno de y
   (forma de conteo de "wins"), expressado vía suma de rangos (identidad
   algebraica clásica). `U` = el menor de los dos (estadístico bilateral).
```python
mu = nx*ny/2
sigma = sqrt(nx*ny*(nx+ny+1)/12)
z = (U - mu)/(sigma + 1e-12)
p = erfc(abs(z)/sqrt(2))
```
4. **p-valor por aproximación normal**: bajo H₀ (distribuciones iguales), U es
   ~N(μ=nx·ny/2, σ²=nx·ny·(nx+ny+1)/12). `erfc` = complementary error function;
   `erfc(|z|/√2)` = 2Φ(−|z|) = p bilateral exacta para la normal.
   Nota: sin corrección de empates en σ (aproximación aceptada: empates raros
   y N pequeño hace la corrección irrelevante aquí); para el artículo final
   con N=20 puede validarse contra `scipy.stats.mannwhitneyu` (scipy es
   opcional justamente para esto).

**Lectura:** p < 0.05 ⇒ "las distribuciones de best-NPV de los dos brazos son
distinguibles". Con N=2 (toy) es esperable que NO lo sea — el test solo
demostró estar vivo.

### `rank_biserial(x, y) -> float ∈ [-1, +1]`

```python
for a in x:
    for b in y:
        if a > b: U1.append(1)
        elif a < b: U2.append(1)
wins = sum(U1) - sum(U2)
return wins / (len(x)*len(y))
```
- Tamaño de efecto **rank-biserial** (r_rb): conteo directo de "wins" por pares
  (x supera a y ⇒ +1, pierde ⇒ −1, empate ⇒ 0) dividido por el nº total de
  pares. `r_rb = +1` ⇒ x SIEMPRE supera a y (en el toy eso pasó: bo > bo_prior
  en todos los pares). `0` ⇒ indistinguibles.
- O(n·m) bucles: N=20 ⇒ 400 comparaciones, nada.
- Se reporta junto al p porque: p dice "hay diferencia", r_rb dice "cuánta".

---

## Cómo se conecta con el resto

`toy_experiment.py` las usa todas en el paso 4 (comparación `bo_prior` vs
`bo`). En MC2, el script de análisis del artículo producirá la tabla completa
(4 brazos × N=20) con estas mismas funciones.
