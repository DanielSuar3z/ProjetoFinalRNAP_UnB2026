# Guía: `src/resbo/models/bnn.py` — el surrogate bayesiano

**Rol de una línea:** red neuronal que aproxima `NPV ≈ f(controles)` **y dice
cuánto no sabe** (σ). El BO vive de esa σ: sin incertidumbre no hay
exploración.

**Por qué "BNN" si no es una red bayesiana de verdad:** es la versión
pedagógica/local del modelo. La incertidumbre se aproxima con dos trucos
reconocidos en la literatura:
- **Deep Ensembles** (Lakshminarayanan 2017): varias redes entrenadas desde
  cero con datos bootstrap → su DESACUERDO estima incertidumbre **epistémica**
  (lo que el modelo no sabe).
- **MC-Dropout** (Gal & Ghahramani 2016): mantener el dropout ACTIVO en
  inferencia, muestrear muchas veces → la VARIANZA entre muestreos estima
  incertidumbre **aleatoria** (ruido).

En MC2 se podrá sustituir por una versión en torch **sin tocar el laço**: el
contrato es `fit(X, y)` + `predict(X) → (mu, sigma)`.

---

## `def _relu(x)`

`np.maximum(x, 0)`: la función de activación ReLU (Rectified Linear Unit).
Privada (guion bajo) porque solo la usa `_MLP`.

---

## `class _MLP` — una red_feedforward mínima

Red `[d → 32 → 32 → 1]`. Privada: solo `BnnEnsemble` la construye.

### `__init__(self, d, h=(32,32), seed=0, p_drop=0.05)`

| Atributo | Shape | Qué es y por qué se llama así |
|---|---|---|
| `self.W` | lista de matrices | **Weights**: pesos de cada capa. Lista porque hay una matriz por capa (`(in,out)`). |
| `self.b` | lista de vectores | **biases** (términos independientes). |
| `self.p_drop` | float | **p**robabilidad de **drop**out: fracción de neuronas que se "apagan" aleatoriamente en cada mini-batch de entrenamiento. Regularizador y, paradójicamente, la fuente de la incertidumbre MC-dropout. 0.05 = 5% (suave: con redes pequeñas, dropout alto destruye la señal). |
| `self.mW/vW/mb/vb` | ceros como W/b | Estado del optimizador **Adam**: **m** = media móvil del gradiente (momentum), **v** = varianza móvil del gradiente (escala adaptativa por parámetro). Sufijos W/b = a qué parámetro acompañan. |
| `self._t` | int | **t**imestep de Adam: cuenta mini-batches, necesario para la corrección de sesgo. |
| `self._rng` | RNG | Generador privado de la red: cada miembro del ensemble tiene su secuencia reproducible. |

Inicialización `normal(0, sqrt(2/(fan_in+fan_out)))`: Gaussian scaled — evitar
que las señales se apaguen/exploten en profundidad (variante de He/Glorot).

### `_forward(self, X, train) -> acts`

Pasa `X` por las capas guardando las activaciones para el backprop:
- `acts` — lista de activaciones por capa (entrada incluida), se reutiliza en
  backprop.
- `train=False` ⇒ dropout DESACTIVADO (inferencia determinista por muestreo);
  `train=True` ⇒ dropout ACTIVO y se guarda la máscara `m`.
- Última capa **sin ReLU** (salida lineal: es regresión, no clasificación).
- Dropout "inverted": `a*m/(1−p_drop)` re-escala para que la media no cambie
  entre train e inference (técnica estándar).

### `predict(self, X, train=False) -> (n,)`

`_forward(X, train)[-1].ravel()`: la última activación (escalar por punto).
Con `train=True` deja el dropout activo → cada llamada a un mismo `X` da una
salida ligeramente distinta (el muestreo MC).

### `fit(self, X, y, epochs=400, lr=3e-3, batch=64, wd=1e-4)`

Entrenamiento manual (sin frameworks). Parámetros:

| Parámetro | Qué es |
|---|---|
| `epochs` | Pasadas completas del dataset. |
| `lr` | **l**earning **r**ate: tamaño del paso de gradiente (3e-3, moderado). |
| `batch` | Tamaño de mini-batch. |
| `wd` | **w**eight **d**ecay: L2 (penaliza pesos grandes; regulariza). |

Dentro del bucle (mini-batch `Xb, yb`):

1. **Forward con dropout** → `acts`.
2. **Gradiente de la última capa:**
   ```python
   delta = (acts[-1] - yb) / max(n,1) * 2.0
   ```
   Derivada de la pérdida MSE (`L = mean((ŷ−y)²)`): el gradiente es
   `2(ŷ−y)/n`. `delta` (δ) es el "error" de la capa que se retropropaga.
   El `max(n,1)` evita 0/0 en batches vacíos (defensivo).
3. **Backprop** (de capa en capa hacia atrás):
   ```python
   gW[i] = a_prev.T @ delta + wd*W[i]      # gradiente de pesos (+ L2)
   gb[i] = delta.mean(...)                 # gradiente de biases
   delta = (delta @ W[i].T) * (z > 0)      # propaga el error × ReLU'
   ```
   `a_prev.T @ delta` — la regla de la cadena matricial (gradiente de W =
   entradaᵀ·error). `z>0` es la derivada de ReLU (1 si activada, 0 si no); si
   la capa tenía dropout, se multiplica además por la máscara `m/(1−p_drop)`
   (las neuronas "apagadas" no propagan).
4. **Adam** (actualiza pesos):
   ```python
   m = b1*m + (1-b1)*g          # momentum (b1=0.9)
   v = b2*v + (1-b2)*g²         # varianza móvil (b2=0.999)
   mh = m/(1-b1^t); vh = v/(1-b2^t)   # corrección de sesgo (arranca en 0)
   P -= lr * mh/(sqrt(vh)+eps)  # paso adaptativo
   ```
   `eps=1e-8`: evita dividir por 0. `b1/b2`: decaimientos estándar de Adam.

Devuelve `self` (fluent).

---

## `class BnnEnsemble` — el "modelo" que ve el BO

n_models redes `_MLP` entrenadas sobre bootstrap; en inferencia, las n·mc
salidas (bootstrap × MC-dropout) dan la distribución.

### `__init__(self, n_models=5, hidden=(32,32), p_drop=0.05, seed=0)`

| Atributo | Qué es |
|---|---|
| `n_models` | nº de miembros del ensemble (5). Más = mejor σ pero más cómputo. |
| `hidden` | arquitectura oculta compartida por todos. |
| `self.y_mu, self.y_sd` | media y **s**tandard **d**eviation de `y` (NPV). |
| `self.x_mu, self.x_sd` | ídem por columna de `X`. `None` hasta `fit_norm_stats`. |

### `fit_norm_stats(self, X)` y `_norm(self, X)` — normalizar X

Guarda media/desviación por dimensión y `(X−mu)/sd`. **Por qué:** las 12
dimensiones tienen escalas absurdamente distintas (rates ~10³ vs BHP ~10²);
sin normalizar, una red con gradiente descendente no converge. `+1e-12` en sd
evita dividir por 0 si una columna fuera constante (p. ej. un inyector
siempre al mismo rate). `_norm` lanza RuntimeError si no se llamó antes al
stats (orden obligatorio documentado en el error).

### `fit(self, X, y, epochs=300, lr=3e-3)`

1. Normaliza `y` también (`y_mu/y_sd`): el NPV es de ~10⁸, entrenar una red sobre
   target de 10⁸ es numéricamente horrible; se entrena sobre `yn` ~N(0,1) y se
   re-escala al predecir.
2. Por cada miembro k:
   ```python
   idx = rng.integers(0, n, size=n)     # bootstrap
   m = _MLP(..., seed=seed*100+k)
   m.fit(Xn[idx], yn[idx], ...)
   ```
   **Bootstrap** = muestra con reemplazo: cada red ve un subconjunto distinto
   (≈63% únicos) → desacuerdo entre redes = incertidumbre epistémica. Semillas
   distintas (`seed*100+k`) para inicializaciones distintas.

### `predict(self, X, mc_samples=16) -> (mu, sd)`

El corazón bayesiano:

```python
for m in self.models:
    for _ in range(mc_samples):
        preds.append(m.predict(Xn, train=True))   # dropout ACTIVO
P = np.vstack(preds)        # (n_models*mc_samples, n_pts)
mu = P.mean(axis=0)*y_sd + y_mu
sd = P.std(axis=0)*y_sd
```
- `mc_samples` — nº de pases MC-dropout por red. 16×5 = 80 predicciones por
  candidato.
- `P` — matriz de todas las predicciones. `axis=0` agrega SOBRE los modelos
  (media y desviación por candidato).
- `mu` — la media, devuelta a la escala original (×y_sd+y_mu). Es el "μ(x)".
- `sd` — la dispersión de las 80 predicciones, re-escalada. Es la **σ(x)** que
  come la adquisición EI.
- El `(mu, sd)` se reescala con `y_sd` porque σ debe vivir en las unidades de
  y para sumar con EI (que usa unidades de NPV).

**Lectura física de la σ:** donde hay datos → bootstrap elige subconjuntos
similares y ya entrenados → σ baja. Donde no hay datos → las redes divergen →
σ alta → EI explora ahí. Así la σ *es* el motor de exploración del BO.

---

## Cómo se conecta con el resto

`BayesOptimizer._train_surrogate` lo entrena cada iteración sobre
`db.controls_npv(pool)` y `model.predict(cand)` alimenta a
`expected_improvement`. Ver [bo_loop.md](bo_loop.md) y
[bo_acquisition.md](bo_acquisition.md).
