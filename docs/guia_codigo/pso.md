# Guía: `src/resbo/pso.py` — Particle Swarm Optimization (réplica del repo BIOPSO)

**Rol de una línea:** PSO de enjambre donde cada **partícula** es un candidato
de control; réplica **fieles a las fórmulas exactas** de
`storms_unb/pso/pso_wellcontrol.py` del repo BIOPSO (copia de referencia en
`docs/ref_biopso/pso_wellcontrol.py`).

**Por qué "réplica del repo" y no la versión canónica de EvoloPy:** los datos
con los que entrenará la red salen de las corridas de BIOPSO en MC2; si el
código local difiriera (aunque fuera en el decaimiento de `w`), el toy
simularía un algoritmo que no existe. Las diferencias contra EvoloPy están
listadas en el docstring del propio archivo.

---

## `class PSO`

### `__init__(self, protocol, simulator, db, seed=0, w_max=0.9, ...)`

Contrato idéntico a MFO/GWO (`protocol, simulator, db, seed`) + hiperparámetros
del repo:

| Atributo | Default | Qué es | Por qué ese nombre/valor |
|---|---|---|---|
| `self.n`, `self.iters`, `self.lb/ub`, `self.sim`, `self.db`, `self.seed` | del protocolo | Igual que la familia (ver [mfo.md](mfo.md)). | |
| `self.w` | **0.7 (CONSTANTE)** | Peso de inercia: cuánto conserva la partícula su velocidad anterior. Alto → vuela lejos (explora); bajo → escucha a los líderes (explota). **El repo NO lo decae** (a diferencia de EvoloPy 0.9→0.2): se replica tal cual. | `w` = inertial**w**eight, nombre canónico de Kennedy & Eberhart. Se lee de `protocol["pso"]["inertia"]` (default = default del repo, `psoInertia=0.7`). |
| `self.c1` | 1.5 | Coeficiente **cognitivo**: cuánto atrae el MEJOR PROPIO historial (`pbest`). | `c`ognitive — literal. YAML: `psoCognitive`. |
| `self.c2` | 1.5 | Coeficiente **social**: cuánto atrae el mejor de TODA la manada (`gbest`). | c**s**ocial... `c2` = segundo coeficiente. YAML: `psoSocial`. |

Nota: `w_max/w_min` del __init__ firmado en EvoloPy desaparecieron en esta
versión: el repo usa `w` fijo, y mantener parámetros muertos sería mentir en
la firma.

### `run(self) -> float` — línea a línea

```python
span = self.ub - self.lb
pos = self.lb + rng.random((self.n, d)) * span
pos[0, :] = 0.5 * (self.lb + self.ub)
```
- `span` — ancho de caja por coordenada.
- `pos` — matriz de posiciones `(n,12)`. **La partícula 0 se ancla al centro
  de la caja**: es el equivalente local del "control base del caso" del repo
  (en MC2 la partícula 0 arranca en los controles históricos del Egg; aquí el
  punto central es el análogo neutral). Convención compartida con la mosca 0
  de FOA y el gas 0 de HGSO: **el candidato base siempre se evalúa**.

```python
vel = (rng.random((self.n, d)) - 0.5) * span * 0.1
```
- Velocidades iniciales: uniformes en ±5% del ancho de caja. `(rand−0.5)·span`
  da ±(span/2); ×0.1 deja ±5%. Mismo truco del repo: el enjambre arranca
  "caminando", no "esprintando".

```python
pbest = pos.copy()
pval = np.full(self.n, -np.inf)
gbest, gval = pos[0].copy(), -np.inf
```
| Variable | Qué es | Por qué |
|---|---|---|
| `pbest` | `(n,12)` mejor posición **histórica de cada partícula** | **p**article **best**. La memoria individual es LA característica del PSO frente a MFO/GWO. |
| `pval` | `(n,)` mejor NPV histórico de cada partícula. `-inf` inicial: cualquier NPV real la mejora. | valores de pbest. |
| `gbest`, `gval` | mejor posición/valor **global** (de toda la manada, jamás) | **g**lobal **best**. |

**El laço (por iteración):**

1. **Evaluación dentro del laço** (no antes): cada partícula paga su
   simulación aquí → presupuesto total `pop·iters` (MFO/GWO pagan una ronda
   extra inicial; esta diferencia de contabilidad vive en los tests).
2. **Actualización de memorias:**
   ```python
   improved = val > pval
   pval = np.where(improved, val, pval)
   pbest[improved, :] = pos[improved, :]
   ```
   Vectorizado: `improved` es máscara booleana `(n,)`; `np.where` hace
   "si mejoró, nuevo valor; si no, conservo el viejo". Solo las partículas que
   mejoraron actualizan su memoria. (El repo lo hace idéntico, línea por línea.)
3. **gbest derivado de pbest** (no de la ronda actual):
   ```python
   j = int(np.argmax(pval));  if pval[j] > gval: gval, gbest = ...
   ```
   El líder global es la mejor MEMORIA, no la mejor posición del momento —
   semántica exacta del repo.
4. **Actualización de velocidad (la ecuación de PSO):**
   ```python
   vel = self.w * vel + self.c1 * r1 * (pbest - pos) + self.c2 * r2 * (gbest - pos)
   ```
   Tres fuerzas: inercia (sigo mi rumbo) + atracción a mi memoria + atracción
   al líder. `r1, r2` — ruido por coordenada que descorrelaciona las
   partículas. **Sin clip de velocidad** (el repo no usa Vmax; la caja se
   respeta clipando solo la posición):
   ```python
   pos = np.clip(pos + vel, self.lb, self.ub)
   ```
   Consecuencia: una partícula puede acumular velocidad grande y "rebotar"
   contra la caja — comportamiento observado también en MC2, replicado a
   propósito.

`return float(gval)` — el mejor NPV de la corrida.

### `_eval(self, x, it)`

El punto de contacto habitual: `sim.evaluate` + `db.add("pso", ...)`.

---

## Cómo se conecta con el resto

Sus filas engordan la BD (excitación del CRM: pasar de 96→168 runs al entrar
PSO+FOA en el toy) y son parte del pool de entrenamiento del BNN. La prueba de
fidelidad contra el repo es `tests/test_core.py::test_pso_matches_repo_semantics`
(parámetros, conteo de sims, ancla de la partícula 0).
