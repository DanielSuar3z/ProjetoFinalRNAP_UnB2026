# Guía: `configs/protocol.yaml` — el protocolo experimental

**Rol de una línea:** única fuente de verdad de QUÉ se optimiza (caja), con qué
se mide (economía), cuánto se puede gastar (presupuestos) y cómo pesa el prior.
Cambiar este archivo = cambiar el experimento; por eso NINGÚN módulo tiene
ninguno de estos números escrito en el código.

**Por qué YAML y no un .py:** los parámetros del artículo tienen que poder
revisarlos/probarlos lectores y colaboradores sin leer código; YAML es
legible, diff-eable y no ejecutable (imposible "colar" lógica).

---

## Bloque de identidad del caso

```yaml
project: ResBO
reservoir: egg-mock          # MockSimulator local; egg-opm en MC2
n_injectors: 8
n_producers: 4
horizon_months: 130
```

| Clave | Qué es |
|---|---|
| `project` | Nombre (etiqueta en figuras/CSV). |
| `reservoir` | Qué "yacimiento" define este protocolo: `egg-mock` = el juguete local; en MC2 pasará a `egg-opm`. Se registra aquí porque los resultados NO son comparables entre los dos. |
| `n_injectors` / `n_producers` | 8/4, la estructura del Egg Model (E1-INJ × PROD en el modelo de Schouborg). Definen la dimensión del problema: 8+4 = **12**. |
| `horizon_months` | Vida evaluada: 130 meses ≈ ~11 años, la historia de pronóstico del Egg. Cada simulación cubre TODO el horizonte con controles constantes (decisión de scope: no hay control recendente, a diferencia de StoSAG/Simplex). |

---

## `bounds` — la caja de decisión

```yaml
bounds:                      # = injeRateMin/Max y prodBhpMin/Max del caso EGG_TEST
  rate:  [0.0, 150.0]        # sm3/d por inyector
  bhp:   [200.0, 395.0]      # bar por productor

base_controls:               # = injectorValues/producerValues del deck
  rate: 79.5
  bhp: 395.0
```

- Escalares POR TIPO de pozo: `Protocol.bounds` los expande a los vectores
  `lb/ub` de 12. No se usa una lista de 12 pares porque en el Egg todos los
  inyectores comparten límites.
- **T-R1 (2026-11-17):** la caja ya NO es un rango genérico "tipo Egg" — es
  EXACTAMENTE la del caso `EGG_TEST` de BIOPSO (`configs/egg_*_protocol.yaml`:
  injeRateMin/Max=0/150, prodBhpMin/Max=200/395). Así el mock, los brazos
  locales y la coleta S1 comparten la misma moneda y el dataset entrenado es
  utilizable por el código local sin renormalizar. Antes: 500–3000/150–350.
- `base_controls` — el control base del deck (inyectores 79.5 sm³/d,
  productores BHP 395). En BIOPSO las versiones reales de PSO y FOA anclan su
  partícula/mosca 0 al `injectorValues/producerValues` del caso; la réplica
  local lo lee de aquí (`Protocol.base_controls`, fallback = centro de caja).
  Nótese que BHP 395 es el BORDE superior de caja: el ancla del deck es un
  punto pobre a propósito (los algoritmos deben mejorar el control base).
- Física del rango: muy bajo rate → el reservorio se depleta y la presión (y
  el NPV) se hunde; muy alto → breakthrough temprano y coste de manejar agua.
  El óptimo está en el medio, con compensaciones por pozo.

---

## `economics` — cómo se convierte producción en dinero

```yaml
economics:                   # = NPVParams del caso EGG_TEST (USD/sm3), via NPVHandler
  oil_revenue: 45.0          # oilRevenue
  water_cost: 6.0            # waterCost
  water_injection_cost: 2.0  # waterInjectionCost
  annual_discount: 0.08      # discountRate
```

Son LOS CUATRO números del bloque `NPVParams` que BIOPSO pasa a
`NPVHandler` (STORMSPlugin/trunk/ChronosUNB/storms_unb/core/npv_handler.py)
en todos los configs del protocolo S1. La versión anterior (price_oil 60 /
cost_liquid 8 / discount 0.10, dos términos) se jubiló en T-R1 para que
"NPV local" y "fieldNPVcum de BIOPSO" signifiquen lo mismo.

| Clave | Qué es | Por qué ese valor |
|---|---|---|
| `oil_revenue` | Ingreso por m³ de **aceite** producido. | 45 USD/m³ de EGG_TEST (es rentabilidad neta del aceite, no precio de mercado). |
| `water_cost` | Coste por m³ de **agua producida**. | 6 USD/m³. El villano clásico: producir agua cuesta, por eso "más inyección" no siempre gana. |
| `water_injection_cost` | Coste por m³ de **agua inyectada**. | 2 USD/m³. TERCER término (novedad T-R1): bombear agua ya cuesta aunque nunca produzcas de vuelta ese volumen — cierra la puerta a "la inyección gratis" que sesgaba al mock viejo (y al prior CRM, que premia soporte). |
| `annual_discount` | 8% anual → `disc = 1.08^(1/12)` mensual. | `discountRate: 0.08` de EGG_TEST. |

**Fórmula del NPV que estas claves definen** (idéntica en mock y en la ruta
de lectura de BIOPSO, que usa `NPVHandler.fieldNPVcum[-1]`):

```
NPV = Σ_t [ (OPR_t·45 − WPR_t·6 − WIR_t·2)·30 ] · (1.08^(1/12))^(−t)
```

con OPR/WPR/WIR en m³/d y los volúmenes mensuales = tasa×30 días.

---

## `budget` — cuánto puede gastar cada brazo

```yaml
budget:                      # por brazo y por seed
  metaheuristics: {pop: 15, iters: 6}                # 15*6 = 90 sims = coleta S1
  bo: {batch: 5, iters: 16, pool: 4000, n_init: 10}  # 10 + 16*5 = 90 sims
n_seeds: 20                  # protocolo real; la demo toy lo sobreescribe
```

| Clave | Qué es y su contabilidad |
|---|---|
| `metaheuristics.pop/iters` | 15×6 = **90 sims**, idéntico a los configs `egg_{mfo,gwo,pso,foa}_protocol.yaml` de la coleta S1 (allí: mfoMoths/gwoWolves/psoParticles/foaFlies=15, *Iterations=6). Los baselines del artículo vienen de BIOPSO, así que el presupuesto local solo define las réplicas mock. |
| `bo.batch` | `q` = sims en paralelo por iteración. 5 ≤ regla MC2 de 7 procesos. |
| `bo.iters` + `bo.n_init` | `n_init(10) + batch(5)·iters(16) = 90` sims propias del BO — **igualado a 90 a propósito**: el fairness bo vs bo_prior (y frente a baselines reales) se juega con el mismo número de simulaciones físicas. `n_init` se lee del YAML (antes era un literal del loop). |
| `bo.pool` | candidatos LHS puntuados por iter (no cuesta sims, solo CPU del surrogate). |
| `n_seeds` | Repeticiones por brazo: **20 = protocolo del artículo** (potencia del Mann-Whitney); `toy_experiment.py --quick` usa 2. La contingencia del cronograma: bajar `iters` antes que `n_seeds`. |

Regla de MC2 que no está en el YAML pero gobierna todo: **≤7 sims simultáneas**
y ~7 min/sim → los presupuestos de arriba son también un plan de tiempo.

---

## `crm` y `prior` — el conocimiento físico

```yaml
crm:
  min_excitation: 50.0       # umbral de tasa para considerar excitada una arista
prior:
  w0: 0.5                    # peso inicial del prior en la adquisicao
  decay_obs: 30.0            # w = w0 * exp(-n_obs/decay_obs)
  use_full_crm: true         # false => voidage escalar (caso degenerado)
```

| Clave | Qué es |
|---|---|
| `crm.min_excitation` | 50 sm3/d: por debajo, una tasa "no está inyectando de verdad" y su contribución a la identificación de τ no debería contarse. Hoy declarado sin uso (deuda anotada en prior_crm.md). |
| `prior.w0` | Peso inicial del prior en la mezcla por rangos (ver bo_acquisition.md). 0.5 = "el CRM vale la mitad de una señal EI". Es HIPERPARÁMETRO A CALIBRAR (el experimento de mock con lag heterogéneo del gate S2 dirá). |
| `prior.decay_obs` | Constante de decaimiento en nº de sims PROPIAS del BO (¡no pool!). Tras 30 sims el prior pesa ~0.18. |
| `prior.use_full_crm` | `true` = CRM completo (τ̂); `false` = `VoidagePrior` (degenerado, score≡0). Es la palanca de la ablación "prior vacío" del protocolo. |

---

## `paths`

```yaml
paths:
  experiments: experiments
```
Carpeta raíz de resultados. Reservado: el script de MC2 leerá esta clave para
no hardcodear rutas; hoy el toy construye `experiments/toy` por su cuenta
(deuda cosmética menor).

---

## Regla de oro

Si un número de un módulo deja de leerse del protocolo (aparece un literal),
es un bug de arquitectura salvo que sea un default de FIDELIDAD al repo
(PSO/FOA) o una constante matemática. La revisión es: `grep` de literales en
`src/` antes de cada release del artículo.
