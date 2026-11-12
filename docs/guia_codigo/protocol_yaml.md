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
bounds:
  rate:  [500.0, 3000.0]     # sm3/d por inyector
  bhp:   [150.0, 350.0]      # bar por productor
```

- Escalares POR TIPO de pozo: `Protocol.bounds` los expande a los vectores
  `lb/ub` de 12. No se usa una lista de 12 pares porque en el Egg todos los
  inyectores comparten límites.
- `rate` — tasa de inyección. Rango elegido tipo Egg histórico (los defaults
  del repo BIOPSO son 0–150 en sus unidades de campo; el protocolo local usa
  sm³/d — al migrar a MC2 hay que alinear UNO de los dos: tarea 1.3).
- `bhp` — presión de fondo del productor. Menor BHP ⇒ mayor caudal, pero el
  agua del vecino entra más fuerte: el trade-off del problema vive aquí.
- Física del rango: muy bajo rate → el reservorio se depleta y la presión (y
  el NPV) se hunde; muy alto → breakthrough temprano y coste de manejar agua.
  El óptimo está en el medio, con compensaciones por pozo.

---

## `economics` — cómo se convierte producción en dinero

```yaml
economics:
  price_oil: 60.0            # USD/sm3
  cost_liquid: 8.0           # USD/sm3 (manejo de líquido)
  annual_discount: 0.10
```

| Clave | Qué es | Por qué ese valor |
|---|---|---|
| `price_oil` | Ingreso por m³ de aceite producido. | Precio tipo de estudio (los 60 USD/sm³ del informe ANP del equipo — mantenerlo idéntico a BIOPSO para que los NPV sean comparables con las corridas del repo). |
| `cost_liquid` | Coste por m³ de **líquido total** (agua+aceite). | Es el villano del problema: producir agua cuesta y por eso "meter más inyección" NO siempre gana (el mock la usa como castigo; es la fuente del riesgo "prior engañoso"). |
| `annual_discount` | 10% anual. El dinero futuro vale menos: `disc = 1.1^(1/12)` mensual. | Tasa estándar en evaluación de proyectos petroleros (y default del framework BIOPSO/ChronosUNB). |

**Fórmula del NPV que estas claves definen** (implementada idéntica en mock y
futuro OpmSimulator):

```
NPV = Σ_t [ (Σ_j q_oil,j·price − Σ_j q_liq,j·cost)·30 ] · (1.1^(1/12))^(−t)
```

---

## `budget` — cuánto puede gastar cada brazo

```yaml
budget:
  metaheuristics: {pop: 10, iters: 5}        # 10 + 10*5 = 60 sims
  bo:               {batch: 5, iters: 10, pool: 4000}   # 50 sims (incluye inicial DoE)
n_seeds: 4                   # demo toy; protocolo real: 20
```

| Clave | Qué es y su contabilidad |
|---|---|
| `metaheuristics.pop/iters` | Todos los brazos baselines comparten presupuesto (fairness): MFO/GWO gastan `pop·(iters+1)=60` (evalúan la inicial), PSO/FOA `pop·iters=50` (evalúan dentro del lazo). Esta asimetría de ±10 sims está documentada en tests y notas. |
| `bo.batch` | `q` = sims en paralelo por iteración. 5 ≤ regla MC2 de 7 procesos. |
| `bo.iters` | Ciclos del laço → `n_init(5) + batch(5)·iters(10) = 55` sims propias del BO. |
| `bo.pool` | candidatos LHS puntuados por iter (no cuesta sims, solo CPU del surrogate). |
| `n_seeds` | Repeticiones por brazo. 4 = demo; **20 = protocolo del artículo** (potencia del Mann-Whitney). La contingencia del cronograma: bajar `iters` antes que `n_seeds` (la estadística manda). |

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
