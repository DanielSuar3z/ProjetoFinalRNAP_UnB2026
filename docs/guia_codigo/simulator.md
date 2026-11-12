# Guía: `src/resbo/simulator.py` — el contrato con el "mundo físico"

**Rol de una línea:** define QUÉ es una simulación para todo el proyecto
(la interfaz `BaseSimulator`), da una versión de juguete que corre en
milisegundos (`MockSimulator`) y documenta cómo se implementará la real
(`OpmSimulator`, stub).

**La idea arquitectónica:** MFO/GWO/PSO/FOA/BO no saben nada de OPM Flow. Solo
conocen el contrato:

```python
out = sim.evaluate(controls)     # controls: (12,) [rates×8, bhp×4]
out["npv"]        -> float       # fitness
out["prod_curve"] -> (n_prod, horizon)   # producción por pozo y mes
```

Si el pipeline funciona contra el mock, el único archivo que cambia para MC2
es este: se reemplaza `MockSimulator` por `OpmSimulator` y nada más se toca.

---

## `class BaseSimulator`

Clase base ("interfaz") que documenta el contrato. No tiene estado ni lógica:
sus métodos hacen `raise NotImplementedError` para que usarla directamente
falle pronto y en voz alta.

| Método | Qué es |
|---|---|
| `evaluate(controls) -> dict` | **La unidad atómica del proyecto**: una simulación completa del campo entero con controles constantes. 1 agente = 1 evaluate (patrón BIOPSO). |
| `evaluate_many(controls) -> list[dict]` | Bucle por comodín sobre `evaluate`. En MC2 se re-escribirá con cola de procesos paralelos (máx 7), por eso es un método y no un bucle suelto. |

---

## `class MockSimulator(BaseSimulator)`

Un yacimiento de bolsillo con la **estructura** del Egg (8 inyectores, 4
productores, 130 meses, conectividades ocultas) pero dinámica cerrada en numpy.
Su trabajo es ser lo suficientemente "física" para que:
1. el CRM pueda recuperar τ (identificabilidad), y
2. el NPV premie/castigue las mismas decisiones que el Egg real
   (soporte de presión ↑ = bueno; sobre-inyectar = agua cara).

### `__init__(self, protocol, seed=42)` — los parámetros físicos

| Atributo | Valor | Qué es | Por qué ese nombre |
|---|---|---|---|
| `n_inj`, `n_prod`, `T` | 8, 4, 130 | Vienen del `Protocol`. `T` = nº de meses del horizonte (convención: `T` para el nº de pasos de tiempo). |
| `price` | 60 USD/sm3 | Precio del petróleo. |
| `cost` | 8 USD/sm3 | Coste por m³ de **líquido** producido (agua+aceite): manejar agua cuesta. |
| `disc` | (1+0.10)^(1/12) | Factor de descuento **mensual**. Se guarda la base (1.1)^(1/12) para aplicar `disc**(-t)`. `disc` = discount. |
| `tau_true` | `(8,4)`, oculta | **Las conectividades verdaderas.** `rng.uniform(0.3,1.0)` y normalizada **por fila** (`tau/tau.sum(axis=1)`): cada inyector reparte un total de 1.0 entre sus 4 productores. Solo el mock la conoce; el CRM intenta adivinarla. `axis=1` = sumar por filas (por inyector). |
| `PI` | 0.9 m³/d/bar | Productivity Index: `q = PI·(p − bhp)`. Es la "ley de Darcy" del pozo. Se llama PI por la convención petrolera (IP/PI). |
| `c_store` | 12 bar·d/m³ | Capacidad del reservorio por productor: cuánto sube la presión por m³ acumulado (`dp = (soporte − q)/c_store`). `c` = capacitance (¡de ahí que el CRM sea un modelo de capacitancia-resistencia!). |
| `p0` | 300 bar | Presión inicial. `p0` = presión en t=0. |
| `lag` | 24 meses | Retardo con que el soporte del inyector "llega" al productor: `1−exp(−t/lag)`. Un solo valor para todas las aristas → **esta es la simplificación que la variante GCPRM relajaría** (decisión S2 pendiente). |
| `pv` | 2.6×10⁷ m³ | Volumen poroso por productor: escala contra la que se mide el "agua ya dirigida acumulada" para el breakthrough. `pv` = pore volume. |
| `days` | 30 | Días por mes (convierte tasas diarias en volúmenes mensuales). |

**Nota sobre `seed=42`:** semilla del mock ≠ semilla del algoritmo. Cambiar la
seed del mock cambia el "yacimiento" (otra τ_true); todos los brazos del toy usan
`seed=42` para competir sobre **el mismo campo**.

### `evaluate(self, controls) -> dict` — paso a paso

Recoge los candidatos y simula el campo entero (vectorizado donde puede, bucle
donde hay recurrencia temporal):

1. **Cortar el vector de control**
   - `rate` — `(8,)` tasas de inyección. `bhp` — `(4,)` presiones de fondo.
2. **Soporte de presión con retardo**
   - `t = np.arange(1, T+1)` — los meses 1..130 (nunca mes 0: no hay producción "en t=0").
   - `support = rate @ tau_true` — `(4,)`. Multiplicación matricial: `support_j = Σ_i τ_ij·rate_i`, el soporte **en régimen permanente** que cada inyector brinda al productor j. `@` es el operador de producto de matrices de Python.
   - `support_t = support[None,:]·(1−exp(−t[:,None]/lag))` — `(130,4)`: el mismo soporte encendido gradualmente. `[None,:]` añade el eje de filas para el broadcasting (mes × pozo).
3. **Agua dirigida acumulada (motor del breakthrough)**
   - `inj_vol_month = rate·days` — m³ inyectados por inyector en un mes.
   - `directed = inj_vol_month @ tau_true` — `(4,)` m³/mes que llegan a cada productor.
   - `cum_dir = np.cumsum(directo·curva_de_lag, axis=0)` — `(130,4)`: acumulado **en el tiempo** (`cumsum` = cumulative sum; `axis=0` acumula bajando por las filas/meses). Este `axis=0` fue un bug cazado: sin él el cumsum aplanaba todo y reventaba el broadcasting.
4. **Bucle de presión/producción (mes a mes)** — no se puede vectorizar porque `p` de hoy depende de `q` de ayer:
   - `q[k] = np.maximum(PI·(p_cur − bhp), 0)` — ley del pozo; el `maximum` impone `q ≥ 0` (un pozo no "succiona" al revés si p<bhp).
   - `p_cur += (support_t[k] − q[k])/c_store` — balance de masas: lo que entra menos lo que sale, acumulado como presión.
   - `p`/`q` — historiales `(130,4)` (se guardan para diagnóstico; el NPV no usa `p`).
5. **Fracción de agua y split aceite/agua**
   - `fw = clip(0.15 + 0.85·(cum_dir/pv)^0.6, 0, 0.98)` — **fracción de agua** (water cut). Arranca en 15%, sube con el volumen de agua dirigido acumulado (potencia 0.6 = curva suave, no física exacta), techo 98%. `fw` = fraction of water (equivalente a WWCT en los summaries de OPM).
   - `q_oil = q·(1−fw)` / `q_liq = q` — lo que vende vs lo que paga coste.
6. **NPV**
   - `disc_t = disc**(-t)` — factor de descuento (un USD del mes 130 vale menos).
   - `cash = (Σ_j q_oil·price − Σ_j q_liq·cost)·days` — flujo mensual del campo (sumando pozos con `axis=1`).
   - `npv = (cash·disc_t).sum()` — el fitness.
7. **Retorno**: `{"npv", "prod_curve": q_liq.T, "fw_curve": fw.T}` —
   el `.T` transpone de `(130,4)` (orden del cálculo) a `(4,130)`: **la
   convención del proyecto** (pozos en filas). `fw_curve` va "de propina" para
   análisis, nadie la consume aún.

---

## `class OpmSimulator(BaseSimulator)` — STUB

`__init__` lanza `NotImplementedError` **a propósito**: si alguien lo
instancia en local, quiere que explote, no que simule algo falso
silenciosamente. El docstring es el plan de la tarea 1.3 del cronograma:
1. re-usar el motor de spawn de BIOPSO (`mpirun -n 1 flow --output-dir ...`),
2. parsear summaries WOPR/WWCT/WBHP por pozo (130 meses) → `(n_prod, horizon)`,
3. mismo NPV que el mock (mismos price/cost/descuento del protocolo),
4. `evaluate_many` con cola de ≤7 procesos + reintentos + `db.add` incremental.

---

## Cómo se conecta con el resto

Recibe llamadas de los cinco algoritmos y de `BayesOptimizer`; entrega NPV y
curvas que acaban en `SimDatabase`. El mock es también la "verdad" del test de
identificabilidad del CRM (`tau_true` vs `crm.tau`).
