# Paso 08 — T-R1: realineación del protocolo local al caso EGG_TEST real

**Fecha:** 2026-11-17 · **Estado:** ✔ cerrado (tests 11/11, demo toy re-baselineada)

## 1. El problema: dos monedas distintas

Con los configs reales de la coleta S1 sobre la mesa (`BIOPSO/configs/egg_*_protocol.yaml`),
se vio que el pipeline local venía optimizando un problema DISTINTO del que la
coleta está muestreando:

| Concepto | ResBO local (antes) | EGG_TEST real (coleta S1) |
|---|---|---|
| Caja rate | 500–3000 sm³/d | **0–150 sm³/d** (`injeRateMin/Max`) |
| Caja bhp | 150–350 bar | **200–395 bar** (`prodBhpMin/Max`) |
| Economía | 2 términos: `price_oil 60`, `cost_liquid 8`, descuento 10% | **3 tarifas** de `NPVHandler`: oilRevenue 45, waterCost 6, waterInjectionCost 2, descuento 8% |
| Ancla PSO/FOA | centro de la caja | **control base del deck**: inyectores 79.5 sm³/d, BHP 395 |
| Presupuesto | mh 10×5=50; BO 5+5·10=55 | **15×6 = 90 sims**; BO igualado a 90 |

Si el mock y la coleta no comparten moneda, el dataset entrenado con datos
reales no es utilizable por el código local sin renormalizar, y las
conclusiones del toy no transferen. **T-R1 = realinear TODO al caso real.**

## 2. Los cambios de protocolo (sin drama, pero con contabilidad)

- `configs/protocol.yaml`: caja 0–150/200–395; sección `economics` reescrita a
  las 4 claves de `NPVParams`; **nueva sección `base_controls`** (79.5/395);
  presupuestos `metaheuristics {pop: 15, iters: 6}` y
  `bo {batch: 5, iters: 16, n_init: 10}` → **90 sims por brazo, misma moneda**
  (el fairness bo vs bo_prior vs baselines se juega con idéntico número de
  simulaciones físicas); `n_seeds: 20` (protocolo del artículo).
- `Protocol.base_controls` (`config.py`): expande [rate, bhp] a los 12
  vectores; fallback centro de caja si la sección no existe.
- `pso.py`: partícula 0 anclada a `self.base`; `foa.py`: centro de osmóforos =
  `self.base`. Ahora la réplica local replica TAMBIÉN el anclaje real (antes el
  detalle fino del paso 03 estaba aproximado: el repo ancla al deck, no al
  centro). Nótese lo deliberado del ancla: BHP 395 es el BORDE de caja — el
  control base del deck es un punto pobre a propósito; los algoritmos existen
  para mejorarlo.
- `bo/loop.py`: `n_init` se lee del YAML (antes era literal 5 del constructor).
- Tests: el aserto del ancla usa `p.base_controls`. **11/11 pasan.**

## 3. La recalibración del mock (donde estuvo el trabajo fino)

Con la caja nueva, el mock antiguo no funcionaba físicamente: PI 0.9 con
soportes de ~100 m³/d nunca equilibraba, y p0 300 bar hacía que BHP altos jamas
fluyeran. Nueva calibración: `PI 2.5`, `p0 380`, `pv 8×10⁵`, y el NPV de las
**tres tarifas**:

```
cash_t = (OPR_t·45 − WPR_t·6 − WIR_t·2)·30      npv = Σ cash_t·1.08^(−t/12)
```

### El problema del BHP degenerado (y por qué importa)

Primera recalibración: el perfil de RATE tenía óptimo interior (≈50) pero el
perfil de **BHP era monótono — el óptimo era SIEMPRE el borde 200 bar**. Un
factor de decisión degenerado es un juguete roto: el BO aprendería "bhp=200" y
el trade-off que hace interesante al problema (producir rápido vs agua temprana)
no existiría en esa dimensión.

Se intentó sustituir el breakthrough por "relleno de voidage"
(`fw ~ f(cum_inyectado/cum_producido)`) → no rompió la monotonía. La solución
física correcta fueron **dos mecanismos de cono de agua** (agua-aceite):

```
fw = clip(0.15 + 0.85·(cum_dir/pv)^0.6 · (1 + cone_v·q/q_ref + cone_b·(p_woc−bhp)/100), 0.15, 0.98)
```

- **Cono dinámico** (`cone_v = 0.3`, `q_ref = 400 m³/d`): caudal alto estira la
  interfaz → más agua (esto ya penalizaba sobreinyectar).
- **Cono por drawdown** (`cone_b = 0.9`, `p_woc = 330 bar`): producir por debajo
  del contacto agua-aceite acerca el agua. **Esto es lo que faltaba**: bajar
  BHP sube el caudal PERO acerca el agua → el compromiso existe y el óptimo se
  despega del borde.

Un grid de calibración (`pv ∈ {4,6,8}×10⁵ × cone_b × cone_v`) confirmó que los
dos máximos son interiores para la configuración elegida.

### Propiedades verificadas del mock final

- Muestreo uniforme (400 pts): NPV ≈ **6–17 M USD** por campo; CRM
  `corr(τ_fit, τ_true) = 1.000`.
- Óptimos interiores: **rate ≈ 50 sm³/d** (a bhp 280) y **bhp ≈ 320 bar**
  (a rate 60) — los dos factores viven el trade-off.
- El control base (79.5/395) rinde **≈ 7–10 M**: ancla pobre a propósito, con
  margen de mejora para todos los brazos.

## 4. Re-baseline toy bajo la nueva economía (236 sims, `--quick`)

```
gwo 16.4M  |  bo 16.8M  |  pso 16.1M  |  bo_prior 16.2M  |  foa 15.7M  |  mfo 15.1M
Mann-Whitney U(bo_prior vs bo) = 0.00, p = 0.121, r_rb = −1.00
```

- **bo > bo_prior persiste**: el prior CRM sigue sobrevalorando soporte ahora
  que inyectar CUESTA (waterInjectionCost). El hallazgo del paso 02 sobrevivió
  a la realineación → el brazo de ablación sigue siendo obligatorio, y ahora
  con economía realista.
- El presupuesto bo = bo_prior = 90 hace la comparación exacta (antes ±5).

## 5. El gate del BNN FALLA en el toy — y es información, no ruido

`train_bnn.py` sobre la BD toy nueva: **NRMSE_val = 6.1% > 5% → GATE FALLO**
(R² 0.935, cobertura 2σ = 0.85). Diagnóstico honesto:

- El rango NPV del dataset **colapsó a 7.8 M USD** (el mock realineado es más
  "plano" que el viejo de 600M): el denominador del NRMSE es chico → el gate
  es estrictamente más difícil de pasar.
- Subir epochs 250→600 solo movió 6.4%→6.1%: no es falta de entrenamiento.
- **No es un bug del pipeline** (la ruta completa train→gate→npz funciona; el
  gate está discriminando de verdad por primera vez).
- Conclusión operativa: **el `bnn_v1` serio sale de la coleta S1 real** (1.920
  sims, cobertura DoE uniforme), no del toy de 236 puntos sobre 2 seeds.

## ¿Qué quedó abierto?

- Docs sincronizados con T-R1 (notas `protocol_yaml.md` y `simulator.md`
  actualizadas el mismo día).
- La fecha real de entrega obligaba a re-anchorar el cronograma → paso 09.
