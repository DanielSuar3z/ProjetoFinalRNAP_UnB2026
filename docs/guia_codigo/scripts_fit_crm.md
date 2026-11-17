# Guía: `scripts/fit_crm.py` + contracto `crm_v1.npz` — el prior con gate (T-R4)

**Rol de una línea:** convierte los históricos de producción de las corridas
(coleta S1 o cualquier BD con curvas) en el **prior físico persistido** que el
brazo BO consumirá: la matriz τ (8×4) con **certificado de calidad por pozo**.

Hermano de `train_bnn.py`: el BNN es el surrogate, el CRM es el prior; ambos
se ganaron el derecho a entrar en el lazo mediante un gate numérico.

---

## Por qué un script aparte y no "CRM dentro del loop BO"

El CRM de la coleta se ajusta **una vez, con los ~1.920 históricos de la ruta
A**, antes de lanzar cualquier brazo. Meter el re-ajuste dentro del lazo
acoplaría el prior a los puntos que el propio BO visita (self-confirming) y
haría irrepetible la comparación entre brazos. Prior congelado = ablación
honesta.

## Flujo del script

```
--harvest experiments/coleta_s1        (o --db experiments/toy, o --self-check)
        │  loader_biopso.load_harvest: acepta SOLO ruta A (simlog+npz)
        ▼
  emparejar (controles ↔ curva) desde los records   ← mismo filtro para ambos
        ▼                                            (filtrar por separado desalinearía X y curves)
  diagnosticar excitación por inyector: ¿que frac de sims tiene rate_i >= 50?
        ▼                                            (< 25% → aviso: columna τ poco identificable)
  CRMModel.fit(late_frac=0.4): q_mean del ultimo 40% del historial ~ rates·τ + q0
        ▼                                            lstsq + clip por productor → R² por pozo
  gate: well_mask = R² >= 0.7. Pozo fallido → columna de τ EN CERO (el score
        ignora ese pozo). < 2 pozos OK → exit 2: no se lanza BO+prior.
        ▼
  experiments/models/<tag>.npz (crm_v1) + <tag>_metrics.json
```

## El contracto `crm_v1.npz` (`prior/crm.py`, funciones `save_crm_npz`/`load_crm_npz`)

Misma filosofía que `bnn_v1.npz`: npz comprimido, **sin pickle**, schema
versionado, legible desde BIOPSO (T-R5) sin importar ResBO.

| Clave | Shape | Qué es |
|---|---|---|
| `schema` | str | `"crm_v1"`; `load_crm_npz` rechaza cualquier otro valor (falla pronto, no lea basura). |
| `tau` | (8,4) | τ **final con máscara aplicada** (columnas de pozos fallidos en cero). Lo que usa `score()`. |
| `tau_full` | (8,4) | τ cruda del ajuste, antes de purgar — auditoría. |
| `q0` | (4,) | Producción "natural" del productor sin soporte de inyector. |
| `fit_r2` | (4,) | R² late-time por pozo (la métrica del gate). |
| `well_mask` | (4,) int8 | 1 = pozo confiable. Vive aquí, no solo en el JSON, porque el lado BIOPSO decide sin leer archivos sueltos. |
| `n_runs_used` | int | Nº de sims del ajuste. |
| `meta_json` | str | Proveniencia (`fuente`), `curva_source`, hyperparams, estadísticas de excitación. |

## Decisiones de diseño (las que hay que poder defender en la oral)

1. **¿Por qué τ en cero y no excluir el pozo del score?** El score suma soporte
   sobre pozos; una columna nula hace que ese pozo no contribuya SIN tocar
   `score()` — la función no necesita conocer el mask (que igualmente viaja en
   el npz para auditoría y para el artigo).
2. **¿Por qué late-time (último 40%)?** La ecuación CRM es steady-state; los
   primeros meses del mock/Flow están dominados por la transitoria de presión.
   El promedio del tramo final es el observable que la linealidad `q ≈ τ·I + q0`
   puede representar.
3. **¿Por qué solo ruta A?** La ruta B (rescate) aporta 1 fila/corrida y con
   longitud de curva ajena al horizonte: no identifica 32 parámetros de τ.
4. **`curva_source`**: en el mock la curva es tasa de LÍQUIDO; en el npz real,
   `wpr` (agua). Se registra en meta para que el análisis con datos reales
   decida (¿usar wpr o líquida?) con evidencia, no de memoria.

## Autovalición sin datos reales (estado actual)

- `--self-check`: fixture de 5 runs × 40 sims DoE uniformes sobre UN mock
  (una sola tau, como la coleta real) → R² por pozo = 1.0,
  `corr(τ_fit, τ_true) = 1.000`, roundtrip del npz. Con datos de Flow los R²
  bajarán: para eso está el umbral 0.7 y el aviso de excitación.
- `--db experiments/toy`: 236 sims de la demo → gate PASS (mismo comentario).
- Test `test_crm_npz_roundtrip_and_mask`: roundtrip, mask, pozo cero≡score sin
  pozo, rechazo de schema ajeno.

Vea también: [prior_crm.md](prior_crm.md) (la matemática del CRM y del score),
[loader_biopso.md](loader_biopso.md) (de dónde salen las curvas).
