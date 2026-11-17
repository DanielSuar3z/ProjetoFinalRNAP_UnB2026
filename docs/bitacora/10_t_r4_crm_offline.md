# Paso 10 — T-R4 preparado offline: `scripts/fit_crm.py`, contracto `crm_v1.npz` y gate por pozo

**Fecha:** 2026-11-17 · **Estado:** ✔ cerrado (validado contra fixture; falta solo apuntarlo a `coleta_s1`)

## 1. La decisión que gobierna el paso

Preguntado el ejecutor por el origen de los datos de la coleta (MC2 / PC nuevo /
no corrida), la respuesta fue: **"aún no hay acceso — preparar todo offline"**.
Eso define la estrategia del día: construir T-R4 (ajuste del prior CRM) COMPLETO
y auto-validado contra un fixture sintético, de modo que cuando lleguen los
datos el único cambio sea la ruta del argumento `--harvest`. Un script que no se
puede probar sin datos es un script que se depura con datos — y esos datos
costaron 30 horas de pared.

## 2. Qué es T-R4 en una frase

Convertir los ~1.920 históricos de producción de la coleta (los
`sim_curves.npz` de la ruta A del contrato) en el **prior físico persistido**
que el brazo BO consumirá: la matriz τ (8×4) con **certificado de calidad por
pozo**.

## 3. El contracto `crm_v1.npz` (hermano de `bnn_v1.npz`)

Misma filosofía de persistencia que el BNN (paso 07): npz comprimido, **sin
pickle**, schema versionado, legible desde BIOPSO (T-R5) sin importar ResBO.
Claves: `schema`, `tau`, `tau_full`, `q0`, `fit_r2`, `well_mask`,
`n_runs_used`, `meta_json` (detalle en `docs/guia_codigo/scripts_fit_crm.md`).

Tres decisiones empaquetadas ahí:

1. **`tau` es la tau ENMASCARADA** (columnas de pozos fallidos en cero), y
   `tau_full` guarda la cruda para auditoría. Un cero en la columna hace que
   `score()` simplemente no tome en cuenta ese pozo — sin if-statements ni
   conocimiento del mask dentro de la función (verificación testeada:
   pozo-cero ≡ score sin ese pozo, al valor exacto).
2. **`well_mask` viaja en el npz**, no solo en el metrics.json, porque el lado
   BIOPSO tiene que poder decidir sin leer archivos sueltos.
3. **`curva_source` en meta**: en el mock `prod_curve` es tasa de LÍQUIDO; en el
   npz real la curva es `wpr` (agua). El CRM modela producción total, así que
   esta salvedad se REGISTRA para que con datos reales delante se decida con
   evidencia (¿fita con `wpr` o reconstruir líquida?), no de memoria.

`load_crm_npz` rechaza cualquier schema ≠ `crm_v1` (falla pronto, no lee basura).

## 4. `scripts/fit_crm.py` — el flujo y sus guardas

```
--harvest coleta_s1 (solo ruta A)  |  --db local  |  --self-check fixture
        ▼
emparejar (controles ↔ curva) desde los records, UN solo filtro
        ▼
diagnóstico de excitación: ¿qué fracción de sims tiene rate_i ≥ 50 sm³/d?
        ▼                       (< 25% → aviso: columna τ poco identificable)
CRMModel.fit(late_frac=0.4): promedio del último 40% del historial ≈ rates·τ + q0
        ▼
gate por pozo: R² ≥ 0.7 → well_mask; pozo fallido → columna τ en cero
        ▼
< 2 pozos OK → exit 2 ("no lanzar BO+prior")   |   ≥ 2 → crm_v1.npz + metrics.json
```

Detalles con historia:

- **Solo ruta A.** El rescate (ruta B) aporta 1 fila/corrida con longitud de
  curva ajena al horizonte: no identifica los 32 parámetros de τ. Filtrar por
  `iteration >= 0` es decir: la τ sale de poblaciones excitadas, no de ganadores.
- **El bug que se corrigió antes de correr**: la primera versión usaba
  `db.curves()` y luego filtraba las curvas mal formadas *por separado* de X —
  eso desalineaba en silencio controles↔curvas (exactamente el modo de falla
  que el contrato del paso 05 prohíbe). La versión final reconstruye X y curves
  desde la MISMA lista filtrada de records.
- **`late_frac=0.4`**: la ecuación CRM es steady-state; los primeros meses están
  dominados por la transitoria de presión, así que el observable lineal es el
  promedio del tramo final.
- **El gate es por pozo, no global**: un R² global 0.8 puede esconder un pozo
  con R² 0.2 cuyo τ inventado envenena el prior. Mejor un prior que cubre 3
  pozos que uno que miente sobre 4. Y el piso de 2 pozos: con 1 solo pozo
  confiable el "campo de conectividades" es una anécdota.

## 5. Validación sin datos reales (lo que corre hoy)

| Prueba | Resultado |
|---|---|
| `fit_crm.py --self-check` (fixture: 5 runs × 40 sims DoE uniformes sobre UN mock = una sola τ, como S1, generados con `make_synthetic_run` y cargados por el loader REAL) | R² = 1.0 en los 4 pozos, **corr(τ_fit, τ_true) = 1.000**, roundtrip del npz OK, gate PASS |
| `fit_crm.py --db experiments/toy` (BD toy realineada, 236 sims) | gate PASS (4/4 pozos R² = 1.0) |
| `tests/test_core.py` suite completa | **12/12** (nuevo: `test_crm_npz_roundtrip_and_mask` — roundtrip, mask, pozo-cero≡score-sin-pozo, schema ajeno rechazado) |
| Nota de la guía | `docs/guia_codigo/scripts_fit_crm.md` (nota 18) |

**Advertencia honesta, en el espíritu del paso 08:** R² = 1.0 en el fixture es
*esperable* (en el mock el flujo late-time es casi lineal en los rates) y NO
dice nada sobre los datos de Flow, donde la transitoria, la gravidad y el agua
libre van a estresar el umbral 0.7. El fixture prueba el PIPELINE (emparejar →
excitar → ajustar → gatear → persistir → recargar), no la física del Egg. Para
eso existen el umbral, el well_mask y el exit 2.

## 6. Efecto en el calendario

La ventana 20–23/11 del §Calendario al 09/12 (S2) ya no empieza "escribir
fit_crm.py": empieza "apuntar fit_crm.py a coleta_s1 e interpretar R² reales".
El cuello del proyecto sigue siendo UNO SOLO: la transferencia de
`coleta_s1` + `harvest_2026_11` al local.

## ¿Qué quedó abierto?

- **Con datos delante**: `fit_crm.py --harvest` real → interpretar `r2_por_pozo`
  y `excitation_frac` (¿el DoE de 60 sims cubre bien los 8 inyectores?), y
  decidir la curva del fit (`wpr` vs líquida) con el `curva_source` del metrics.
- Confirmación del ejecutor sobre el gate **GCPRM-lite** (upgrade del CRM a
  variantes de tiempo por arista SOLO si Δcorr(score,NPV) > 0.1 **y** R² ≥ 0.7).
- `bnn_v1` real con `train_bnn.py --harvest` (gate NRMSE < 5%) — mismo cuello.
