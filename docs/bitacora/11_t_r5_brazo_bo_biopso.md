# Paso 11 — T-R5: el brazo BO cruza la frontera (BIOPSO corre la Optimización Bayesiana)

**Fecha:** 2026-11-17 · **Estado:** ✔ cerrado en código (18+10 tests verdes, rama
`feature/bayes-crm-ingesta` pusheada); el smoke real sobre OPM queda pendiente
del acceso a MC2 y del deploy con datos de la coleta

## 1. La decisión que gobierna el paso

Todas las tareas anteriores vivieron DEL lado ResBO (mock, numpy, tests).
T-R5 es el cruce: el lazo BO — la contribución central del proyecto — tiene
que correr DENTRO del motor BIOPSO como un `Algorithm` más, evaluando cada
candidato con una simulación real de OPM Flow, con el mismo presupuesto y la
misma caja que los cuatro baselines de la coleta S1. Y sin poder aún tocar
los datos reales: todo se construye contra contractos y se valida con el
lector/escritor cruzados de los dos repos.

## 2. La frontera: dos repos que solo se hablan por npz

BIOPSO NO importa ResBO (es privado, y la ciencia debe ser reproducible sin
él). El puente son tres artefactos sin pickle que el ResBO público produce y
el brazo privado consume:

```
resbo (publico)                              BIOPSO (privado, storms_unb/resbo/)
  scripts/export_deploy.py  ──rsync──▶  deploy/resbo_v1/
    data_v1.npz  (experiencia pooled: X, y de la coleta)   → entrena el ensemble
    crm_v1.npz   (tau enmascarada + well_mask)             → prior en la adquisicion
    bnn_v1.npz   (arquitectura + atestacion del gate)      → hiperparams, NO surrogate congelado
```

El round-trip se probó de verdad: el smoke export del toy (236 sims) fue
leído por los CARGADORES del lado BIOPSO (`bo_core.load_experience`,
`load_crm_npz`, `BnnEnsemble.load_npz`) — las dos mitades del contracto
verificadas juntas, sin esperar a MC2.

## 3. Las tres decisiones de diseño del brazo (y su defensa)

**3.1 Reentrenar cada iteración, no congelar el `bnn_v1`.** El npz entrenado
en la coleta es el estado del arte ANTES de la corrida; usarlo congelado
significaría que el surrogate nunca aprende de las 90 sims que el propio
genera — contradiendo el espíritu de la BO secuencial. Cada iteración:
ensemble = experiencia pooled (los datos de los baselines, la premisa del
proyecto) + datos propios. El `bnn_v1.npz` entra solo por dos cosas: fija la
arquitectura (mismos `hidden`/`p_drop` que validó el gate) y su metrics
atestigua que el pipeline pasó el NRMSE < 5% — el runtime de BIOPSO jamás
usa sus pesos para predecir.

**3.2 El decaimiento cuenta SOLO las sims propias.** `w(n) = w0·e^(−n_sims/τ)`
con `n_sims` = evaluaciones del propio lazo (10, 15, 20…), no las ~1.920
observaciones pooled. Argumento (lectura de DynMeanBO): el prior existe para
compensar que ESTE lazo está data-scarce; el conocimiento de los baselines no
enciende al prior — si contáramos el pool, `w` nacería muerto y el brazo con
prior sería idéntico a la ablación desde la iteración 1. Es también la
versión conservadora: si el prior ayuda, no podrá ser por un artefacto del
esquema de conteo.

**3.3 Contractos fail-fast, logging blindado.** Simétrico al dolor real: una
corrida BO mal configurada (sin experiencia, prior de otro caso, <2 pozos
confiables) gastaría ~10 h de pared para nada → muere en el segundo cero con
`ValueError` explícito. En cambio un fallo de disco escribiendo `simlog.csv`
NUNCA tumba una corrida que ya está a la mitad (mismo principio del parche
T-B de la coleta). El gate S2 queda reforzado en runtime: `<2` pozos
confiables en el `crm_v1.npz` → el brazo con prior está vetado aunque el YAML
insista.

## 4. Geometría del simlog: la ronda 0 y `--init-agents`

El brazo no tiene población fija: DoE LHS propio (`resboNInit=10`) + 16
rondas de `resboBatch=5`. El contrato `simlog_spec_v1` NO se rompió — se
documentó como **Anexo A**: ronda `iteration=0` con `n_init` filas, rondas
`1..16` con 5, total 90 (= 15×6 de la coleta, la misma moneda). El verificador
acepta la geometría con `--init-agents` y repliega la ronda 0 al máximo
corriente de `convergence.csv` (el gbest de la ronda 0 ya está plegado en la
primera fila, como arranca el lazo). El contrato clásico sin la opción
verifica IDÉNTICO que antes — hay un test de regresión que lo garantiza.

## 5. Ablación limpio = configs que se prueban entre sí

`egg_resbo_prior.yaml` y `egg_resbo_noprior.yaml` difieren SOLO en
`resboUsePrior`/`resboPriorFile`: misma experiencia, misma arquitectura (el
ablado también carga `resboModelFile`), mismos hiper, misma caja, misma
economía, mismo `predictionWindow`. `resboW0`/`resboDecay` solo existen en el
config con prior porque con `resboUsePrior=0` el lazo fuerza `w=0` —
escribirlos en la ablación sería mentir en el YAML. Un test lee los dos YAML y
verifica que el conjunto de diferencias sea exactamente ese: si alguien toca
un hyper de un brazo y no del otro, el test rompe antes de gastar sims.

## 6. `scripts/export_deploy.py` — el portero de los gates

Nuevo en el repo público: empaqueta `deploy/resbo_v1/{data,crm,bnn} +
deploy_manifest.json` (sha256 para verificar la transferencia a MC2) y
ABORTA si los metrics hermanos no atestiguan `gate_pass: true` o si el CRM
tiene <2 pozos confiables. `--smoke` relaja solo el gate (cableado con
artefactos del toy, que reprobó el NRMSE con la economía real — paso 09) y
deja `smoke: true` grabado en el manifest y en el `meta_json` de la
experiencia: un deploy de juguete no puede disfrazarse de uno de medida.
El reject está probado de verdad: un intento de export real (sin `--smoke`)
con el BNN reprobado del toy ABORTÓ con el mensaje del gate — el portero
funcionó en una invocación real, no solo en un test.

## 7. Lo que corrió hoy (resultados de la sesión)

- `test_resbo_bo.py` (nuevo, BIOPSO): **18/18** — BNN fit/save/load, loaders
  que rechazan schemas ajenos y experiencias cortas, mezcla por rangos con
  EI de picos brutales acotada, decaimiento estricto, score CRM ignora pozo
  enmascarado (al valor exacto) y prefiere el inyector conectado, LHS
  estratificado 1-punto-por-franja, `batch_select` diverso con relleno sin
  cuelgue, mini-lazo BO end-to-end (10 sims, `w` decreciente, gbest ≥ init),
  geometría del checker (acepta resbo, rechaza resbo sin la opción, clásica
  intacta), equidad de presupuesto 10+5×16=90=15×6 y de economía contra el
  `egg_foa_protocol.yaml`.
- `test_simlog.py`: **10/10** sin regresión (el checker modificado no tocó el
  contrato de la coleta).
- Export smoke → lectura con los cargadores de BIOPSO: experiencia (236,12),
  τ (8,4) mask [1,1,1,1], ensemble (8 modelos, 32×32) reconstruido.
- Rama BIOPSO `feature/bayes-crm-ingesta` pusheada a origin (nueva rama
  remota) con commit T-R5; fila en el CHANGELOG de BIOPSO y `docs/COLECTA_S3.md`
  (runbook MC2: rsync+sha256, smokes de cableado de ambos brazos, N=20 con
  contingencia 16→12, verificación run a run con `--init-agents 10`).

## 8. Efecto en el calendario

Con T-R5, la cola crítica queda reducida a lo que solo el acceso puede
desbloquear: `fit_crm --harvest` real → `train_bnn` real (gate NRMSE) →
`export_deploy` (sin smoke) → smokes S3 → las 2×20 corridas de medida. Todo
lo demás del pipeline está escrito, testeado y pusheado en los dos repos.

## ¿Qué quedó abierto?

- Smoke S3 sobre OPM real (necesita MC2 + deploy real; el runbook está listo).
- Decisión pendiente del ejecutor desde hace pasos: confirmar el gate S2 para
  el upgrade GCPRM-lite (Δcorr > 0.1 **y** R² ≥ 0.7) — afecta a S4, no a S3.
- Con datos delante: `curva_source` (líquida del mock vs `wpr` real) decide
  con qué curva se fita el CRM real (anotado en el meta del `crm_v1.npz`).
- Si el checker de un run de medida falla por geometría, revisar que se pasó
  `--init-agents 10` (tabla de troubleshooting en COLECTA_S3 §6).
