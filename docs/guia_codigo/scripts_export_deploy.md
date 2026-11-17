# `scripts/export_deploy.py` — empaquetado del deploy del brazo BO para BIOPSO

**Aparece por:** T-R5 (integración BIOPSO). Consumidor: `BIOPSO/deploy/resbo_v1/`
configurado por `configs/egg_resbo_*.yaml` y ejecutado por
`storms_unb/resbo/resbo_wellcontrol.py` (lado privado, no publicado).

## Para qué existe

El brazo BO de BIOPSO NO importa ResBO: habla con él a través de TRES
artefactos binarios (contractos npz sin pickle). Este script es el ÚNICO
productor de esos tres archivos:

| archivo | contenido | productor original |
|---|---|---|
| `data_v1.npz` | schema `resbo_data_v1`: `X` (controles 12D), `y` (NPV), `meta_json` | este script (desde `--harvest` con el loader `simlog_spec_v1`, o `--db`) |
| `crm_v1.npz` | prior τ enmascarado + `well_mask` + `fit_r2` | `scripts/fit_crm.py` (copiado y VERIFICADO acá) |
| `bnn_v1.npz` | pesos del ensemble (contrato `save_npz`) | `scripts/train_bnn.py` (copiado y VERIFICADO acá) |

Más un cuarto: `deploy_manifest.json` con sha256 y bytes de cada archivo —
la transferencia a MC2 se verifica contra él (los binarios viajan por
rsync/scp, NUNCA por git: `deploy/` está en `.gitignore`).

## La atestación (por qué aborta)

Regla S2 del proyecto: **prohibido lanzar el brazo BO con un surrogate o un
prior sin gate**. El script la hace cumplir ANTES de empaquetar:

- `--crm`/`--bnn` exigen el `<tag>_metrics.json` hermano con `gate_pass: true`.
  Ausente o `false` → aborta (sale 1) con el path del metrics en el mensaje.
- `well_mask` con < 2 pozos confiables → aborta (mismo gate que el runtime
  de BIOPSO volverá a aplicar: exportar igual no tiene sentido).
- `X` con NaN/inf o < 30 sims → aborta.
- Dimensiones cruzadas: `tau (8,4)` debe casar con `X` de 12 columnas
  (otro caso = otro deploy; el config de BIOPSO lo ratifica en runtime).

`--smoke` relaja SOLO el gate (para cablear MC2 con los artefactos chicos
del toy, cuyo BNN reprobó el gate NRMSE con la economía real — hallazgo
honesto del paso 09 de la bitácora) y deja `smoke: true` en el manifest y en
el `meta_json` de la experiencia: imposible confundir un deploy de juguete
con uno de medida. Mínimo de sims en smoke: 8.

## El `meta_json` de la experiencia

`{fecha, fuente, n_sims, por_algo, dims_x, smoke, nota}` — el lado BIOPSO lo
imprime al cargar (`resbo: experiencia 1920 sims (fuente: …)`), así cada
corrida queda auto-documentada en su log sin leer archivos sueltos.

## Verificación cruzada (hecha en la sesión T-R5)

El round-trip se validó CONTRA EL LECTOR DE BIOPSO (`bo_core.load_experience` /
`load_crm_npz` / `BnnEnsemble.load_npz`): lo que este script escribe es
exactamente lo que el brazo consume — los dos lados del contracto probados
juntos con el smoke export del toy (236 sims, τ 8×4, ensemble 32×32).

## Uso

```bash
# real (con gates pasados):
python scripts/export_deploy.py --harvest experiments/coleta_s1 \
    --crm experiments/models/crm_v1.npz --bnn experiments/models/bnn_v1.npz
# smoke cableado (NO reportar numeros):
python scripts/export_deploy.py --db experiments/toy \
    --crm experiments/models/crm_toy.npz \
    --bnn experiments/models/bnn_toy_demo.npz --smoke
```
