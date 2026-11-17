# Paso 04 — Repo público `ProjetoFinalRNAP_UnB2026` y reglas de publicación

**Fecha:** 2026-11-16 · **Estado:** ✔ cerrado (el repo se enriquece en cada paso posterior)

## 1. Qué se creó

Repo **público** en GitHub: `https://github.com/DanielSuar3z/ProjetoFinalRNAP_UnB2026`
(rama `main`), que es el directorio `resbo/` versionado con git. Cumple el
requisito del curso de artigo + código en repositorio público. Commits:

| Commit | Contenido |
|---|---|
| `782c5a2` | ResBO v0.1.0 — pipeline completo sobre el mock (paso 02/03) |
| `15cfdbd` | Contrato `simlog_spec_v1` + loader BIOPSO estricto + persistencia BNN + `train_bnn.py` con gate + verificador; 11/11 tests; guía notas 16–17 (pasos 05–07) |

El CHANGELOG como **tabla de decisiones** (filas: fecha, rutas, responsable,
descripción) es formato heredado de BIOPSO — no "keepachangelog" con secciones,
porque lo que se audita aquí son DECISIONES, no releases.

## 2. Estructura publicada

```
resbo/
├── CHANGELOG.md              # bitácora de decisiones (tabla)
├── configs/protocol.yaml     # el protocolo experimental (única fuente de verdad)
├── src/resbo/                # simulador, BD, 4 baselines, CRM, BNN, BO, análisis
├── scripts/                  # toy_experiment.py, train_bnn.py, validate_loader.py
├── tests/test_core.py        # 11 pruebas (mini-runner, sin pytest)
├── docs/
│   ├── estudo/               # notas de estudio + simlog_spec_v1.md (contrato)
│   ├── guia_codigo/          # 17 notas: una por archivo, explicadas a detalle
│   └── bitacora/             # ESTO que estás leyendo (añadida 2026-11-17)
└── experiments/toy/          # resultados de la demo toy (CSV + figuras) SÍ versionados
```

## 3. Reglas de publicación (lo que NUNCA sube)

1. **`docs/ref_biopso/` está en `.gitignore`.** Contiene las copias de
   `pso_wellcontrol.py` / `foa_wellcontrol.py` del repo PRIVADO
   `petroleoinfralab-ai/BIOPSO`. Son referencia local para las réplicas (paso
   03); el repo público solo puede citar las fórmulas y la fuente, nunca
   redistribuir el código. (Las notas de la guía describen las fórmulas, que es
   citation-safe.)
2. **`experiments/models/` y `experiments/demo_harvest/` gitignored**: artefactos
   grandes/reentrenables (`.npz` de pesos, cosechas sintéticas). Lo que SÍ se
   versiona: `results.csv`/`simulations.csv` y figuras (reproducibilidad sin
   binarios).
3. El motor headless del lab no se reescribe en el repo público ni se copia: la
   integración real vive del lado BIOPSO (pasos 05–06), y el repo público consume
   SUS salidas por un contrato de datos, no por import de código.

## 4. Flujo de push usado (para reproducibilidad del propio proceso)

La máquina no tiene credenciales de GitHub persistentes para este repo; se usa
un credential helper **efímero** que vive solo dentro del comando:

```bash
git -c credential.helper="!f(){ echo username=x-access-token; echo password=$GITHUB_TOKEN; };f" push origin main
```

(login `DanielSuar3z`). El token del lab con acceso al repo privado BIOPSO está
en `~/.git-credentials` para el clone privado; su **rotación sigue pendiente**
(tarea hygiene en el paso 09).

## ¿Qué quedó abierto?

- El pipeline local estaba completo… pero nadie había verificado todavía CÓMO
  salen los datos del BIOPSO real. Esa verificación (paso 05) resultó ser el
  punto de inflexión del proyecto.
