# Paso 01 — Contexto, ancla bibliográfica y adopción de ResBO

**Fecha:** octubre → 2026-11-06 · **Estado:** ✔ cerrado

## 1. El marco del curso (lo que define el tamaño del problema)

Materia: **Redes Neuronais Aplicadas (RNAP), 2026/2, Universidade de Brasília**,
Prof. Díbio. El "Projeto Individual 2" (PDF `projeto2-rnap-2026_2.pdf`) exige,
INDIVISIBLEMENTE:

1. **Relatório de 2 páginas** (formato libre): qué IA se usó, para qué, y qué
   evidencia hay (nosotros usamos como evidencia los logs de sesión + el
   `CHANGELOG.md`, que es una bitácora de decisiones con fecha y autor).
2. **Artigo de 6–8 páginas en formato IEEE** alojado en un **repo público**
   llamado `ProjetoFinalRNAP_UnB2026` (GitHub pessoal).
3. **Presentación oral: 5 minutos, 4 slides**, en la aula subsequente.
4. **Fecha real de entrega fijada por el ejecutor: 09/12/2026** (re-ancoraje
   del 2026-11-17, ver paso 09).

Consecuencia de planificación: todo el diseño técnico tiene que caber en
~3 semanas de ejecución individual (~15–20 h/semana). Eso es lo que más tarde
recorta el cronograma (3.4 sensibilidad pasa a opcional, etc.).

## 2. Qué es ResBO (la idea en un párrafo de ingeniería)

El problema: **control óptimo de pozos** en el campo sintético **Egg Model**
(8 inyectores + 4 productores, horizonte 130 meses). Cada evaluación de la
función objetivo (NPV del campo) cuesta una simulación física de reservorio
con **OPM Flow** (~7 min en el cluster MC2). Optimizar con metaheurísticas
clásicas (MFO, GWO, PSO, FOA) consume cientos de simulaciones porque cada
individuo = una simulación ciega.

La propuesta: **Bayesian Optimization (BO) guiada por un prior físico**:

- **Surrogate**: una **BNN** (red bayesiana: ensemble de MLPs + MC-dropout en
  numpy) que aprende `controles (12D) → NPV` con incertidumbre, alimentada con
  los datos de corridas previas de las baselines.
- **Prior físico**: un modelo **CRM (Capacitance-Resistance Model)** ajustado
  con los históricos de producción de esas mismas corridas. El CRM entrega la
  matriz de conectividades τᵢⱼ inyector→productor: una estimación BARATA y
  físicamente interpretable de "qué inyector sostiene a qué productor".
- **Mezcla**: la función de adquisición se combina `EI_normalizado + w(n)·prior`,
  con decaimiento estilo **DynMeanBO** (`w = w0·exp(−n_obs/decay)`): al inicio,
  con pocos datos, manda la física; conforme la red ve simulaciones, manda la
  evidencia.
- **Hipótesis**: con el mismo presupuesto de simulaciones, BO+prior ≥ BO solo ≥
  baselines, y la ablación (brazo "BO sin prior") atribuye la ganancia al prior.

## 3. El ancla bibliográfica y por qué "BioBO"

El paper ancla es **BioBO (Li et al., ICLR 2026)**: BO para diseño de
experimentos CRISPR donde el prior viene de *pathway enrichment* (biología).
La innovación del proyecto es la **sustitución del prior biológico por un
prior de ingeniería de reservorio (CRM)** manteniendo la arquitectura
prior-informed — un cambio de dominio inédito, que es exactamente el tipo de
contribución que pide el curso. Se descargaron como contexto en
`papers_ICLR2026/`: BioBO, BOLT, TAMO, GenBO y la tesis de Vinícius Freitas
(ABNT, para el contraste "proxy en el lazo vs prior en la adquisición").

## 4. Adopción del proyecto de Danilo (decisión clave)

El ejecutor heredó/adaptó el proyecto **ResBO** del contexto del lab
(colega Danilo): ya existía la definición (`Definicao_Projeto_RNAP_JaimeDaniel_ResBO.*`,
en 4 formatos sincronizados: `.md` fuente → `.html` → `.pdf`/`.docx` vía
libreoffice; no hay pandoc ni wkhtmltopdf en la máquina), el motor
(**BIOPSO**, repo privado `petroleoinfralab-ai/BIOPSO`, framework
ChronosUNB/STORMS sobre OPM Flow) y el campo de prácticas del grupo (Egg).
Adoptarlo decidió de entrada:

- Los **baselines NO se inventan**: son las implementaciones reales del repo
  BIOPSO — **MFO, GWO, PSO y FOA** (la proposta todavía decía "MFO e GWO"; la
  sincronización a 4 brazos es tarea del paso 09).
- El **NPV no se redefine**: es el de `NPVHandler.fieldNPVcum[-1]` del framework
  (esto estalla en el paso 08, cuando hay que realinear el mock a esa fórmula).
- El **motor de simulación no se reescribe en el repo público**: la regla de
  publicación prohíbe copiar código del lab (paso 04).

## 5. Restricciones del entorno local (definen toda la arquitectura de código)

- Python 3.14 con **numpy/scipy/matplotlib** Solamente: sin torch, sin pandas,
  sin sklearn, sin pytest. → La BNN es un **MLP-ensemble implementado a mano
  en numpy con MC-dropout**; los tests corren con un mini-runner inline
  (`importlib` + `inspect`) en lugar de pytest; el CSV se lee con `csv` de la
  stdlib.
- El cluster **MC2** (8 núcleos): máximo **7 simulaciones simultáneas**, jobs
  largos con `nohup … & wait` (nunca `& disown`), sesión SSH conectada mientras
  vivan los jobs. Regla heredada de BIOPSO y vigente desde el día uno.
- MC2 dejó de responder SSH desde el 2026-11-07 (DNS/Tailscale) — por eso la
  verificación de fórmulas del paso 03 se hizo por API de GitHub, y por eso la
  transferencia de datos de la coleta (paso 09) es ahora el cuello crítico.

## ¿Qué quedó abierto?

- Un pipeline local que pueda desarrollarse SIN MC2 → nace la idea del
  MockSimulator (paso 02).
- Confirmar que PSO/FOA existen en BIOPSO y con qué fórmulas → paso 03.
