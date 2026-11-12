# Notas de estudio dirigido (vinculadas al cronograma S1–S6)

Cada nota se escribe JUSTO ANTES de la tarea que la necesita (aprender-haciendo).
Estas notas alimentan el relatório de 2 páginas (uso de IA en el desarrollo) y
la defensa oral. Formato sugerido: 1 página por tema — qué dice el paper/curso,
qué fórmula uso en `src/resbo/`, qué pregunta abierta queda.

| Semana | Tema | Nota | Estado |
|---|---|---|---|
| S1 | BO fundamentos (GP/BNN surrogate, EI/UCB, batch q) | `s1_bo.md` | pendiente |
| S1 | BioBO (ICLR 2026) — la innovación que se sustituye | `s1_biobo.md` | pendiente |
| S1 | Baselines del repo BIOPSO: PSO (w const 0.7, c1=c2=1.5) y FOA (Pan 2012, elitismo de centro unico) — que hacen los hiperparametros y por que FOA es el mas simple | aprendido construyendo pso.py/foa.py (2026-11-07); nota por escribir `s1_baselines.md` | iniciada |
| S2 | CRM: Sayarpour 2009; You-Albertazz GCPRM 2019 | `s2_crm.md` | **iniciada** (ver abajo) |
| S2 | BNN: MC-Dropout (Gal 2016), deep ensembles (Lakshminarayanan 2017) | `s2_bnn.md` | pendiente |
| S3 | Prior-informed BO: DynMeanBO (ICLR 2026) | `s3_dynmeanbo.md` | pendiente |
| S3 | Mann-Whitney, tamaño de efecto (rank-biserial), potencia N=20 | `s3_stats.md` | pendiente |
| S4 | Related work: SAEAs + tesis Freitas (OAAMS: proxy EN laço vs prior EN adquisición) | `s4_related.md` | pendiente |

## Nota iniciada — S2 CRM (aprendida construyendo prior/crm.py, 2026-11-06)

**Qué dice el modelo:** producción de un productor j es la suma de los aportes
de cada inyector i ponderados por una conectividad τ_ij, más una declinación
propia:  q_j(t) = Σ_i τ_ij·I_i(t) + (1−Στ_ij)·q_j⁰. En régimen permanente con
controles constantes, q_j → Σ_i τ_ij·rate_i: **lineal en las tasas** → se puede
ajustar por mínimos cuadrados apilando runs (la excitación viene de que cada
run tiene rates DISTINTOS; dentro del run son constantes).

**Lo que mostró el toy (tras fixes de orientación de curvas):** tanto con runs
LHS como con los runs de MFO/GWO, `tau_fit` recuperó `tau_true` (corr 1.0 en
ambos casos; el mock es estrictamente lineal-estacionario en el late-time).
El test `test_crm_recovers_connectivity_trend` exige corr > 0.4 con 40 runs
LHS. Ojo: en el simulador REAL la identificación será más sucia (ruido de
simulación, poços con declinación mutua, controles constantes dentro del run)
→ mantener la precaución de **incluir un DoE latino en la coleta S1** y
reportar el R² por poço como diagnóstico de identificabilidad.

**Pendiente de estudiar:** variante GCPRM (You-Albertazz) con constantes de
tiempo por arista (nuestro mock tiene un único lag=24; el CRM completo ajusta
τ_c_ij por arista) — decidir en S2 si la versión simple basta para el prior.
