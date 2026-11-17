# Paso 13 — Decisiones del ejecutor: anexo de alcance en la proposta, gate S2 confirmado y plan de datos

**Fecha:** 2026-11-17 · **Estado:** ✔ cerrado (docs sincronizados en los 4
formatos; dos decisiones de proyecto formalizadas)

## 1. Por qué este paso va solo

Tres temas que no son código pero decantan el experimento: cómo se reconcilia
la proposta presentada con el experimento real de 4 baselines, qué criterio
rige el upgrade GCPRM-lite de S4, y cómo llegan por fin los datos. Se
preguntaron juntos al ejecutor y las respuestas cierran flechas que venían
abiertas desde pasos anteriores.

## 2. A decisão: anexo, no reescritura

Opciones sobre la mesa para el "MFO e GWO" de la proposta vs los 4 baselines
reales (MFO/GWO/PSO/FOA): actualizar el cuerpo, añadir un anexo, o no tocar
nada. El ejecutor eligió **anexo de escopo atualizado**. El argumento de
fondo es de honestidad documental: la proposta viajó al profesor como fue
presentada el 04/11; reescribirla a posteriori borraría el registro de lo que
se aprobó. El anexo, fechado (17/11/2026), deja el cuerpo intacto y dice en
una página: los baselines pasaron a cuatro porque son las implementaciones
reales del ambiente BIOPSO/STORMS; que la base de entrenamiento y el fit CRM
absorben también PSO/FOA + el DoE de 60 sims; que el presupuesto por
ejecución (90 = 15×6) NO cambia; que la comparación final incluye los 4
baselines + ResBO con prior + ablación sin prior, N=20, Mann-Whitney. La
pregunta de pesquisa y la ancoragem BioBO quedan declaradas inalteradas.

Efectos materiales:
- `Definicao_Projeto_RNAP_JaimeDaniel_ResBO.md` y `.html` reciben la sección
  "Anexo — Escopo Atualizado da Experimentação (17/11/2026)"; `.pdf` y
  `.docx` regenerados desde el HTML con libreoffice headless (misma pipeline
  del paso 01: sin pandoc ni wkhtmltopdf).
- Los cuatro archivos ASUMETIDOS quedaron respaldados en
  `version_submetida_2026-11-04/` antes de tocar nada. Verificación:
  `pdftotext` y lectura del `document.xml` del docx confirman el anexo.

## 3. Gate S2 del upgrade GCPRM-lite: confirmado tal cual

El criterio que venía pendiente de confirmación explícito desde S2 quedó
formalizado: se hace el upgrade a GCPRM-lite (conectividad con gradiente de
score) **solo si Δcorr(score, NPV) > 0.1 Y R² ≥ 0.7**; en cualquier otro
escenario se congela CRM-lite y el artigo reporta el resultado tal como
salió. Anotado en la fila S2 del cronograma. Es el tipo de decisión que
conviene fijar ANTES de ver los números — y justo por eso estaba pendiente.

## 4. Plan de datos: SSH a MC2

El ejecutor restaurará el SSH a MC2 (caído desde 07/11 por DNS/Tailscale).
El comando de transferencia quedó anotado en el cronograma (fila Lun 17 –
Mié 19/11):

```
rsync -avP MC2:/mnt/c/DAGO/BIOPSO/experiments/{coleta_s1,harvest_2026_11}/ resbo/experiments/
```

Con los árboles locales, la cola crítica se desbloquea en el orden ya
preparado: `check_simlog.py` + `validate_loader.py` → `fit_crm.py --harvest`
→ `train_bnn.py --harvest` (gate NRMSE) → `export_deploy.py` (sin `--smoke`)
→ rsync del deploy a MC2 + smokes S3 → lanzamiento de los brazos. Todo el
código de esa cadena existe, está testeado y pusheado.

## 5. Estado de las flechas abiertas del projeto

| flecha | estado tras este paso |
|---|---|
| sync proposta ↔ 4 baselines | ✔ anexo en los 4 formatos, originales respaldados |
| gate GCPRM-lite (S4) | ✔ confirmado tal cual (Δcorr > 0.1 Y R² ≥ 0.7) |
| llegada de `coleta_s1`/`harvest_2026_11` | ⏳ en manos del ejecutor (restaurar SSH; comando anotado) |
| rotación del GH_TOKEN expuesto alguna vez | ⏳ sigue pendiente (higiene, no bloquea) |
