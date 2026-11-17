# Paso 14 — El rescate cruza el cable; y la coleta S1 resulta que no había corrido

**Fecha:** 2026-11-17 (tarde) · **Depende de:** paso 13 · **Commit ResBO:** ver CHANGELOG

El paso 13 cerró con una sola cosa en la cola crítica: el SSH a MC2 volvió
("ssh ya volvió") y había que traer los datos. Este paso hace la parte del
rescate que sí existía y descubre, buscándola, que la parte grande **no existe
todavía**: la coleta S1 nunca se lanzó. Los dos hechos cambian el cronograma
en direcciones opuestas, y ambos quedan documentados.

## 1. Cómo se entra hoy a MC2 (y por qué no es un ssh normal)

El host `mc2` del `~/.ssh/config` responde, pero el shell por defecto del
servidor es **PowerShell**, no bash: cualquier comando con `2>/dev/null` o
comillas anidadas se corrompe (PowerShell intenta abrir `C:\dev\null`). El
patrón de trabajo que quedó establecido es meter el script por stdin dentro de
WSL:

```bash
printf '%s\n' 'comando1' 'comando2' \
  | ssh mc2 'wsl.exe -e bash -lc "cat > /tmp/q.sh; bash /tmp/q.sh"'
```

Así PowerShell solo ve un comando trivial y el bash real se come el script
entero sin interpretar comillas. Para **transferencia de archivos binarios**
este patrón NO sirve (PowerShell re-codifica stdout y rompería un tar), y en
cambio funcionó directamente `scp` — el servidor OpenSSH de Windows atiende
SFTP por su cuenta, sin pasar por el shell:

```bash
scp -r "mc2:C:/DAGO/BIOPSO/experiments/harvest_2026_11" ./harvest_2026_11
```

Lección para futuras transferencias: texto → patrón `wsl.exe` por stdin;
bytes → `scp`/SFTP con rutas Windows (`C:/...`).

## 2. El rescate `harvest_2026_11` ya está en local, verificado

Procedimiento (en este orden, que es el correcto para no firmar un checksum
propio):

1. En MC2, dentro de la carpeta del rescate: `find . -type f | sort | xargs
   sha256sum > SHA256SUMS`.
2. `scp -r` del árbol completo (42 MB) a `resbo/experiments/harvest_2026_11/`.
3. Local: `sha256sum -c` contra el manifiesto remoto → **8/8 archivos OK**
   (los 2 `*_progress.pkl` con las curvas de 18 sims, los 4 `results.csv` con
   las 42 filas de NPV de PSO/StoSAG n20 y smokes, `MANIFIESTO.md` y
   `rescate_dago.sh`).
4. Borrar el `SHA256SUMS` temporal en ambos lados (tenía una línea
   autorreferente: `find` lo enlistó a mitad de su propia creación — harmless
   pero lo sacamos para no dejar manifiestos sucios).

`.gitignore` se actualizó para la política de siempre: `experiments/harvest_*/`
y `experiments/coleta*/` quedan fuera del repo — los datos del motor privado
viajan por scp/rsync con sha256, el repo final es público.

Qué permite este rescate HOJE, mientras S1 no existe: la **ruta B** del loader
(extracción de curvas por-sim desde los pkl con el parser T-B4, proceso
aislado por el riesgo H-07 de segfault) y ver el pipeline `fit_crm`/`train_bnn`
con 18 sims reales… que no alcanzan para el gate (se necesitan ≥30 para export
y mucho más para NRMSE < 5%), pero sí para desbugar la cadena con datos
auténticos en vez de toy.

## 3. La búsqueda de `coleta_s1`: exhaustiva y negativa

Se buscó en MC2 con el patrón del §1:

| Ruta | Resultado |
|---|---|
| `/mnt/c/DAGO/BIOPSO/experiments/` | solo `harvest_2026_11` + smokes pre-contrato (`mfo/gwo_smoke_jaime`: `results.csv` + runs, **sin `simlog.csv`**) |
| `/mnt/c/JAIME/` | solo `Scripts` (los directorios `02-Proyectos`/`05-Datos-crudos` están en `Disco D`) |
| `/mnt/c/Disco D/` (archivo) | inventarios y docs, nada de runs |
| `/mnt/d` | discos de modelos (P50/P90…), sin `simlog.csv` |
| huella `find … -name simlog.csv` en los 4 montos + `/home` | **cero resultados** |

Preguntado el ejecutor: **"la coleta aún no se ha corrido"**. Es decir: no
estaba escondida — nunca se generó. El cronograma decía "coleta S1 lanzada ✔"
(hecho asumido del paso 9) y era falso como estado de datos. Se corrigió en el
mismo cronograma (re-ancoraje v2 de hoy), y queda la leña al fuego del método:
*un hito no está ✔ porque lo escribimos una vez; está ✔ cuando el archivo está
en disco y el checker sale en OK.*

Además: el host `ssh.laboratorioiot.online` (túnel cloudflared, presumiblemente
el PC nuevo) responde — cloudflared instalado y túnel vivo — pero la
autenticación rechaza al usuario `jaime_daniel` (`Permission denied
(publickey,password)`). Habrá que poner usuario correcto + clave autorizada, o
ejecutar la coleta con sesión local en ese PC.

## 4. Re-ancoraje del calendario (lo que esto implica)

El calendario del paso 9 daba por hecho que S1 venía corriendo y hoy solo se
transfería. Con S1 partiendo de cero, y ~7 min/sim a 7 procesos (≈57 s/sim de
pared):

- **Lun 17 (hoy):** PC nuevo — verificar entorno (`ENVIRONMENT_SETUP.md`),
  `git checkout feature/bayes-crm-ingesta`, `test_simlog.py`, los 4 smokes del
  §2 de `COLECTA_S1.md` con `check_simlog.py` → lanzar el primer brazo.
- **Mar 18 – Sáb 21:** los 1.920 sims ≈ 30–35 h de pared (un brazo por sesión
  `nohup ... & wait`, NUNCA disown). En paralelo local: nada bloqueado — los
  scripts de S2 ya existen y apuntan a `--harvest`.
- **Dom 22 – Lun 23:** transferencia del lote al local + checker a los 22 runs
  + `validate_loader.py` real.
- **Mar 24 – Jue 26:** S2 (fit_crm → gate R² por pozo; train_bnn → gate NRMSE
  < 5%; decisión GCPRM-lite con el criterio confirmado en el paso 13).
- **Vie 27 – Lun 01/12:** S3 smokes + `export_deploy` real + brazos BO (≈28 h
  de pared por brazo; contingencia n_seeds 20→16→12).
- **Mar 02 – 09/12:** análisis (`compare_arms.py`), artigo, relatório, oral.

La fecha de entrega **no se mueve**, pero el colchón se acabó: cada día que
retrasa el lanzamiento de S1 sale del tiempo de escritura, no del de máquina.
La palanca de contingencia número uno ahora es **cuándo se sienta uno al PC
nuevo**, no las perillas del presupuesto.

## 5. ¿Qué quedó abierto?

| flecha | estado tras este paso |
|---|---|
| `harvest_2026_11` en local | ✔ transferido y verificado sha256 (8/8), gitignored |
| **lanzar coleta S1 en el PC nuevo** | ⛔ la tarea #1 del proyecto entero (gate: smokes §2 en CHECK_OK + brazo `mfo` corriendo a más tardar Mar 18) |
| acceso SSH al PC nuevo | ⏳ túnel cloudflared vivo pero auth rechazada; o ejecutar localmente en ese PC |
| GH_TOKEN rotado | ⏳ sigue pendiente (higiene, no bloquea) |
