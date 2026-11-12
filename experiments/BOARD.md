# BOARD de jobs MC2 (regla: registrar ANTES de lanzar; actualizar al terminar)

Formato de fila: id, brazo, config, seeds, sims totales, lanzado (UTC), terminado,
estado (CORRIENDO/OK/PARCIAL/CAIDO), n sims ok, notas/reintentos.

| id | brazo | config | seeds | sims | lanzado | terminado | estado | ok | notas |
|----|-------|--------|-------|------|---------|-----------|--------|----|-------|
| (vacio — la demo toy local no usa este board) | | | | | | | | | |

## Recordatorios MC2 (del HANDOFF BIOPSO)
- Max 7 sims simultaneas (8 nucleos, nunca saturar). `--max-processes 7`.
- Sesion SSH CONECTADA mientras vivan los jobs: `nohup ... & wait $!`, NUNCA `& disown`.
- WSL se reinicia por inactividad; `.wslconfig` con `vmIdleTimeout=-1`.
- Results/progress en /mnt/c (sobreviven); logs en /tmp se pierden.
- Antes de la coleta: `configs/env/audit_env.sh` (flow 2026.04, env chronosunb).
- Verificar `python --version` -> 3.11.17 con env activado.
