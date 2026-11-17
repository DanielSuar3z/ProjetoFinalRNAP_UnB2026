"""Loader de artefactos BIOPSO -> SimDatabase (contrato simlog_spec_v1).

Fuentes aceptadas por `load_run(work_dir, ...)`:

  A) Coleta con parche simlog (T-B1/B2):   <run_XX>/simlog.csv + sim_curves.npz
     -> una SimRecord por simulacion (controles 12d, NPV, curva WPR (n_prod,T)).
  B) Rescate de corridas antiguas (T-B4):  <run_XX>/well_curves.csv + results.csv
     -> 1 SimRecord por corrida (controles ganadores de best_controls.json,
        NPV de results.csv, curvas de well_curves.csv). Los controles NO viven
        en well_curves.csv: el harvest debe dejar el sidecar best_controls.json
        (extracto del *_progress.pkl) o pasarlos por argumento `controls`.

Todos los desacuerdos con la spec levantan `SpecViolation` (fallo ruidoso,
nunca silencioso): la cabecera, los shapes y el orden son PARTE del contrato.

`make_synthetic_run` genera runs sinteticos que cumplen la spec al 100% —
fixtures para tests y para probar el pipeline antes de tener datos MC2.
"""
from __future__ import annotations

import csv
import json
import os
import re

import numpy as np

from .database import SimDatabase

# Cabecera congelada por docs/estudo/simlog_spec_v1.md §1
SPEC_HEADER = ["iteration", "agent",
               "inje_1", "inje_2", "inje_3", "inje_4",
               "inje_5", "inje_6", "inje_7", "inje_8",
               "prod_1", "prod_2", "prod_3", "prod_4", "npv"]

NPZ_KEYS = ("wpr", "wir", "time_days", "sim_index",
            "producer_wells", "injector_wells")


class SpecViolation(RuntimeError):
    """El artefacto de BIOPSO no cumple simlog_spec_v1."""


# --------------------------------------------------------------------- simlog
def _read_simlog(path: str):
    with open(path) as f:
        rd = csv.reader(f)
        header = next(rd)
        if header != SPEC_HEADER:
            raise SpecViolation(
                f"cabecera de simlog.csv no coincide con simlog_spec_v1:\n"
                f"  esperado: {','.join(SPEC_HEADER)}\n"
                f"  leido:    {','.join(header)}")
        rows = []
        for ln, row in enumerate(rd, start=2):
            if not row:
                continue
            try:
                it, ag = int(row[0]), int(row[1])
                ctrl = np.array([float(v) for v in row[2:14]], dtype=float)
                npv = float(row[14])
            except (ValueError, IndexError) as e:
                raise SpecViolation(f"simlog.csv linea {ln} mal formada: {e}")
            rows.append((it, ag, ctrl, npv))
    if not rows:
        raise SpecViolation(f"{path}: simlog.csv sin filas de datos")
    return rows


def _read_npz(path: str, n_rows: int, n_prod: int, n_inj: int):
    if not os.path.exists(path):
        raise SpecViolation(f"falta sim_curves.npz ({path}) junto a simlog.csv")
    z = np.load(path, allow_pickle=False)
    missing = [k for k in NPZ_KEYS if k not in z.files]
    if missing:
        raise SpecViolation(f"sim_curves.npz sin claves {missing}")
    wpr, wir = z["wpr"], z["wir"]
    errs = []
    if wpr.ndim != 3 or wir.ndim != 3:
        errs.append(f"ndim wpr/wir = {wpr.ndim}/{wir.ndim}, esperado 3")
    else:
        if wpr.shape[0] != n_rows or wir.shape[0] != n_rows:
            errs.append(f"n_sims npz {wpr.shape[0]}/{wir.shape[0]} != "
                        f"{n_rows} filas del simlog")
        if wpr.shape[1] != n_prod:
            errs.append(f"wpr tiene {wpr.shape[1]} productores, esperado {n_prod}")
        if wir.shape[1] != n_inj:
            errs.append(f"wir tiene {wir.shape[1]} inyectores, esperado {n_inj}")
        if wpr.shape[2] != wir.shape[2]:
            errs.append(f"n_t distinto entre wpr ({wpr.shape[2]}) y wir ({wir.shape[2]})")
        if np.all(wpr == 0):
            errs.append("wpr todo-ceros: falta WWPR en el SUMMARY del deck")
        if np.all(wir == 0):
            errs.append("wir todo-ceros: falta WWIR en el SUMMARY del deck")
    if not np.array_equal(z["sim_index"], np.arange(n_rows)):
        errs.append("sim_index no es arange(n_sims): el orden npz<->simlog esta roto")
    if errs:
        raise SpecViolation("sim_curves.npz invalido:\n  - " + "\n  - ".join(errs))
    return np.asarray(wpr, dtype=float), np.asarray(z["time_days"], dtype=float)


def load_run(work_dir: str, algo: str, seed: int, db: SimDatabase,
             n_prod: int = 4, n_inj: int = 8,
             controls: np.ndarray | None = None) -> int:
    """Carga un run_XX/ de BIOPSO en `db`. Devuelve nº de SimRecords añadidas."""
    simlog = os.path.join(work_dir, "simlog.csv")
    if os.path.exists(simlog):
        rows = _read_simlog(simlog)
        wpr, _ = _read_npz(os.path.join(work_dir, "sim_curves.npz"),
                           len(rows), n_prod, n_inj)
        for i, (it, _ag, ctrl, npv) in enumerate(rows):
            db.add(algo, seed, it, ctrl, npv, prod_curve=wpr[i])
        return len(rows)
    return _load_rescue_run(work_dir, algo, seed, db, n_prod, controls)


def _load_rescue_run(work_dir: str, algo: str, seed: int, db: SimDatabase,
                     n_prod: int, controls: np.ndarray | None) -> int:
    """Ruta B (rescate T-B4): sin simlog -> 1 fila con curvas de well_curves.csv."""
    wc = os.path.join(work_dir, "well_curves.csv")
    if not os.path.exists(wc):
        raise SpecViolation(
            f"{work_dir}: no hay simlog.csv ni well_curves.csv — no es un run BIOPSO")

    # NPV: results.csv de la corrida (run_id del nombre run_XX). En BIOPSO el
    # results.csv vive en el padre del run (experiments/<tag>/results.csv).
    npv = None
    run_id = None
    m = re.search(r"run[_-]?(\d+)$", os.path.basename(os.path.normpath(work_dir)))
    if m:
        run_id = int(m.group(1))
    for cand in (os.path.join(work_dir, "results.csv"),
                 os.path.join(os.path.dirname(os.path.normpath(work_dir)), "results.csv")):
        if run_id is None or not os.path.exists(cand):
            continue
        with open(cand) as f:
            for row in csv.DictReader(f):
                if int(row["run_id"]) == run_id:
                    npv = float(row["npv_final"])
                    break
        if npv is not None:
            break
    if npv is None:
        raise SpecViolation(f"{work_dir}: no encuentro npv_final en results.csv "
                            f"para el run_id de la carpeta")

    # Controles ganadores: argumento > best_controls.json (harvest) > error explicito
    sidecar = os.path.join(work_dir, "best_controls.json")
    if controls is None and os.path.exists(sidecar):
        with open(sidecar) as f:
            controls = np.array(json.load(f)["controls"], dtype=float)
    if controls is None:
        raise SpecViolation(
            f"{work_dir}: rescate sin controles. Extrae el vector ganador del "
            f"*_progress.pkl en el harvest y dejalo en best_controls.json "
            f'({{"controls": [12 floats], "npv": ...}}) o pasa controls= a load_run.')
    controls = np.asarray(controls, dtype=float)
    if controls.shape != (n_prod + 8,):
        raise SpecViolation(f"controles {controls.shape}, esperado "
                            f"({n_prod + 8},) [rates x8 + bhp x{n_prod}]")

    # Curvas: well_curves.csv (dia,pozo,tipo,<vars>) -> WPR por productor
    wells: dict[str, dict] = {}
    with open(wc) as f:
        rd = csv.DictReader(f)
        if "pozo" not in (rd.fieldnames or []) or "tipo" not in (rd.fieldnames or []):
            raise SpecViolation(f"{wc}: cabecera sin pozo/tipo (formato curvas.py)")
        for row in rd:
            w = wells.setdefault(row["pozo"], {"tipo": row["tipo"], "WPR": []})
            w["WPR"].append(float(row.get("WPR") or 0.0))
    prod = [w for w, d in wells.items() if d["tipo"] == "PRODUCER"]
    if len(prod) != n_prod:
        raise SpecViolation(f"{wc}: {len(prod)} pozos PRODUCER, esperado {n_prod}")
    lens = {len(wells[w]["WPR"]) for w in prod}
    if len(lens) != 1:
        raise SpecViolation(f"{wc}: longitudes de curva desiguales {lens}")
    prod.sort()                                     # orden determinista (spec §2.3)
    curve = np.vstack([np.asarray(wells[w]["WPR"]) for w in prod])
    db.add(algo, seed, -1, controls, npv, prod_curve=curve)
    return 1


def load_harvest(root: str, db: SimDatabase, seed: int = 0) -> int:
    """Recorre <root>/<algo>/run_XX/ (o run_XX sueltos) y carga todo.

    La etiqueta `algo` se toma del padre si es un tag conocido; si no, del
    propio nombre de carpeta (convención coleta_s1 del spec §5).
    """
    tags = {"mfo", "gwo", "pso", "foa", "hgso", "random", "stosag", "bo", "bo_prior"}
    total = 0
    for dirpath, _dirnames, filenames in sorted(os.walk(root)):
        if "simlog.csv" in filenames or "well_curves.csv" in filenames:
            parent = os.path.basename(os.path.dirname(os.path.normpath(dirpath)))
            algo = parent if parent in tags else os.path.basename(dirpath)
            total += load_run(dirpath, algo, seed, db)
    if total == 0:
        raise SpecViolation(f"{root}: no encontre ningun run BIOPSO "
                            f"(simlog.csv / well_curves.csv)")
    return total


# ----------------------------------------------------- fixtures (tests + demos)
def make_synthetic_run(out_dir: str, sim, protocol, seed: int = 0,
                       n_sims: int = 24, agents_per_round: int = 4,
                       rescue: bool = False) -> str:
    """Genera un run_XX/ que cumple simlog_spec_v1 usando `sim.evaluate` real.

    rescue=True produce ademas el formato viejo (well_curves.csv + results.csv +
    best_controls.json) SIN simlog, para probar la ruta B.
    Devuelve la ruta del run creado.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    lb, ub = protocol.bounds
    n_inj, n_prod = protocol.n_inj, protocol.n_prod
    T = protocol.horizon

    samples = []
    for _ in range(n_sims):
        x = lb + rng.random(len(lb)) * (ub - lb)
        out = sim.evaluate(x)
        samples.append((x, float(out["npv"]), np.asarray(out["prod_curve"], float)))

    if not rescue:
        with open(os.path.join(out_dir, "simlog.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(SPEC_HEADER)
            for i, (x, npv, _c) in enumerate(samples):
                w.writerow([i // agents_per_round + 1, i % agents_per_round]
                           + [repr(float(v)) for v in x] + [repr(npv)])
        time_days = 30.0 * np.arange(1, T + 1)
        wpr = np.stack([c for _x, _n, c in samples])            # (n_sims, n_prod, T)
        wir = np.stack([np.tile(x[:n_inj][:, None], (1, T))      # tasa constante
                        for x, _n, _c in samples])               # (n_sims, n_inj, T)
        np.savez_compressed(
            os.path.join(out_dir, "sim_curves.npz"),
            wpr=wpr, wir=wir, time_days=time_days,
            sim_index=np.arange(n_sims, dtype="int64"),
            producer_wells=np.array([f"PROD-{j+1}" for j in range(n_prod)]),
            injector_wells=np.array([f"INJ-{i+1}" for i in range(n_inj)]))
        return out_dir

    # --- ruta B (rescate): solo artefactos del motor SIN parche ---------------
    best_x, best_npv, best_c = max(samples, key=lambda s: s[1])
    with open(os.path.join(out_dir, "well_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dia", "pozo", "tipo", "WPR", "WIR"])
        days = (30.0 * np.arange(1, T + 1)).astype(int)
        for j in range(n_prod):
            for t in range(T):
                w.writerow([days[t], f"PROD-{j+1}", "PRODUCER",
                            repr(float(best_c[j, t])), "0.0"])
        for i in range(n_inj):
            for t in range(T):
                w.writerow([days[t], f"INJ-{i+1}", "INJECTOR", "0.0",
                            repr(float(best_x[i]))])
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "npv_final", "elapsed_seconds", "timestamp"])
        m = re.search(r"run[_-]?(\d+)$", os.path.basename(os.path.normpath(out_dir)))
        run_id = int(m.group(1)) if m else 1        # run_XX <-> run_id XX (harvest)
        w.writerow([run_id, repr(best_npv), 60.0, "2026-11-17 00:00:00"])
    with open(os.path.join(out_dir, "curves_meta.json"), "w") as f:
        json.dump({"version": 1, "fuente": "harvest", "pasos": T,
                   "pozos": [f"PROD-{j+1}" for j in range(n_prod)]
                           + [f"INJ-{i+1}" for i in range(n_inj)],
                   "variables": ["WPR", "WIR"]}, f)
    with open(os.path.join(out_dir, "best_controls.json"), "w") as f:
        json.dump({"controls": [float(v) for v in best_x], "npv": best_npv,
                   "source": "fixture"}, f)
    return out_dir
