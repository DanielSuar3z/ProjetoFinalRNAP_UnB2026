"""Verificador del contrato simlog_spec_v1 (T-R2).

Valida runs/artbolos BIOPSO pasandolos por el loader estricto y reporta
conteos, shapes y el cruce de invariantes. Sale con codigo != 0 si algo
viola la spec (o si `--self-check` falla).

Uso:
    python scripts/validate_loader.py <ruta_run_o_harvest> [--seed N]
    python scripts/validate_loader.py --self-check     # fixture sintetico
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from resbo.database import SimDatabase                        # noqa: E402
from resbo.loader_biopso import (SpecViolation, load_harvest,  # noqa: E402
                                 load_run, make_synthetic_run)


def _report(db: SimDatabase):
    by_algo = {}
    for r in db.records:
        by_algo.setdefault(r.algo, []).append(r)
    for algo, recs in sorted(by_algo.items()):
        X = np.vstack([r.controls for r in recs])
        y = np.array([r.npv for r in recs])
        curves = [r.prod_curve for r in recs if r.prod_curve is not None]
        iters = sorted({r.iteration for r in recs})
        print(f"  [{algo}] {len(recs)} filas | X {X.shape} | "
              f"NPV [{y.min()/1e6:.0f}, {y.max()/1e6:.0f}] M | "
              f"iteraciones {iters[:6]}{'...' if len(iters) > 6 else ''}")
        if curves:
            shp = {tuple(c.shape) for c in curves}
            print(f"         curvas: {len(curves)}/{len(recs)} filas con "
                  f"prod_curve, shapes {shp}")
        bad = np.sum(~np.isfinite(y)) + int(np.sum(~np.isfinite(X)))
        if bad:
            print(f"  !! {bad} valores no finitos")
    return db


def _self_check() -> int:
    from resbo.config import Protocol
    from resbo.simulator import MockSimulator
    proto = Protocol.load(os.path.join(ROOT, "configs", "protocol.yaml"))
    sim = MockSimulator(proto, seed=42)
    rc = 0
    with tempfile.TemporaryDirectory() as td:
        # ruta A: simlog + npz
        run_a = make_synthetic_run(os.path.join(td, "random", "run_01"),
                                   sim, proto, seed=1, n_sims=12)
        db = SimDatabase(os.path.join(td, "db_a"))
        n = load_run(run_a, "random", seed=0, db=db)
        assert n == 12 and len(db.records) == 12, f"ruta A: {n} filas"
        assert all(len(r.prod_curve) == 4 * proto.horizon for r in db.records)
        print("ruta A (simlog+npz): OK")
        # ruta A via arbol completo
        db2 = SimDatabase(os.path.join(td, "db_a2"))
        total = load_harvest(td, db2)
        # ruta B (rescate) se crea dentro del mismo arbol y se cuenta aparte
        # ruta B: rescate con sidecar
        run_b = make_synthetic_run(os.path.join(td, "mfo", "run_02"),
                                   sim, proto, seed=2, n_sims=8, rescue=True)
        db3 = SimDatabase(os.path.join(td, "db_b"))
        n_b = load_run(run_b, "mfo", seed=0, db=db3)
        assert n_b == 1 and db3.records[0].iteration == -1
        assert len(db3.records[0].prod_curve) == 4 * proto.horizon
        assert np.any(db3.records[0].prod_curve > 0), "curva de rescate toda-ceros"
        print("ruta B (rescate well_curves + best_controls.json): OK")
        # cabecera corrupta -> SpecViolation
        bad = os.path.join(td, "roto")
        os.makedirs(bad)
        with open(os.path.join(bad, "simlog.csv"), "w") as f:
            f.write("iter,agente,x1\n1,0,1.0\n")
        try:
            load_run(bad, "x", 0, SimDatabase(os.path.join(td, "db_c")))
        except SpecViolation as e:
            print(f"cabecera corrupta rechazada: OK ({str(e).splitlines()[0]}...)")
        else:
            print("  !! cabecera corrupta NO rechazada")
            rc = 1
        # wpr todo-ceros -> SpecViolation
        zp = os.path.join(td, "zeros")
        os.makedirs(zp)
        import csv as _csv
        from resbo.loader_biopso import SPEC_HEADER
        with open(os.path.join(zp, "simlog.csv"), "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(SPEC_HEADER)
            w.writerow([1, 0] + ["1.0"] * 12 + ["1.0"])
        np.savez_compressed(os.path.join(zp, "sim_curves.npz"),
                            wpr=np.zeros((1, 4, 130)), wir=np.ones((1, 8, 130)),
                            time_days=np.arange(130.0), sim_index=np.arange(1),
                            producer_wells=np.array(["a", "b", "c", "d"]),
                            injector_wells=np.array([f"i{i}" for i in range(8)]))
        try:
            load_run(zp, "x", 0, SimDatabase(os.path.join(td, "db_d")))
        except SpecViolation as e:
            print(f"wpr todo-ceros rechazado: OK ({str(e).splitlines()[0]}...)")
        else:
            print("  !! wpr todo-ceros NO rechazado")
            rc = 1
    return rc


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", help="run_XX/ o arbol de coleta")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        sys.exit(_self_check())
    if not args.path:
        ap.error("falta <ruta> o --self-check")

    db = SimDatabase(os.path.join(tempfile.gettempdir(), "validate_loader_db"))
    try:
        if os.path.exists(os.path.join(args.path, "simlog.csv")) or \
           os.path.exists(os.path.join(args.path, "well_curves.csv")):
            n = load_run(args.path, "desconocido", args.seed, db)
            print(f"run unico: {n} filas")
        else:
            n = load_harvest(args.path, db, seed=args.seed)
            print(f"harvest {args.path}: {n} filas")
    except SpecViolation as e:
        print(f"SPEC VIOLATION:\n{e}")
        sys.exit(1)
    _report(db)
    print("CONTRATO OK" if db.records else "VACIO")


if __name__ == "__main__":
    main()
