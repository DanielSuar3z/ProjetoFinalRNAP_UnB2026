"""Agregado de los 6 brazos para el artigo (S5/T-R6): tabla + estadisticos.

Lee las results.csv que deja el harness de BIOPSO por brazo (coleta S1:
mfo/gwo/pso/foa; S3: bo_prior/bo_solo) y escribe en --out-dir:

  arms_summary.csv   n, media, mediana, sd, min, max, AUC (sims), runs s/curvas
  pairwise.csv       Mann-Whitney U + p + rank-biserial por comparacion
  arms_table.md      las dos tablas en markdown (directas al artigo IEEE)
  arms_boxplot.png   figura (solo si matplotlib esta instalado)

Uso (con el layout real de MC2, despues de traer los arboles):
    python scripts/compare_arms.py \\
        --arm mfo=experiments/coleta_s1/mfo \\
        --arm gwo=experiments/coleta_s1/gwo \\
        --arm pso=experiments/coleta_s1/pso \\
        --arm foa=experiments/coleta_s1/foa \\
        --arm bo_prior=experiments/s3/bo_prior \\
        --arm bo_solo=experiments/s3/bo_solo \\
        --ref bo_prior [--threshold 1.6e7]

Sin brazos por linea de comandos corre --self-check: brazos sinteticos con
corrimiento conocido que prueban el pipeline completo (direccion del efecto,
geometria del simlog->AUC, rescate sin curvas, run_id duplicado). Igual que
fit_crm/train_bnn: el script se auto-valida hoy, manana solo cambia la ruta.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from resbo.analysis.compare import (ARMS_CANON, load_arm,  # noqa: E402
                                    run_compare)

CABECERA_SIMLOG = ("iteration,agent,inje_1,inje_2,inje_3,inje_4,inje_5,inje_6,"
                   "inje_7,inje_8,prod_1,prod_2,prod_3,prod_4,npv")


def _self_check() -> int:
    """Fixture con verdad conocida: bo_prior domina a foa (corrimiento
    +2e6), geometria resbo (ronda 0 + 2x3) en un brazo y clasica (3x2) en el
    otro, un run de rescate (sin curvas) y un run_id duplicado."""
    d = tempfile.mkdtemp(prefix="compare_selfcheck_")

    def brazo(nombre, base_npv, n_runs, n_init, agents, iters,
              rescate_ids=(), duplicado=None):
        bdir = os.path.join(d, nombre)
        os.makedirs(bdir, exist_ok=True)
        rng = np.random.default_rng(abs(hash(nombre)) % 2**31)
        filas = []
        for rid in range(1, n_runs + 1):
            npv = base_npv + rng.normal(0, 2e5)
            filas.append((rid, npv))
            if rid in rescate_ids:           # rescate: solo fila en results
                continue
            rdir = os.path.join(bdir, f"run_{rid:02d}")
            os.makedirs(rdir, exist_ok=True)
            acum = -np.inf
            conv = ["iteration,best_npv"]
            slog = [CABECERA_SIMLOG]
            rondas = ([(0, n_init)] if n_init else []) + \
                     [(it, agents) for it in range(1, iters + 1)]
            for it, n_ag in rondas:
                vals = npv - rng.random(n_ag) * 1e5   # techo: el mejor <= npv
                acum = max(acum, float(vals.max()))
                for k, v in enumerate(vals):
                    slog.append(f"{it},{k}," + ",".join("79.5" for _ in range(12))
                                + f",{float(v)!r}")
                if it > 0:                   # convergence: solo 1..iters
                    conv.append(f"{it},{acum!r}")
            with open(os.path.join(rdir, "convergence.csv"), "w") as f:
                f.write("\n".join(conv) + "\n")
            with open(os.path.join(rdir, "simlog.csv"), "w") as f:
                f.write("\n".join(slog) + "\n")
        if duplicado:                                  # re-intento: gana el último
            filas.append((duplicado[0], duplicado[1]))
        with open(os.path.join(bdir, "results.csv"), "w") as f:
            f.write("run_id,npv_final,elapsed_seconds,timestamp\n")
            for rid, npv in filas:
                f.write(f"{rid},{npv:.2f},1.0,2026-11-17 00:00:00\n")
        return bdir

    # bo_prior: geometria resbo (4 + 2x3 = 10 sims), con rescate run 5
    b_pri = brazo("bo_prior", 1.8e7, 6, n_init=4, agents=2, iters=3,
                  rescate_ids=(5,))
    # foa: geometria clasica (3x2 = 6 sims), mas debil
    b_foa = brazo("foa", 1.6e7, 6, n_init=0, agents=3, iters=2)
    # run_id duplicado en gwo: el ultimo valor es el que debe quedar
    b_gwo = brazo("gwo", 1.7e7, 4, n_init=0, agents=3, iters=2,
                  duplicado=(2, 9.9e7))

    arms = [load_arm("bo_prior", b_pri), load_arm("gwo", b_gwo),
            load_arm("foa", b_foa)]
    a_foa, a_pri, a_gwo = arms[2], arms[0], arms[1]

    # 1) duplicado: gana la ultima fila
    assert 9.9e7 in list(a_gwo.npvs), "run_id duplicado: no tomo el re-intento"
    assert a_gwo.y.size == 4
    # 2) AUC en sims: geometria resbo pliega la ronda 0 (10 sims), la
    #    clasica 6; la AUC del fixture debe venir de historias completas
    assert len(a_pri.histories) == 5 and a_pri.sin_curvas == [5]
    h = a_pri.histories[1]
    assert h[-1][0] == 10, f"AUC resbo deberia ser a 10 sims, es {h[-1][0]}"
    assert a_foa.histories[1][-1][0] == 6
    # 3) estadistica: bo_prior > foa con efecto positivo y p pequeno
    res = run_compare(arms, os.path.join(d, "out"), ref="bo_prior")
    par = {p["comparacion"]: p for p in res["pares"]}
    pfoa = par["bo_prior vs foa"]
    assert pfoa["r_biserial"] > 0.9 and pfoa["p_valor"] < 0.05, pfoa
    assert par["bo_prior vs gwo"]["r_biserial"] > 0, "direccion equivocada"
    # 4) artefactos
    for fn in ("arms_summary.csv", "pairwise.csv", "arms_table.md"):
        assert os.path.exists(os.path.join(res["out_dir"], fn)), fn
    assert "bo_prior" in res["markdown"] and "r_rb" in res["markdown"]
    fila_pri = [f_ for f_ in res["filas"] if f_["arm"] == "bo_prior"][0]
    assert fila_pri["runs_sin_curvas"] == 1, "el rescate debe figurar, no desaparecer"
    print(f"self-check OK: {len(res['filas'])} brazos, {len(res['pares'])} pares, "
          f"bo_prior vs foa r_rb={pfoa['r_biserial']:+.3f} p={pfoa['p_valor']:.1e}")
    print(f"  (brazos canonicos del experimento: {', '.join(ARMS_CANON)})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", default=[], metavar="NOMBRE=RUTA",
                    help="brazo: dir con results.csv (o path al csv); repetible")
    ap.add_argument("--ref", default="bo_prior",
                    help="brazo objetivo de las comparaciones (default bo_prior; "
                         "si no esta entre los dados, compara todos los pares)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="umbral de NPV para sims-to-threshold (opcional)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "experiments",
                                                      "compare"))
    ap.add_argument("--self-check", action="store_true",
                    help="pipeline contra fixture sintetico (sin datos reales)")
    args = ap.parse_args()

    if args.self_check or not args.arm:
        sys.exit(_self_check())

    arms = []
    for espec in args.arm:
        if "=" not in espec:
            sys.exit(f"error: --arm pide NOMBRE=RUTA, recibi '{espec}'")
        nombre, ruta = espec.split("=", 1)
        arms.append(load_arm(nombre, ruta))
    descon = [a.name for a in arms if a.name not in ARMS_CANON]
    if descon:
        print(f"aviso: nombres fuera del canon: {descon}")
    n_min = min(a.y.size for a in arms)
    if n_min < 3:
        print(f"aviso: hay brazo con n={n_min}; Mann-Whitney no tiene poder "
              "a ese tamano (reportar solo descriptivos)")
    res = run_compare(arms, args.out_dir, ref=args.ref,
                      threshold=args.threshold)
    print(res["markdown"])
    print(f"escrito en {res['out_dir']}"
          + (f" (figura: {os.path.basename(res['png'])})" if res["png"] else
             " (sin figura: matplotlib no disponible)"))
    for p_ in res["pares"]:
        if p_["aviso_n_bajo"]:
            print(f"aviso: {p_['comparacion']} con n bajo "
                  f"({p_['n_x']}/{p_['n_y']}) -> el p aproxima mal, usar r_rb")


if __name__ == "__main__":
    main()
