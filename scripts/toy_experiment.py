"""Demo end-to-end de ResBO sobre MockSimulator (sin MC2/OPM).

Pipeline completo del proyecto:
  1. corre los baselines MFO, GWO, PSO y FOA (formulas del repo BIOPSO)
     -> puebla la base de datos
  2. ajusta CRM con las curvas de produccion de esos runs
  3. corre BO con prior CRM y BO sin prior (ablacion) reutilizando los datos
  4. compara best-NPV / sims-al-umbral / AUC + Mann-Whitney (BO+prior vs BO)

Uso:
    cd resbo && python scripts/toy_experiment.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from resbo.analysis import auc, mannwhitney, rank_biserial, sims_to_threshold  # noqa
from resbo.bo import BayesOptimizer  # noqa
from resbo.config import Protocol  # noqa
from resbo.database import SimDatabase  # noqa
from resbo.foa import FOA  # noqa
from resbo.gwo import GWO  # noqa
from resbo.mfo import MFO  # noqa
from resbo.prior import CRMModel  # noqa
from resbo.pso import PSO  # noqa
from resbo.simulator import MockSimulator  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="menos seeds/iters")
    args = ap.parse_args()

    proto = Protocol.load(os.path.join(ROOT, "configs", "protocol.yaml"))
    if args.quick:
        proto.d["budget"]["metaheuristics"] = {"pop": 6, "iters": 3}
        proto.d["budget"]["bo"] = {"batch": 3, "iters": 4, "pool": 800}
        proto.d["n_seeds"] = 2

    outdir = os.path.join(ROOT, "experiments", "toy")
    os.makedirs(outdir, exist_ok=True)

    print("== ResBO toy (MockSimulator) ==")
    print(f"presupuesto: metaheuristica={proto['budget']['metaheuristics']}, "
          f"bo={proto['budget']['bo']}, n_seeds={proto.n_seeds}")

    # base de datos compartida por los brazos
    db = SimDatabase(outdir)
    arms: dict[str, list[float]] = {"mfo": [], "gwo": [], "pso": [], "foa": [],
                                    "bo_prior": [], "bo": []}
    histories: dict[str, list] = {k: [] for k in arms}

    t0 = time.time()
    for seed in range(proto.n_seeds):
        # 1) baselines (PSO/FOA con las formulas exactas del repo BIOPSO)
        for algo, Klass in (("mfo", MFO), ("gwo", GWO), ("pso", PSO),
                            ("foa", FOA)):
            sim = MockSimulator(proto, seed=42)      # misma "verdad" fisica
            best = Klass(proto, sim, db, seed=seed).run()
            arms[algo].append(best)
        print(f"  seed {seed}: MFO={arms['mfo'][-1]:,.0f}  "
              f"GWO={arms['gwo'][-1]:,.0f}  PSO={arms['pso'][-1]:,.0f}  "
              f"FOA={arms['foa'][-1]:,.0f}")

    # 2) CRM con las curvas de los baselines (PSO/FOA aportan excitacion)
    sim = MockSimulator(proto, seed=42)
    base_algos = ["mfo", "gwo", "pso", "foa"]
    Xc, curves = db.curves(base_algos)
    crm = CRMModel(proto.n_inj, proto.n_prod).fit(Xc, curves)
    corr = _corr_tau(crm.tau, sim.tau_true)
    print(f"  CRM ajustado con {crm.n_runs_used} runs; R2 medio="
          f"{crm.fit_r2.mean():.3f}; corr(tau_fit,tau_true)={corr:.3f}")

    # 3) brazos BO (reutilizan la base de los baselines: MFO+GWO+PSO+FOA)
    for tag, use_prior in (("bo_prior", True), ("bo", False)):
        for seed in range(proto.n_seeds):
            bo = BayesOptimizer(proto, MockSimulator(proto, seed=42), db,
                                prior_model=crm if use_prior else None)
            bo.run(algo_tag=tag, seed=seed, base_algos=base_algos)
            recs = [r for r in db.records if r.algo == tag and r.seed == seed]
            npvs = [r.npv for r in recs]
            arms[tag].append(max(npvs))
            histories[tag].append((seed, bo.history))
        print("  " + tag + ": " + "  ".join(f"{v:,.0f}" for v in arms[tag]))

    db.save()
    dt = time.time() - t0

    # 4) comparacion
    bp = np.array(arms["bo_prior"])
    b0 = np.array(arms["bo"])
    U, p = mannwhitney(bp, b0) if len(bp) > 1 else (float("nan"), float("nan"))
    rb = rank_biserial(bp, b0) if len(bp) > 1 else float("nan")
    print("\n== Resultados (NPV) ==")
    for k in arms:
        arr = np.array(arms[k])
        print(f"  {k:9s} media={arr.mean():>14,.0f}  max={arr.max():>14,.0f}  "
              f"min={arr.min():>14,.0f}")
    print(f"  Mann-Whitney U(bo_prior vs bo)={U:.2f}  p={p:.3f}  r_rb={rb:+.2f}")
    print(f"  (tiempo toy {dt:.1f}s; N pequeno -> solo demo funcional)")

    _write_summary(outdir, arms)
    _plot_convergence(outdir, db)
    print(f"\nBase guardada en {db.sim_file}  ({len(db.records)} sims)")


def _plot_convergence(outdir, db):
    """Figura: best-so-far de cada brazo (media de seeds) vs nº de sims."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib no disponible; figura omitida)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"mfo": "tab:gray", "gwo": "tab:olive", "pso": "tab:green",
              "foa": "tab:brown", "bo": "tab:blue", "bo_prior": "tab:red"}
    for arm in ("mfo", "gwo", "pso", "foa", "bo", "bo_prior"):
        curves = {}
        for r in db.records:
            if r.algo != arm:
                continue
            k = r.seed
            curves.setdefault(k, []).append(r.npv)
        if not curves:
            continue
        maxn = max(len(v) for v in curves.values())
        mean = np.zeros(maxn)
        for v in curves.values():
            bsf = np.maximum.accumulate(v)
            bsf = np.concatenate([bsf, np.full(maxn - len(bsf), bsf[-1])])
            mean += bsf
        mean /= len(curves)
        ax.plot(np.arange(1, maxn + 1), mean / 1e6, lw=2,
                color=colors.get(arm), label=arm)
    ax.set_xlabel("simulaciones")
    ax.set_ylabel("NPV [millones USD]")
    ax.set_title("ResBO toy — best-so-far por brazo (MockSimulator)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(outdir, "convergence.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  figura: {out}")


def _corr_tau(tau_fit, tau_true):
    a, b = tau_fit.ravel(), tau_true.ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _write_summary(outdir, arms):
    import csv
    with open(os.path.join(outdir, "arms_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n_seeds", "mean_npv", "max_npv", "min_npv"])
        for k, v in arms.items():
            a = np.array(v)
            w.writerow([k, len(v), f"{a.mean():.1f}", f"{a.max():.1f}", f"{a.min():.1f}"])


if __name__ == "__main__":
    main()
