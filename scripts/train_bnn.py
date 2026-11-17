"""Entrenamiento standalone del surrogate BNN (T-R3).

Carga una SimDatabase (o un harvest BIOPSO via el loader de la spec v1),
hace split train/val fijo, entrena el BnnEnsemble y deja:
  <out-dir>/<tag>.npz           pesos (contrato que consume resbo_wellcontrol/T-R5)
  <out-dir>/<tag>_metrics.json  NRMSE train/val, R2 val, gate de calidad
  <out-dir>/<tag>_scatter.png   pred vs real en val (si hay matplotlib)

Gate del cronograma (S2): NRMSE_val < 5% del rango de NPV del dataset.

Uso:
    python scripts/train_bnn.py --db experiments/toy [--tag bnn_toy]
    python scripts/train_bnn.py --harvest experiments/coleta_s1 --tag bnn_v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from resbo.database import SimDatabase            # noqa: E402
from resbo.models.bnn import BnnEnsemble          # noqa: E402


def _metrics(mu, y, y_range):
    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
    ss_res = float(np.sum((y - mu) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return {"rmse": rmse, "nrmse": rmse / max(y_range, 1e-12),
            "r2": 1.0 - ss_res / ss_tot}


def _scatter_png(path, y_true, y_pred):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("aviso: sin matplotlib, no pinto el scatter")
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(y_true / 1e6, y_pred / 1e6, ".", alpha=0.6, ms=5)
    lo = min(y_true.min(), y_pred.min()) / 1e6
    hi = max(y_true.max(), y_pred.max()) / 1e6
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("NPV real (M USD)")
    ax.set_ylabel("NPV predicho (M USD)")
    ax.set_title("BNN validation: pred vs real")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--db", help="carpeta SimDatabase (simulations.csv)")
    g.add_argument("--harvest", help="arbol de runs BIOPSO (simlog_spec_v1)")
    ap.add_argument("--algos", default=None,
                    help="lista csv de tags a entrenar (default: todos)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "experiments", "models"))
    ap.add_argument("--tag", default="bnn_v1")
    ap.add_argument("--db-out", default=None,
                    help="donde guardar la BD derivada del harvest")
    ap.add_argument("--n-models", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=12345)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--gate", type=float, default=0.05,
                    help="NRMSE_val maxima (fraccion del rango de NPV)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    algos = args.algos.split(",") if args.algos else None

    # ---- datos ---------------------------------------------------------
    if args.harvest:
        from resbo.loader_biopso import load_harvest
        db_dir = args.db_out or os.path.join(args.out_dir, f"{args.tag}_db")
        db = SimDatabase(db_dir)
        n = load_harvest(args.harvest, db)
        print(f"harvest: {n} simulaciones cargadas en {db_dir}")
    else:
        db = SimDatabase.load(args.db)
    X, y, recs = db.controls_npv(algos)
    if X.shape[0] < 30:
        sys.exit(f"error: {X.shape[0]} filas (<30). Carga la BD/harvest primero.")
    from collections import Counter
    por_algo = Counter(r.algo for r in recs)
    print(f"dataset: {X.shape[0]} sims {dict(por_algo)} | dims X={X.shape[1]}")

    # ---- split 80/20 con semilla fija ----------------------------------
    rng = np.random.default_rng(args.split_seed)
    idx = rng.permutation(X.shape[0])
    n_val = max(int(round(args.val_frac * X.shape[0])), 5)
    vi, ti = idx[:n_val], idx[n_val:]
    Xtr, ytr, Xva, yva = X[ti], y[ti], X[vi], y[vi]
    y_range = float(y.max() - y.min())
    print(f"split: train={len(ti)} val={len(vi)} (semilla {args.split_seed}) "
          f"| rango NPV = {y_range/1e6:.1f} M USD")

    # ---- entreno --------------------------------------------------------
    t0 = time.time()
    model = BnnEnsemble(n_models=args.n_models, hidden=(args.hidden, args.hidden),
                        seed=args.seed)
    model.fit_norm_stats(Xtr)
    model.fit(Xtr, ytr, epochs=args.epochs, lr=args.lr)
    print(f"entrenado en {time.time()-t0:.1f}s")

    mu_tr, _ = model.predict(Xtr, mc_samples=8)
    mu_va, sd_va = model.predict(Xva, mc_samples=8)
    m_tr, m_va = _metrics(mu_tr, ytr, y_range), _metrics(mu_va, yva, y_range)
    # cobertura: fraccion de val cuyo ±2sigma cubre el real (sanidad de sigma)
    cov = float(np.mean(np.abs(mu_va - yva) <= 2 * sd_va))
    ok = m_va["nrmse"] < args.gate

    print(f"train  NRMSE={m_tr['nrmse']:.4f}  R2={m_tr['r2']:.3f}")
    print(f"val    NRMSE={m_va['nrmse']:.4f}  R2={m_va['r2']:.3f}  "
          f"cobertura2s={cov:.2f}")
    print(f"GATE NRMSE_val < {args.gate}: "
          f"{'PASS' if ok else 'FALLO (no lanzar el brazo BO con este modelo)'}")

    # ---- artefactos -----------------------------------------------------
    npz = model.save_npz(os.path.join(args.out_dir, f"{args.tag}.npz"))
    png = _scatter_png(os.path.join(args.out_dir, f"{args.tag}_scatter.png"),
                       yva, mu_va)
    out = {
        "tag": args.tag, "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "fuente": args.harvest or args.db, "algos": algos or dict(por_algo),
        "n_train": int(len(ti)), "n_val": int(len(vi)),
        "nrmse_rango_def": "rmse / (max(y)-min(y)) del dataset completo",
        "train": m_tr, "val": m_va, "cobertura_2sigma_val": cov,
        "gate_nrmse": args.gate, "gate_pass": bool(ok),
        "hiper": {"n_models": args.n_models, "hidden": args.hidden,
                  "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
                  "split_seed": args.split_seed},
    }
    with open(os.path.join(args.out_dir, f"{args.tag}_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"guardado: {npz}\n         "
          f"{os.path.join(args.out_dir, args.tag + '_metrics.json')}"
          + (f"\n         {png}" if png else ""))
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
