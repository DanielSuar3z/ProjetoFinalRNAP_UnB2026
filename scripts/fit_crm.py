"""Ajuste del prior CRM sobre datos de corrida (T-R4).

Lee runs BIOPSO (simlog_spec_v1, ruta A) o una SimDatabase local, apila los
historiales de produccion late-time con sus vectores de control, ajusta la
matriz de conectividades tau (8x4) por productor (lstsq + clip, ver
prior/crm.py) y deja el prior persistido en el contracto crm_v1.npz:

    experiments/models/<tag>.npz            tau + well_mask + meta (sin pickle)
    experiments/models/<tag>_metrics.json   R2 por pozo, excitacion, gate

Gate del cronograma (S2): R2 por pozo >= --r2-min (0.7 default). Pozos que
fallan quedan FUERA del prior (columna de tau en cero, well_mask=0): es mejor
un prior que cubre 3 pozos que uno que miente sobre 4. Si pasan < 2 pozos,
exit 2: el brazo BO+prior no se lanza.

Nota de consistencia (mock vs real): en el MockSimulator `prod_curve` es la
tasa de LIQUIDO; en el sim_curves.npz real la curva es `wpr` (tasa de AGUA).
El CRM modela la produccion total; del lado real la salvedad se documenta en
meta_json (curva_source="wpr") y se decide con datos delante (ver CHANGELOG).

Uso:
    python scripts/fit_crm.py --harvest experiments/coleta_s1 --tag crm_v1
    python scripts/fit_crm.py --db experiments/toy --tag crm_toy
    python scripts/fit_crm.py --self-check        # fixture synth, sin datos reales
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from resbo.config import Protocol                        # noqa: E402
from resbo.database import SimDatabase                   # noqa: E402
from resbo.prior.crm import CRMModel, save_crm_npz       # noqa: E402


# ---------------------------------------------------------------------- datos
def db_from_harvest(harvest_root: str, out_db_dir: str) -> SimDatabase:
    """Carga el arbol BIOPSO y deja SOLO la ruta A (simlog) para el CRM.

    La ruta B (rescate) aporta 1 fila/corrida con curvas de longitud ajena al
    horizon del protocolo: util para entrenar el BNN, inutil para identificar
    tau (necesitamos N vectores de control DISTINTOS con sus historiales).
    """
    from resbo.loader_biopso import load_harvest
    db = SimDatabase(out_db_dir)
    n = load_harvest(harvest_root, db)
    kept = [r for r in db.records if r.iteration >= 0 and r.prod_curve is not None]
    db.records, db._next_id = kept, len(kept)
    db.save()
    print(f"harvest: {n} sims cargadas | {len(kept)} de ruta A (con simlog) "
          f"disponibles para CRM")
    return db


def excitation_report(X: np.ndarray, min_exc: float) -> tuple[np.ndarray, list]:
    """Fractions de registros con rate_i >= min_exc por inyector + avisos."""
    frac = np.array([(X[:, i] >= min_exc).mean() for i in range(X.shape[1] - 4)])
    avisos = [f"inyector {i}: solo {frac[i]:.0%} de sims con rate >= {min_exc:.0f} "
              f"-> columna de tau poco excitada"
              for i in range(len(frac)) if frac[i] < 0.25]
    return frac, avisos


# ------------------------------------------------------------------- ajuste
def fit_and_gate(db: SimDatabase, protocol: Protocol, r2_min: float,
                 late_frac: float):
    """Ajusta CRM y aplica el gate por pozo. Devuelve (crm, mask, stats).

    Empareja control<->curva desde los records (mismo filtro para ambos):
    filtrar por separado desalinearia X con curves en silencio.
    """
    recs = [r for r in db.records if r.prod_curve is not None
            and len(r.prod_curve) == protocol.n_prod * protocol.horizon]
    if len(recs) < 3 * protocol.n_prod:
        sys.exit(f"error: {len(recs)} sims con curvas completas; el CRM necesita "
                 f">= {3 * protocol.n_prod} controles distintos excitados")
    X = np.vstack([r.controls for r in recs])
    curves = [r.prod_curve.reshape(protocol.n_prod, protocol.horizon) for r in recs]
    min_exc = float(protocol["crm"]["min_excitation"])
    frac, avisos = excitation_report(X, min_exc)

    crm = CRMModel(protocol.n_inj, protocol.n_prod, min_excitation=min_exc)
    crm.fit(X, curves, late_frac=late_frac)
    mask = (crm.fit_r2 >= r2_min).astype("int8")

    stats = {"n_sims_fit": int(X.shape[0]),
             "horizon": protocol.horizon, "late_frac": late_frac,
             "min_excitation": min_exc,
             "excitation_frac_por_inyector": [round(float(f), 3) for f in frac],
             "avisos_excitacion": avisos,
             "r2_por_pozo": [round(float(r), 4) for r in crm.fit_r2],
             "pozos_ok": int(mask.sum()), "r2_min": r2_min}
    y = np.array([r.npv for r in recs])
    return crm, mask, stats, (X, y)


def apply_mask_and_calibrate(crm: CRMModel, mask: np.ndarray, X: np.ndarray,
                             y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Pozos fuera del gate quedan FUERA del prior (tau, beta y q0 en cero) y
    LUEGO se calibran los pesos del score sobre la tau final: calibrar antes
    seria ajustar pesos a features que el brazo BO nunca vera."""
    tau_full = crm.tau.copy()
    crm.tau[:, mask == 0] = 0.0
    crm.beta_bhp[mask == 0] = 0.0
    crm.q0[mask == 0] = 0.0
    calib = crm.calibrate_score(X, y)
    if calib["spearman_holdout_calib"] <= 0.0:
        print("  AVISO: score calibrado sin poder predictivo en holdout "
              f"(Spearman {calib['spearman_holdout_calib']:+.3f}) — el prior no "
              "informa; la ablacion BO-sin-prior lo va a mostrar")
    return tau_full, calib


# ---------------------------------------------------------------- self-check
def self_check(protocol_path: str, n_runs: int, n_sims: int) -> None:
    """Fixture sintético end-to-end: genera runs spec-compliant, carga por el
    loader REAL, ajusta, gatea, persiste y recarga. Sin MC2, sin datos reales."""
    from resbo.simulator import MockSimulator
    from resbo.loader_biopso import make_synthetic_run, load_run
    from resbo.prior.crm import load_crm_npz

    proto = Protocol.load(protocol_path)
    tmp = tempfile.mkdtemp(prefix="crm_selfcheck_")
    print(f"[self-check] fixture en {tmp}")

    # UN solo mock (un solo "yacimiento", una sola tau) con N runs de DoE
    # uniforme: replica la estructura real de la coleta S1.
    sim = MockSimulator(proto, seed=42)
    db = SimDatabase(os.path.join(tmp, "db"))
    for k in range(n_runs):
        run_dir = make_synthetic_run(os.path.join(tmp, "random", f"run_{k+1:02d}"),
                                     sim, proto, seed=k, n_sims=n_sims,
                                     agents_per_round=5)
        load_run(run_dir, "random", 0, db)
    db.save()

    crm, mask, stats, (Xc, yc) = fit_and_gate(db, proto, r2_min=0.7, late_frac=0.4)
    print(f"[self-check] sims={stats['n_sims_fit']} r2_por_pozo={stats['r2_por_pozo']} "
          f"pozos_ok={stats['pozos_ok']}/{proto.n_prod}")
    assert stats["pozos_ok"] >= 2, "gate por pozo fallido en fixture"
    assert stats["r2_por_pozo"][0] > 0.0

    r = float(np.corrcoef(crm.tau.ravel(), sim.tau_true.ravel())[0, 1])
    print(f"[self-check] corr(tau_fit, tau_true) = {r:.3f}")
    assert r > 0.85, f"fixture no recupera estructura (r={r:.2f})"

    _, calib = apply_mask_and_calibrate(crm, mask, Xc, yc)
    print(f"[self-check] score calib_v1: Spearman holdout "
          f"{calib['spearman_holdout_calib']:+.3f} (legacy "
          f"{calib['spearman_holdout_legacy']:+.3f})")
    assert calib["spearman_holdout_calib"] > calib["spearman_holdout_legacy"], \
        "el score calibrado deberia ordenar el NPV mejor que el legacy"

    path = save_crm_npz(os.path.join(tmp, "crm_selfcheck.npz"), crm, mask,
                        meta={"fuente": "self-check", "curva_source": "mock_liquido"})
    crm2, mask2, meta2 = load_crm_npz(path)
    assert np.allclose(crm.tau, crm2.tau) and np.array_equal(mask, mask2)
    assert crm2.calibrated and np.allclose(crm.score_many(Xc), crm2.score_many(Xc))
    assert meta2["fuente"] == "self-check"
    print("[self-check] crm_v1.npz save/load roundtrip: OK")
    print("SELF-CHECK OK — el pipeline T-R4 esta listo para recibir coleta_s1")


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--harvest", help="arbol de runs BIOPSO (coleta_s1 / harvest)")
    g.add_argument("--db", help="carpeta SimDatabase local (simulations.csv)")
    g.add_argument("--self-check", action="store_true",
                   help="fixture sintetico end-to-end (sin datos reales)")
    ap.add_argument("--protocol", default=os.path.join(ROOT, "configs", "protocol.yaml"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "experiments", "models"))
    ap.add_argument("--tag", default="crm_v1")
    ap.add_argument("--r2-min", type=float, default=0.7,
                    help="R2 late-time minimo por pozo (gate S2)")
    ap.add_argument("--late-frac", type=float, default=0.4,
                    help="fraccion final del historial usada para el steady")
    ap.add_argument("--db-out", default=None,
                    help="donde guardar la BD derivada del harvest")
    args = ap.parse_args()

    if args.self_check:
        self_check(args.protocol, n_runs=5, n_sims=40)
        return

    proto = Protocol.load(args.protocol)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.harvest:
        db = db_from_harvest(args.harvest,
                             args.db_out or os.path.join(args.out_dir, f"{args.tag}_db"))
        curva_source = "wpr"          # sim_curves.npz real = tasa de AGUA
    else:
        db = SimDatabase.load(args.db)
        curva_source = "db_prod_curve"

    crm, mask, stats, (Xc, yc) = fit_and_gate(db, proto, args.r2_min, args.late_frac)

    print(f"ajuste CRM con {stats['n_sims_fit']} sims "
          f"(late_frac={args.late_frac}, horizonte {stats['horizon']} m)")
    for a in stats["avisos_excitacion"]:
        print(f"  aviso: {a}")
    for j in range(proto.n_prod):
        print(f"  pozo {j}: R2={stats['r2_por_pozo'][j]:+.3f}  "
              f"{'OK' if mask[j] else 'FUERA (tau columna en cero)'}")

    # decision de gate ANTES de cero-purgar (auditoria en tau_full); despues
    # se calibra el score sobre la tau final
    tau_full, calib = apply_mask_and_calibrate(crm, mask, Xc, yc)
    gate_ok = int(mask.sum()) >= 2
    print(f"  score calib_v1: Spearman holdout {calib['spearman_holdout_calib']:+.3f} "
          f"(legacy {calib['spearman_holdout_legacy']:+.3f})")

    meta = {"fuente": args.harvest or args.db, "curva_source": curva_source,
            "protocol_reservoir": proto.d.get("reservoir"), "tau_full": tau_full,
            **stats, "score": calib}
    npz = save_crm_npz(os.path.join(args.out_dir, f"{args.tag}.npz"), crm, mask, meta)
    out_json = {k: v for k, v in meta.items() if k != "tau_full"}
    out_json.update({"tag": args.tag, "fecha": time.strftime("%Y-%m-%d %H:%M"),
                     "gate_pozos_min": 2, "gate_pass": bool(gate_ok)})
    jp = os.path.join(args.out_dir, f"{args.tag}_metrics.json")
    with open(jp, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"guardado: {npz}\n         {jp}")
    if not gate_ok:
        print("GATE: < 2 pozos confiables — NO lanzar el brazo BO+prior "
              "(usar solo BO-sin-prior o revisar datos)")
        sys.exit(2)


if __name__ == "__main__":
    main()
