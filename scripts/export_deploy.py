"""Empaqueta el deploy del brazo BO para BIOPSO (T-R5).

Escribe en --out-dir los tres artefactos que consumen los configs
egg_resbo_*.yaml de BIOPSO (deploy/resbo_v1/ en la raiz del repo):

  data_v1.npz   schema 'resbo_data_v1': X (controles), y (NPV), meta_json.
                Experiencia pooled de la coleta S1 (ruta A + rescate via el
                loader simlog_spec_v1, que ya rinde 1 fila/runa de rescate por
                el sidecar best_controls.json). Es la premisa del proyecto:
                el brazo BO entrena con los DATOS DE LOS BASELINES + los
                propios, no simula de mas.
  crm_v1.npz    Copia verificada del prior (schema 'crm_v1', tau ya con
                well_mask aplicado por scripts/fit_crm.py).
  bnn_v1.npz    Copia verificada del surrogate (contrato save_npz de
                models/bnn.py: arquitectura + atestacion del gate; el runtime
                reentrena, ver resbo_wellcontrol.py).

Atestacion (honestidad del pipeline): los npz del toy local NO cumplen el
gate S2 (NRMSE_val < 5% con la economia real, ver bitacora paso 9), por eso:
  * sin --smoke, los metrics <tag>_metrics.json hermanos de --crm/--bnn son
    OBLIGATORIOS y deben decir gate_pass=true; si no, el script aborta.
  * --smoke relaja eso (cableado/CD en MC2 con artefactos pequenos), copia
    igualmente y marca la bandera smoke en deploy_manifest.json.
Ademas escribe deploy_manifest.json con sha256/tamanos para verificar la
transferencia a MC2 (el arbol se mueve con rsync/scp, no con git).

Uso:
    # real (cuando llegue la coleta):
    python scripts/export_deploy.py --harvest experiments/coleta_s1 \
        --crm experiments/models/crm_v1.npz --bnn experiments/models/bnn_v1.npz
    # smoke con artefactos del toy local (NO usar para medir):
    python scripts/export_deploy.py --db experiments/toy \
        --crm experiments/models/crm_toy.npz --bnn experiments/models/bnn_toy_demo.npz \
        --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from resbo.database import SimDatabase                    # noqa: E402
from resbo.models.bnn import BnnEnsemble                  # noqa: E402
from resbo.prior.crm import load_crm_npz                  # noqa: E402


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gate_hermano(npz_path, smoke):
    """Lee <tag>_metrics.json hermano y devuelve (gate_pass|None, nota).

    Con --smoke no exigimos el gate (el toy local NO pasa el gate NRMSE con
    la economia real -- hallazgo honesto de T-R1, bitacora paso 9); sin el,
    metrics ausente o gate_pass=false son razon de salida 2."""
    base, _ = os.path.splitext(npz_path)
    jp = base + "_metrics.json"
    if not os.path.exists(jp):
        return None, "metrics.json hermano ausente"
    with open(jp) as f:
        m = json.load(f)
    return bool(m.get("gate_pass")), jp


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--db", help="carpeta SimDatabase (simulations.csv)")
    g.add_argument("--harvest", help="arbol de runs BIOPSO (simlog_spec_v1)")
    ap.add_argument("--crm", required=True, help="crm_v1.npz de scripts/fit_crm.py")
    ap.add_argument("--bnn", default=None,
                    help="bnn_v1.npz de scripts/train_bnn.py (opcional: sin el, "
                         "el runtime usa la arquitectura por defecto del config)")
    ap.add_argument("--algos", default=None,
                    help="subconjunto de tags para la experiencia (default: todos)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "deploy", "resbo_v1"))
    ap.add_argument("--smoke", action="store_true",
                    help="deploy de cableado con artefactos de toy: no exige "
                         "gate_pass ni 30+ sims (NO usar para medir)")
    args = ap.parse_args()

    # ---- 1) experiencia (controles + NPV de la coleta) --------------------
    if args.harvest:
        from resbo.loader_biopso import load_harvest
        db_dir = os.path.join(args.out_dir, "_db_from_harvest")
        db = SimDatabase(db_dir)
        n = load_harvest(args.harvest, db)
        print(f"harvest: {n} simulaciones -> {db_dir}")
    else:
        db = SimDatabase.load(args.db)
    X, y, recs = db.controls_npv(args.algos.split(",") if args.algos else None)
    from collections import Counter
    por_algo = dict(Counter(r.algo for r in recs))
    minimo = 8 if args.smoke else 30
    if X.shape[0] < minimo:
        sys.exit(f"error: {X.shape[0]} sims de experiencia (<{minimo})")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        sys.exit("error: NaN/inf en la experiencia (corrida corrupta?)")

    # ---- 2) prior CRM (verificacion del contracto antes de copiar) --------
    crm, mask, crm_meta = load_crm_npz(args.crm)   # (CRMModel, mask, meta)
    n_ok = int(np.asarray(mask).sum())
    r2 = np.asarray(crm.fit_r2)
    if X.shape[1] != crm.tau.shape[0] + crm.tau.shape[1]:
        sys.exit(f"error: tau {crm.tau.shape} no corresponde a X de "
                 f"{X.shape[1]} columnas (otro caso?)")

    # ---- 3) atestaciones ----------------------------------------------------
    attest, avisos = {}, []
    gate_crm, nota_crm = _gate_hermano(args.crm, args.smoke)
    attest["crm"] = {"fuente": os.path.basename(args.crm), "gate_pass": gate_crm,
                     "pozos_confiables": n_ok, "r2": [round(float(v), 4) for v in r2],
                     "metrics": nota_crm,
                     "score": "calib_v1" if crm.calibrated else "legacy"}
    if not crm.calibrated:
        avisos.append("crm sin score calib_v1 (npz previo a 2026-11-17): el brazo "
                      "usara el score legacy; re-correr scripts/fit_crm.py")
    if not args.smoke and (gate_crm is False or gate_crm is None):
        # OJO: el gate de CRM es por pozo (well_mask), no binario del global;
        # gate_pass del metrics de fit_crm = >=2 pozos buenos. <2 ya lo corto
        # aca: el brazo con prior esta PROHIBIDO (decision S2 del proyecto).
        sys.exit(f"error: CRM sin atestacion de gate ({gate_crm}); lanzar el "
                 f"brazo con prior sin esto es exactamente lo que el gate S2 "
                 f"prohibe. (--smoke relaja para prueba de cableado)")
    if n_ok < 2:
        sys.exit("error: crm_v1 con <2 pozos confiables: mismo gate del "
                 "runtime, exportar igual no tiene sentido")

    if args.bnn:
        BnnEnsemble.load_npz(args.bnn)          # falla duro si el npz esta roto
        gate_bnn, nota_bnn = _gate_hermano(args.bnn, args.smoke)
        attest["bnn"] = {"fuente": os.path.basename(args.bnn),
                         "gate_pass": gate_bnn, "metrics": nota_bnn}
        if not args.smoke and (gate_bnn is False or gate_bnn is None):
            sys.exit("error: bnn sin atestacion de gate NRMSE (metricos: "
                     f"{nota_bnn}); el proyecto NO lanza el brazo BO con un "
                     "surrogate reprobado. (--smoke relaja para cableado)")
    else:
        attest["bnn"] = None
        avisos.append("sin --bnn: el runtime usara la arquitectura por defecto "
                      "del config (n_models/hidden del YAML)")

    # ---- 4) escritura atomo-ish + manifest ---------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "fuente": args.harvest or args.db,
        "n_sims": int(X.shape[0]), "por_algo": por_algo,
        "dims_x": int(X.shape[1]),
        "smoke": bool(args.smoke),
        "nota": ("experiencia pooled para el brazo BO de BIOPSO "
                 "(resbo_data_v1); NPV en la economia EGG_TEST/NPVParams"),
    }
    data_p = os.path.join(args.out_dir, "data_v1.npz")
    np.savez_compressed(data_p, schema="resbo_data_v1", X=X, y=y,
                        meta_json=json.dumps(meta, ensure_ascii=False))
    crm_p = os.path.join(args.out_dir, "crm_v1.npz")
    shutil.copyfile(args.crm, crm_p)
    bnn_p = None
    if args.bnn:
        bnn_p = os.path.join(args.out_dir, "bnn_v1.npz")
        shutil.copyfile(args.bnn, bnn_p)

    manifiesto = {
        "paquete": "resbo_v1 (deploy BIOPSO)",
        "fecha": meta["fecha"], "smoke": bool(args.smoke),
        "destino_biopso": "deploy/resbo_v1/ (raiz del repo BIOPSO)",
        "archivos": {
            "data_v1.npz": {"sha256": _sha256(data_p),
                            "bytes": os.path.getsize(data_p),
                            "experiencia": meta},
            "crm_v1.npz": {"sha256": _sha256(crm_p),
                           "bytes": os.path.getsize(crm_p), **attest["crm"]},
            "bnn_v1.npz": ({"sha256": _sha256(bnn_p),
                            "bytes": os.path.getsize(bnn_p),
                            **attest["bnn"]} if bnn_p else None),
        },
        "avisos": avisos,
    }
    mp = os.path.join(args.out_dir, "deploy_manifest.json")
    with open(mp, "w") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False)

    print(f"\nexperiencia: {X.shape[0]} sims {por_algo} | X {X.shape[1]}D")
    print(f"prior: {n_ok} pozos confiables, R2 "
          f"{[round(float(v), 3) for v in r2]}")
    print(f"deploy escrito en {args.out_dir}:")
    for k in ("data_v1.npz", "crm_v1.npz", "bnn_v1.npz"):
        p = os.path.join(args.out_dir, k)
        if os.path.exists(p):
            print(f"  {k:14s} {os.path.getsize(p):>10,d} B")
    print(f"  manifest.json  -> {mp}")
    if args.smoke:
        print("SMOKE: artefactos de juguete, solo cableado. NO reportar "
              "numeros de este deploy.")


if __name__ == "__main__":
    main()
