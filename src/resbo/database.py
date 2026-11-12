"""Base de datos del proyecto: cada simulacion = un par (controles, VPL) + curvas.

Formato en disco (versionable, sin binarios):
  <dir>/simulations.csv  -- una fila por simulacion:
     sim_id, algo, seed, iteration, controls (CSV de 12 floats), npv_usd,
     prod_curve (CSV de n_prod*horizon floats, curvas de produccion por pozo)
  <dir>/runs.csv         -- resumen por corrida (algo, seed, best_npv, n_sims)
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimRecord:
    sim_id: int
    algo: str
    seed: int
    iteration: int
    controls: np.ndarray          # (12,)
    npv: float
    prod_curve: np.ndarray = None # (n_prod*horizon,) o None


class SimDatabase:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.sim_file = os.path.join(path, "simulations.csv")
        self.records: list[SimRecord] = []
        self._next_id = 0

    # ---- escrituras -------------------------------------------------
    def add(self, algo: str, seed: int, iteration: int,
            controls: np.ndarray, npv: float,
            prod_curve: np.ndarray | None = None) -> SimRecord:
        rec = SimRecord(self._next_id, algo, seed, iteration,
                        np.asarray(controls, float), float(npv),
                        None if prod_curve is None else np.asarray(prod_curve, float).ravel())
        self.records.append(rec)
        self._next_id += 1
        return rec

    def save(self):
        with open(self.sim_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sim_id", "algo", "seed", "iteration",
                        "controls", "npv_usd", "prod_curve"])
            for r in self.records:
                w.writerow([r.sim_id, r.algo, r.seed, r.iteration,
                            ";".join(map(repr, r.controls.tolist())),
                            repr(r.npv),
                            ";" if r.prod_curve is None else
                            ";".join(map(repr, r.prod_curve.tolist()))])

    # ---- lecturas ---------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "SimDatabase":
        db = cls(path)
        if not os.path.exists(db.sim_file):
            return db
        with open(db.sim_file) as f:
            rd = csv.reader(f)
            header = next(rd)
            for row in rd:
                rec = SimRecord(int(row[0]), row[1], int(row[2]), int(row[3]),
                                np.array([float(v) for v in row[4].split(";") if v != ""]),
                                float(row[5]),
                                None if row[6] == ";" else
                                np.array([float(v) for v in row[6].split(";") if v != ""]))
                db.records.append(rec)
                db._next_id = max(db._next_id, rec.sim_id + 1)
        return db

    # ---- vistas para ML ---------------------------------------------
    def controls_npv(self, algos: list[str] | None = None):
        """(X, y, meta) filtrando por algoritmo; y = NPV."""
        recs = [r for r in self.records if algos is None or r.algo in algos]
        X = np.vstack([r.controls for r in recs]) if recs else np.empty((0, 0))
        y = np.array([r.npv for r in recs])
        return X, y, recs

    def curves(self, algos: list[str] | None = None, horizon: int = 130):
        """(X, curves) para CRM: curves[i] = (n_prod, horizon) del run i."""
        recs = [r for r in self.records
                if (algos is None or r.algo in algos) and r.prod_curve is not None]
        if not recs:
            return np.empty((0, 0)), []
        n_prod = int(len(recs[0].prod_curve) / horizon)
        X = np.vstack([r.controls for r in recs])
        curves = [r.prod_curve.reshape(n_prod, horizon) for r in recs]
        return X, curves
