"""Laco de optimizacion bayesiana batch (q candidatos/iter) con prior CRM.

Alto nivel:
    bo = BayesOptimizer(protocol, simulator, db, prior_model=None|CRMModel)
    bo.run(algo_tag="bo_prior", seed=0)   # reutiliza datos existentes de MFO/GWO
El prior entra SOLO por la adquisicao (w(n) decreciente); nunca sustituye
evaluaciones: toda evaluacion de fitness es una simulacion real.
"""
from __future__ import annotations

import numpy as np

from ..database import SimDatabase
from ..models.bnn import BnnEnsemble
from .acquisition import acquisition, prior_weight


class BayesOptimizer:
    def __init__(self, protocol, simulator, db: SimDatabase,
                 prior_model=None, n_models=5, batch=None, iters=None,
                 pool=4000, n_init=None, w0=None, tau_decay=None, epochs=300):
        self.p = protocol
        self.sim = simulator
        self.db = db
        self.prior = prior_model
        self.lb, self.ub = protocol.bounds
        self.batch = batch or protocol["budget"]["bo"]["batch"]
        self.iters = iters or protocol["budget"]["bo"]["iters"]
        self.pool = pool
        self.n_init = (n_init if n_init is not None
                       else int(protocol["budget"]["bo"].get("n_init", 5)))
        pr = protocol["prior"]
        self.w0 = pr["w0"] if w0 is None else w0
        self.tau_decay = pr["decay_obs"] if tau_decay is None else tau_decay
        self.n_models = n_models
        self.epochs = epochs
        self.rng = np.random.default_rng(0)
        self.history: list[tuple[int, float]] = []   # (n_sims, best_npv)

    # ------------------------------------------------------------------
    def _lhs(self, n: int) -> np.ndarray:
        d = len(self.lb)
        X = np.empty((n, d))
        for j in range(d):
            perm = self.rng.permutation(n)
            X[:, j] = (perm + self.rng.random(n)) / n
        return self.lb + X * (self.ub - self.lb)

    def _train_surrogate(self, base_algos: list[str], algo_tag: str, seed: int):
        """Pool de baselines + SOLO las sims propias de ESTE seed.

        Antes se filtraba por `base_algos + [algo_tag]`, que sobre una db
        compartida metia las sims BO de los seeds anteriores del mismo brazo:
        el seed k veia k*90 sims extra y los N seeds dejaban de ser
        independientes (supuesto de Mann-Whitney). En BIOPSO el pool es el
        data_v1 fijo, asi que esto alinea el toy con el brazo real.
        """
        recs = [r for r in self.db.records
                if r.algo in base_algos or (r.algo == algo_tag and r.seed == seed)]
        X = np.vstack([r.controls for r in recs]) if recs else np.empty((0, 0))
        y = np.array([r.npv for r in recs])
        if len(X) < 6:
            return None
        model = BnnEnsemble(n_models=self.n_models, seed=int(self.rng.integers(1e6)))
        model.fit_norm_stats(X)
        model.fit(X, y, epochs=self.epochs)
        return model

    def _batch_select(self, cand: np.ndarray, scores: np.ndarray,
                      q: int, X_obs: np.ndarray) -> np.ndarray:
        """top-q con diversidad (distancia minima normalizada entre elegidos)."""
        order = np.argsort(-scores)
        sel: list[np.ndarray] = []
        rng_span = self.ub - self.lb
        for i in order:
            c = cand[i]
            if sel:
                S = np.array(sel)
                dist = np.linalg.norm((S - c) / rng_span, axis=1).min()
                if dist < 0.04:
                    continue
            sel.append(c)
            if len(sel) >= q:
                break
        while len(sel) < q:                            # relleno si hay huecos
            sel.append(cand[order[np.random.randint(len(order))]])
        return np.array(sel)

    # ------------------------------------------------------------------
    def run(self, algo_tag: str, seed: int,
            base_algos: list[str] = ("mfo", "gwo", "pso", "foa")):
        self.rng = np.random.default_rng(seed)
        # 1) estado inicial: datos existentes (baselines) + n_init DoE propios
        X_obs, y_obs, _ = self.db.controls_npv(list(base_algos))
        init = self._lhs(self.n_init)
        for x in init:
            out = self.sim.evaluate(x)
            self.db.add(algo_tag, seed, -1, x, out["npv"], out.get("prod_curve"))
            y_obs = np.append(y_obs, out["npv"])
            X_obs = np.vstack([X_obs, x]) if X_obs.size else x[None, :]
        n_sims = self.n_init

        for it in range(self.iters):
            model = self._train_surrogate(list(base_algos), algo_tag, seed)
            if model is None:
                raise RuntimeError("surrogate sin datos")
            cand = self._lhs(self.pool)
            mu, sigma = model.predict(cand)
            f_best = float(y_obs.max())
            # El decaimiento cuenta las sims PROPIAS del laço (n_sims), no las
            # observaciones pooling: los baselines son conocimiento "gratis" de
            # otros algoritmos; el prior debe estar vivo mientras el BO mismo
            # esta en regimen data-scarce (lectura de DynMeanBO, ICLR 2026).
            w = (prior_weight(n_sims, self.w0, self.tau_decay)
                 if self.prior is not None else 0.0)
            prior_s = self.prior.score_many(cand) if self.prior is not None else None
            s = acquisition(mu, sigma, f_best, prior_s, w)
            batch = self._batch_select(cand, s, self.batch, X_obs)
            for x in batch:
                out = self.sim.evaluate(x)
                self.db.add(algo_tag, seed, it, x, out["npv"], out.get("prod_curve"))
                y_obs = np.append(y_obs, out["npv"])
                X_obs = np.vstack([X_obs, x])
                n_sims += 1
                self.history.append((n_sims, float(y_obs.max())))
        return float(y_obs.max())
