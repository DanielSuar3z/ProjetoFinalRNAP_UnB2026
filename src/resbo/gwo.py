"""GWO (Mirjalili et al. 2014) adaptado a control de pozos (patron BIOPSO).

1 lobo = 1 simulacion. Jerarquia alfa/beta/delta (top-3); caza con A,C;
a decae 2->0; clip de caja al final. Fitness = NPV (maximizar).
"""
from __future__ import annotations

import numpy as np


class GWO:
    def __init__(self, protocol, simulator, db, seed: int = 0):
        self.lb, self.ub = protocol.bounds
        b = protocol["budget"]["metaheuristics"]
        self.n = b["pop"]
        self.iters = b["iters"]
        self.sim = simulator
        self.db = db
        self.seed = seed

    def run(self) -> float:
        rng = np.random.default_rng(self.seed)
        d = len(self.lb)
        X = self.lb + rng.random((self.n, d)) * (self.ub - self.lb)
        F = np.array([self._eval(x, 0) for x in X])

        alpha, beta, delta, fa, fb, fd = self._leaders(X, F)
        best = fa
        for it in range(self.iters):
            a = 2.0 - 2.0 * it / max(self.iters - 1, 1)
            Xn = np.empty_like(X)
            for i in range(self.n):
                Xn[i] = self._hunt(X[i], (alpha, beta, delta), a, rng)
            Xn = np.clip(Xn, self.lb, self.ub)
            F = np.array([self._eval(x, it + 1) for x in Xn])
            X = Xn
            alpha, beta, delta, fa, fb, fd = self._leaders(X, F)
            best = max(best, fa)
        return float(best)

    @staticmethod
    def _hunt(x, leaders, a, rng):
        acc = np.zeros_like(x)
        for leader in leaders:
            A = 2 * a * rng.random(len(x)) - a
            C = 2 * rng.random(len(x))
            D = np.abs(C * leader - x)
            acc += leader - A * D
        return acc / 3.0

    @staticmethod
    def _leaders(X, F):
        order = np.argsort(F)[::-1]
        i0, i1, i2 = order[0], order[1], order[2]
        return X[i0].copy(), X[i1].copy(), X[i2].copy(), F[i0], F[i1], F[i2]

    def _eval(self, x, it) -> float:
        out = self.sim.evaluate(x)
        self.db.add("gwo", self.seed, it, x, out["npv"], out.get("prod_curve"))
        return out["npv"]
