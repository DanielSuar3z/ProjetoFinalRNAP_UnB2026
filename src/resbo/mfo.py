"""MFO mono-objetivo (Mirjalili 2015) adaptado a control de pozos.

Patron BIOPSO: 1 polilla = 1 simulacion; vector de control constante por vida
del campo; fitness = NPV (maximizar); espiral hacia una llama elegida entre las
mejores; clip de caja.
"""
from __future__ import annotations

import numpy as np


class MFO:
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
        best_fit = float(F.max())

        for it in range(self.iters):
            order = np.argsort(F)                 # llama 0 = peor, -1 = mejor
            flames = X[order]
            a = -1 - it / max(self.iters - 1, 1)  # decae -1 -> -2
            Xn = np.empty_like(X)
            for i in range(self.n):
                # llama objetivo: sesgo hacia las mejores (extremo superior)
                j = int(np.floor((self.n - 1) * (rng.random() ** 2)))
                flame = flames[min(j, self.n - 1)]
                D = np.abs(flame - X[i])
                t = a * rng.random()
                Xn[i] = D * np.exp(t) * np.cos(2 * np.pi * t) + flame
            Xn = np.clip(Xn, self.lb, self.ub)
            F = np.array([self._eval(x, it + 1) for x in Xn])
            X = Xn
            best_fit = max(best_fit, float(F.max()))
        return best_fit

    def _eval(self, x, it) -> float:
        out = self.sim.evaluate(x)
        self.db.add("mfo", self.seed, it, x, out["npv"], out.get("prod_curve"))
        return out["npv"]
