"""PSO fiel a la version del repo BIOPSO (storms_unb/pso/pso_wellcontrol.py).

Ese es el codigo que genera los datos de entrenamiento en MC2, asi que la
version local replica SUS formulas (no la canonica EvoloPy):
  - w constante (psoInertia=0.7), c1=c2=1.5 (psoCognitive/psoSocial)
  - particula 0 anclada al control base del caso (aqui: centro de la caja)
  - velocidad inicial +-5% del ancho de caja por coordenada
  - SIN clip de velocidad (Vmax); solo clip de posicion a la caja
  - fitness = NPV medido (maximizar directo; sin truco costo=-NPV)
  - pbest/gbest por comparacion de NPV; gbest se deriva de pbest
  - evaluacion dentro del lazo: pop*iters simulaciones (1 particula = 1 sim)

Referencia (repo privado petroleoinfralab-ai/BIOPSO, rama main):
  src/ChronosUNB/storms_unb/pso/pso_wellcontrol.py
"""
from __future__ import annotations

import numpy as np


class PSO:
    def __init__(self, protocol, simulator, db, seed: int = 0):
        self.lb, self.ub = protocol.bounds
        b = protocol["budget"]["metaheuristics"]
        self.n = b["pop"]
        self.iters = b["iters"]
        self.sim = simulator
        self.db = db
        self.seed = seed
        cfg = protocol.d.get("pso", {})
        # defaults = defaults del repo (psoInertia/psoCognitive/psoSocial)
        self.w = float(cfg.get("inertia", 0.7))
        self.c1 = float(cfg.get("cognitive", 1.5))
        self.c2 = float(cfg.get("social", 1.5))

    def run(self) -> float:
        rng = np.random.default_rng(self.seed)
        d = len(self.lb)
        span = self.ub - self.lb

        pos = self.lb + rng.random((self.n, d)) * span
        pos[0, :] = 0.5 * (self.lb + self.ub)      # ancla = "control base"
        vel = (rng.random((self.n, d)) - 0.5) * span * 0.1

        pbest = pos.copy()
        pval = np.full(self.n, -np.inf)
        gbest, gval = pos[0].copy(), -np.inf

        for it in range(self.iters):
            # una particula = una simulacion (evaluacion dentro del lazo)
            val = np.empty(self.n)
            for i in range(self.n):
                val[i] = self._eval(pos[i], it)

            improved = val > pval
            pval = np.where(improved, val, pval)
            pbest[improved, :] = pos[improved, :]
            j = int(np.argmax(pval))
            if pval[j] > gval:
                gval, gbest = float(pval[j]), pbest[j].copy()

            r1 = rng.random((self.n, d))
            r2 = rng.random((self.n, d))
            vel = (self.w * vel
                   + self.c1 * r1 * (pbest - pos)
                   + self.c2 * r2 * (gbest - pos))
            pos = np.clip(pos + vel, self.lb, self.ub)   # solo clip de posicion
        return float(gval)

    def _eval(self, x, it) -> float:
        out = self.sim.evaluate(x)
        self.db.add("pso", self.seed, it, x, out["npv"], out.get("prod_curve"))
        return out["npv"]
