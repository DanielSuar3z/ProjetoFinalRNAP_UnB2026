"""FOA fiel a la version del repo BIOPSO (storms_unb/foa/foa_wellcontrol.py).

Fruit Fly Optimization Algorithm (Pan 2012) segun la adaptacion a well
control del repo — el algoritmo mas simple de la familia BIOPSO:
  - un solo "centro de osmoforos" (ancla = control base del caso; aqui el
    centro de la caja), sin memoria individual entre rondas
  - Paso 2: cada mosca vuela a center +- U(0,1)*foaStep*ancho de caja por
    coordenada (perturbacion uniforme), clip a la caja
  - mosca 0 evaluada EXACTAMENTE en el centro (elitismo sin sim extra: el
    mejor NPV conocido nunca empeora entre rondas)
  - Pasos 3-5: smell = NPV medido directamente (1 mosca = 1 simulacion)
  - Pasos 6-7: el centro SOLO se mueve al mejor de la ronda si supera al
    mejor historico; si no, la manada vuelve a explorar alrededor del mismo
  - sin escape/reinicio (a diferencia de HGSO), sin jerarquia (a diferencia
    de GWO); candidate natural para extension memetica (ver docstring repo)
  - evaluacion dentro del lazo: pop*iters simulaciones

Referencia (repo privado petroleoinfralab-ai/BIOPSO, rama main):
  src/ChronosUNB/storms_unb/foa/foa_wellcontrol.py
"""
from __future__ import annotations

import numpy as np


class FOA:
    def __init__(self, protocol, simulator, db, seed: int = 0):
        self.lb, self.ub = protocol.bounds
        b = protocol["budget"]["metaheuristics"]
        self.n = b["pop"]           # foaFlies
        self.iters = b["iters"]     # foaIterations
        self.sim = simulator
        self.db = db
        self.seed = seed
        cfg = protocol.d.get("foa", {})
        self.step = float(cfg.get("step", 0.1))   # fraccion de caja (repo: 0.1)
        self.base = protocol.base_controls

    def run(self) -> float:
        rng = np.random.default_rng(self.seed)
        d = len(self.lb)
        span = self.ub - self.lb

        # Paso 1: centro de osmoforos = ancla (control base del deck, como en el repo)
        center = self.base.copy()
        gbest, gval = center.copy(), -np.inf

        for it in range(self.iters):
            # Paso 2: vuelo por olfato alrededor del centro
            perturb = (rng.random((self.n, d)) * 2.0 - 1.0) * self.step * span
            fly = center + perturb
            fly[0, :] = center                     # mosca 0 exactamente al centro
            fly = np.clip(fly, self.lb, self.ub)

            val = np.empty(self.n)
            for i in range(self.n):
                val[i] = self._eval(fly[i], it)

            # Pasos 3-5: smell = NPV medido; mejor olor de la ronda
            j = int(np.argmax(val))
            # Pasos 6-7: elitismo puro — el centro solo vuela si hay mejora
            if val[j] > gval:
                gval, gbest = float(val[j]), fly[j].copy()
                center = gbest.copy()
        return float(gval)

    def _eval(self, x, it) -> float:
        out = self.sim.evaluate(x)
        self.db.add("foa", self.seed, it, x, out["npv"], out.get("prod_curve"))
        return out["npv"]
