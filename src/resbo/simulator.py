"""Interfaz de simulador y MockSimulator (juguetes locales, sin MC2/OPM).

Contrato para todos los simuladores:
    sim = SomeSimulator(protocol)
    out = sim.evaluate(controls)   # controls = [rate x n_inj, bhp x n_prod]
    out["npv"], out["prod_curve"]  # (n_prod, horizon), produccion por pozo/mes

`OpmSimulator` (MC2) es un STUB: documenta como envolver el motor headless.
"""
from __future__ import annotations

import numpy as np


class BaseSimulator:
    def evaluate(self, controls: np.ndarray) -> dict:
        raise NotImplementedError

    def evaluate_many(self, controls: np.ndarray) -> list[dict]:
        return [self.evaluate(c) for c in controls]


class MockSimulator(BaseSimulator):
    """Objetivo tipo Egg: conectividades τ_true (8x4) ocultas + dynamics simples.

    Por productor j: presion p_j sigue balance c*dp/dt = soporte_j - q_j,
    con soporte_j = sum_i τ_ij * rate_i * lag(t) (soporte de inyector retardado).
    Queda implicito que la produccion "ve" la conectividad -> un CRM puede
    recuperarla. La fraccion de agua sube con el volumen inyectado acumulado
    dirigido al pozo (breakthrough) -> inyectar de mas acelera agua (coste).

    Economia (T-R1): replica las TRES tarifas de NPVHandler del caso EGG_TEST:
        cash_t = OPR_t*oil_revenue - WPR_t*water_cost - WIR_t*water_injection_cost
    (volúmenes mensuales = tasa*days; descuentos mensuales de la tasa anual).
    Calibrado a la caja real 0-150 sm3/d / 200-395 bar: PI/p0/pv elegidos para
    que el optimum quede INTERIOR (producir sin agua cuesta; inyectar de mas
    paga waterInjectionCost y adelanta breakthrough).
    """

    def __init__(self, protocol, seed: int = 42):
        self.n_inj = protocol.n_inj
        self.n_prod = protocol.n_prod
        self.T = protocol.horizon
        e = protocol["economics"]
        self.oil_rev = e["oil_revenue"]
        self.water_cost = e["water_cost"]
        self.inj_cost = e["water_injection_cost"]
        self.disc = (1.0 + e["annual_discount"]) ** (1.0 / 12.0)

        rng = np.random.default_rng(seed)
        tau = rng.uniform(0.3, 1.0, (self.n_inj, self.n_prod))
        tau = tau / tau.sum(axis=1, keepdims=True)   # cada inyector reparte 1.0
        self.tau_true = tau

        self.PI = 2.5                                 # productividad (m3/d / bar)
        self.c_store = 12.0                           # capacidad (bar*d/m3)
        self.p0 = 380.0                               # presion inicial (bar)
        self.lag = 24.0                               # meses, retardo del soporte
        self.pv = 8.0e5                               # volumen poroso (m3) por productor
        self.q_ref = 400.0                            # caudal de escala del cono (m3/d)
        self.cone_v = 0.3                             # cono dinamico por caudal
        self.p_woc = 330.0                            # presion de contacto agua-aceite
        self.cone_b = 0.9                             # cono por drawdown bajo el WOC
        self.days = 30.0

    def evaluate(self, controls: np.ndarray) -> dict:
        rate = np.asarray(controls[: self.n_inj], float)
        bhp = np.asarray(controls[self.n_inj:], float)

        t = np.arange(1, self.T + 1)
        support = (rate @ self.tau_true)              # (n_prod,) soporte promedio
        support_t = support[None, :] * (1.0 - np.exp(-t[:, None] / self.lag))

        inj_vol_month = rate * self.days              # (n_inj,)
        # volumen inyectado dirigido a cada pozo, acumulado con el mismo lag
        directed = (inj_vol_month @ self.tau_true)    # (n_prod,)
        cum_dir = np.cumsum(
            directed[None, :] * (1.0 - np.exp(-t[:, None] / self.lag)), axis=0
        )                                             # (T, n_prod)

        p = np.empty((self.T, self.n_prod))
        q = np.empty((self.T, self.n_prod))
        p_cur = np.full(self.n_prod, self.p0)
        for k in range(self.T):
            q[k] = np.maximum(self.PI * (p_cur - bhp), 0.0)
            p_cur = p_cur + (support_t[k] - q[k]) / self.c_store
            p[k] = p_cur

        # breakthrough: el volumen inyectado DIRIGIDO acumulado perfora el poro
        # (cum_dir/pv) y dos mecanismos lo aceleran: caudal alto (cono dinamico,
        # cone_v*q) y producir por debajo del contacto agua-aceite (cono por
        # drawdown, cone_b*(p_woc-bhp)). Bajar BHP sube q PERO acerca el agua.
        dd = np.maximum(self.p_woc - bhp[None, :], 0.0) / 100.0   # (1, n_prod)
        fw = np.clip(0.15 + 0.85 * (cum_dir / self.pv) ** 0.6
                     * (1.0 + self.cone_v * q / self.q_ref + self.cone_b * dd),
                     0.15, 0.98)
        q_oil = q * (1.0 - fw)
        q_liq = q
        q_wat = q * fw

        disc = self.disc ** (-t)
        # las 3 tarifas de NPVHandler (EGG_TEST): aceite - agua producida - agua inyectada
        cash = (q_oil.sum(axis=1) * self.oil_rev
                - q_wat.sum(axis=1) * self.water_cost
                - rate.sum() * self.inj_cost) * self.days
        npv = float((cash * disc).sum())
        # convencion del proyecto: curvas en (n_prod, horizon)
        return {"npv": npv, "prod_curve": q_liq.T, "fw_curve": fw.T}


class OpmSimulator(BaseSimulator):
    """STUB para MC2 — envolver el motor headless del entorno BIOPSO.

    Plan de implementacion (tarea 1.3 del cronograma, en MC2/WSL):
      1. En BIOPSO, `interface/run_generic.py` ya escribe results.csv y carpetas
         de corrida. Aqui se reutiliza el mismo motor de spawn:
           mpirun -n 1 flow --output-dir <run_dir> EGG_MODEL.DATA
         y se parsean los summary (WOPR/WBHP/WWCT por pozo, 130 meses) con
         opm/io o `flow` summary -> (n_prod, horizon).
      2. NPV igual formula que MockSimulator (price/cost/descuento del protocolo).
      3. evaluate_many: cola local con max 7 procesos (regla MC2), reintentos
         por sims caidas, y guardado incremental en SimDatabase.add(...).
    Lanza NotImplementedError para que nada "silencioso" dependa de esto local.
    """

    def __init__(self, protocol, workdir: str = "/mnt/c/DAGO/BIOPSO/experiments/opm"):
        raise NotImplementedError("Ver docstring: implementar en MC2 (tarea 1.3).")
