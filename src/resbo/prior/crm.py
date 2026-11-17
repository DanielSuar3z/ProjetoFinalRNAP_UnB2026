"""CRM (Capacitance-Resistance Model) como prior fisico para la adquisicao.

Modelo (steady-state simplificado, Sayarpour 2009 / You-Albertazz GCPRM):
    q_j(t) ~= sum_i tau_ij * I_i(t) + (1 - sum_i tau_ij) * q_j^init

Con controles CONSTANTES por corrida (el caso BIOPSO), la excitacion del sistema
viene de las DISTINTAS corridas: cada run aporta (rates, produccion media del
late-time). Apilando runs, tau_ij se identifica por minimos cuadrados con
restriccion tau >= 0 y fila sumando <= 1.

El prior no sustituye simulaciones: convierte las conectividades estimadas en
una puntuacion escalar por candidato (soporte de presion + expectativa de
breakthrough + balance de voidage como caso degenerado), que la adquisicao
usa como sesgo decreciente en el tiempo (estilo DynMeanBO, ICLR 2026).
"""
from __future__ import annotations

import json

import numpy as np


class CRMModel:
    def __init__(self, n_inj: int, n_prod: int, min_excitation: float = 50.0):
        self.n_inj = n_inj
        self.n_prod = n_prod
        self.min_excitation = min_excitation
        self.tau = np.zeros((n_inj, n_prod))   # conectividades
        self.q0 = np.zeros(n_prod)             # produccion "natural" sin soporte
        self.fit_r2 = np.zeros(n_prod)
        self.n_runs_used = 0

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, curves: list[np.ndarray],
            late_frac: float = 0.4) -> "CRMModel":
        """Ajusta tau por productor via NNLS simplificado (lstsq + clip).

        X: (n_runs, 12) controles por corrida
        curves: lista de (n_prod, horizon) de produccion por pozo
        late_frac: usa el ultimo tramo (late-time) donde el steady-state vale.
        """
        n = len(curves)
        rates, prods = [], []
        for c, ctrl in zip(curves, X):
            T = c.shape[1]
            t0 = int(T * (1.0 - late_frac))
            q_mean = c[:, t0:].mean(axis=1)          # (n_prod,)
            rates.append(ctrl[: self.n_inj])
            prods.append(q_mean)
        if n < self.n_prod + 1:
            raise ValueError(f"CRM necesita >= {self.n_prod + 1} runs excitadas; hay {n}")
        R = np.array(rates)                            # (n, n_inj)
        Q = np.array(prods)                            # (n, n_prod)

        for j in range(self.n_prod):
            A = np.hstack([R, np.ones((n, 1))])
            y = Q[:, j]
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            tau_j = np.clip(sol[: self.n_inj], 0.0, 1.0)
            q0_j = max(sol[self.n_inj], 0.0)
            self.tau[:, j] = tau_j
            self.q0[j] = q0_j
            yhat = A @ np.concatenate([tau_j, [q0_j]])
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
            self.fit_r2[j] = 1.0 - ss_res / ss_tot
        self.n_runs_used = n
        return self

    # ------------------------------------------------------------------
    def score(self, controls: np.ndarray) -> float:
        """Prior score por candidato (mayor = mas promisorio fisicamente).

        Componentes (todas adimensionales, en [0,1] tras normalizar):
          1. soporte de presion predicho  s = sum_j (sum_i tau_ij rate_i) / sum rate
          2. castigo por voidage desbalanceado: el steady predicho q_total
             deberia poder ser sostenido por la inyeccion (v = q_total/r_total ~ 1)
          3. expectativa de breakthrough: concentracion de soporte (un solo
             inyector dominando a un productor rompe agua antes)
        El voidage balance es el caso degenerado (termino 2 solo).
        """
        rate = controls[: self.n_inj]
        support = rate @ self.tau                     # (n_prod,)
        r_tot = rate.sum() + 1e-9
        s = support.sum() / r_tot                     # en [0,1] (filas de tau suman 1)
        voidage = 1.0 - abs(1.0 - support.sum() / r_tot)
        # concentración de soporte: Gini-like sobre soporte por productor
        p = support / (support.sum() + 1e-9)
        concentration = float((p ** 2).sum())          # 1/n_prod uniforme -> 1 dominante
        brk = 1.0 - concentration
        return float(0.45 * s + 0.40 * voidage + 0.15 * brk)

    def score_many(self, X: np.ndarray) -> np.ndarray:
        rate = X[:, : self.n_inj]
        support = rate @ self.tau
        r_tot = rate.sum(axis=1) + 1e-9
        s = support.sum(axis=1) / r_tot
        voidage = 1.0 - np.abs(1.0 - support.sum(axis=1) / r_tot)
        p = support / (support.sum(axis=1, keepdims=True) + 1e-9)
        brk = 1.0 - (p ** 2).sum(axis=1)
        return 0.45 * s + 0.40 * voidage + 0.15 * brk


class VoidagePrior:
    """Prior escalar "balance de voidage" (sin CRM). Baseline de ablacion.

    Sin conectividades no hay prediccion de soporte: el unico escalar fisicamente
    informativo derivable del candidato es su propio balance de masa global, que
    para un vector de control constante es identicamente ~1 -> el prior degenera
    a constante y no informa. Por eso la ablacion "sin prior CRM" usa score=0.
    """

    def __init__(self, n_inj: int, n_prod: int):
        self.n_inj, self.n_prod = n_inj, n_prod

    def score(self, controls: np.ndarray) -> float:
        return 0.0

    def score_many(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


# --------------------------------------------------------------- crm_v1.npz
# Contracto de persistencia del prior (hermano de bnn_v1.npz, misma filosofia:
# npz sin pickle, schema versionado, legible por el brazo BO del lado BIOPSO).
#
#   schema      b"crm_v1"
#   tau         (n_inj, n_prod) float64  — tau FINAL (columnas de pozos fallidos
#                                          en cero: el score ignora pozos no confiables)
#   tau_full    (n_inj, n_prod)          — tau cruda del ajuste (auditoria)
#   q0          (n_prod,)                — produccion "natural" sin soporte
#   fit_r2      (n_prod,)                — R2 late-time por pozo (la metrica del gate)
#   well_mask   (n_prod,) int8           — 1 = pozo confiable (R2 >= r2_min)
#   meta_json   str                      — proveniencia, hiperparametros, fecha
#
# well_mask vive AQUI y no solo en el metrics.json porque T-R5 (BIOPSO) debe
# poder decidir sin leer archivos sueltos: si un pozo falla el gate, su columna
# de tau en cero hace que score() simplemente no lo tome en cuenta.

CRM_SCHEMA = "crm_v1"


def save_crm_npz(path: str, crm: "CRMModel", well_mask: np.ndarray,
                 meta: dict | None = None) -> str:
    meta = dict(meta or {})
    np.savez_compressed(
        path,
        schema=np.array(CRM_SCHEMA),
        tau=crm.tau, tau_full=np.asarray(meta.pop("tau_full", crm.tau), float),
        q0=crm.q0, fit_r2=crm.fit_r2,
        well_mask=np.asarray(well_mask, dtype="int8"),
        n_runs_used=np.int64(crm.n_runs_used),
        meta_json=np.array(json.dumps(meta, default=str)),
    )
    return path


def load_crm_npz(path: str) -> tuple["CRMModel", np.ndarray, dict]:
    """Devuelve (crm, well_mask, meta). Rechaza schemas distintos de crm_v1."""
    with np.load(path, allow_pickle=False) as z:
        schema = str(z["schema"])
        if schema != CRM_SCHEMA:
            raise ValueError(f"{path}: schema '{schema}', esperado '{CRM_SCHEMA}'")
        tau = np.asarray(z["tau"], float)
        crm = CRMModel(n_inj=tau.shape[0], n_prod=tau.shape[1])
        crm.tau = tau
        crm.q0 = np.asarray(z["q0"], float)
        crm.fit_r2 = np.asarray(z["fit_r2"], float)
        crm.n_runs_used = int(z["n_runs_used"])
        mask = np.asarray(z["well_mask"], dtype="int8")
        meta = json.loads(str(z["meta_json"]))
    return crm, mask, meta
