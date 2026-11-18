"""CRM (Capacitance-Resistance Model) como prior fisico para la adquisicao.

Modelo (steady-state simplificado, Sayarpour 2009 / You-Albertazz GCPRM):
    q_j(t) ~= sum_i tau_ij * I_i(t) + beta_j * BHP_j + q_j^init

Con controles CONSTANTES por corrida (el caso BIOPSO), la excitacion del sistema
viene de las DISTINTAS corridas: cada run aporta (rates, produccion media del
late-time). Apilando runs, tau_ij se identifica por minimos cuadrados con
restriccion tau >= 0 y fila sumando <= 1 (ajuste conjunto, ver fit()).

El prior no sustituye simulaciones: convierte las conectividades estimadas en
una puntuacion escalar por candidato (soporte de presion + expectativa de
breakthrough + balance de voidage como caso degenerado), que la adquisicao
usa como sesgo decreciente en el tiempo (estilo DynMeanBO, ICLR 2026).
"""
from __future__ import annotations

import json

import numpy as np


def _project_rows(tau: np.ndarray) -> np.ndarray:
    """Proyeccion euclidea de cada fila sobre {t >= 0, sum(t) <= 1}."""
    out = np.maximum(tau, 0.0)
    for i in range(out.shape[0]):
        if out[i].sum() > 1.0:                     # proyectar al simplex
            u = np.sort(tau[i])[::-1]
            css = np.cumsum(u) - 1.0
            k = np.arange(1, len(u) + 1)
            rho = np.nonzero(u - css / k > 0)[0][-1]
            out[i] = np.maximum(tau[i] - css[rho] / (rho + 1), 0.0)
    return out


class CRMModel:
    def __init__(self, n_inj: int, n_prod: int, min_excitation: float = 50.0):
        self.n_inj = n_inj
        self.n_prod = n_prod
        self.min_excitation = min_excitation
        self.tau = np.zeros((n_inj, n_prod))   # conectividades
        self.q0 = np.zeros(n_prod)             # produccion "natural" sin soporte
        self.fit_r2 = np.zeros(n_prod)
        self.beta_bhp = np.zeros(n_prod)       # sensibilidad del caudal al BHP
        self.score_w = None                    # pesos del score calib_v1
        self.score_mu = self.score_sd = None
        self.n_runs_used = 0

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, curves: list[np.ndarray],
            late_frac: float = 0.4, n_iter: int = 3000) -> "CRMModel":
        """Ajuste CONJUNTO de tau con restricciones reales + BHP como regresor.

        Por productor j (late-time, controles constantes por corrida):
            q_j = sum_i tau_ij * rate_i + beta_j * bhp_j + c_j
        con  tau_ij >= 0  y  sum_j tau_ij <= 1  (un inyector no reparte mas
        agua de la que inyecta). La restriccion de fila ACOPLA los productores,
        asi que no basta un lstsq por pozo + clip (lo que habia antes, y que
        el docstring prometia sin cumplir).

        beta_j: en Flow el BHP de cada productor cambia entre corridas y domina
        su caudal; sin este regresor esa varianza cae en el residuo y el R2
        por pozo se hunde (en el mock casi no pesa porque el late-time ya
        equilibro q_j ~ soporte_j). beta_j y c_j son libres; se eliminan
        proyectando sobre span{1, bhp_j} y tau se resuelve por gradiente
        proyectado (problema convexo, 32 variables).

        X: (n_runs, n_inj + n_prod) controles; curves: lista de (n_prod, T).
        """
        n = len(curves)
        if n < self.n_inj + 2:
            raise ValueError(f"CRM necesita >= {self.n_inj + 2} runs excitadas; hay {n}")
        X = np.asarray(X, float)
        R = X[:, : self.n_inj]                                      # (n, n_inj)
        B = X[:, self.n_inj: self.n_inj + self.n_prod]              # (n, n_prod)
        Q = np.empty((n, self.n_prod))
        for k, c in enumerate(curves):
            t0 = int(c.shape[1] * (1.0 - late_frac))
            Q[k] = c[:, t0:].mean(axis=1)

        # Proyectores que eliminan (beta_j, c_j): M_j = I - H_j, H_j sobre [1, bhp_j]
        Ms, RtMR, RtMQ = [], [], []
        for j in range(self.n_prod):
            Z = np.column_stack([np.ones(n), B[:, j]])
            pinv = np.linalg.pinv(Z)
            MR = R - Z @ (pinv @ R)
            MQ = Q[:, j] - Z @ (pinv @ Q[:, j])
            Ms.append((Z, pinv))
            RtMR.append(R.T @ MR)
            RtMQ.append(R.T @ MQ)
        L = max(np.linalg.eigvalsh(A).max() for A in RtMR) + 1e-12

        # arranque: lstsq sin restricciones proyectado; luego PGD acelerado (FISTA)
        tau = np.column_stack([np.linalg.lstsq(RtMR[j], RtMQ[j], rcond=None)[0]
                               for j in range(self.n_prod)])
        tau = _project_rows(tau)
        y_k, t_k = tau.copy(), 1.0
        for _ in range(n_iter):
            G = np.column_stack([RtMR[j] @ y_k[:, j] - RtMQ[j]
                                 for j in range(self.n_prod)])
            tau_new = _project_rows(y_k - G / L)
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2))
            y_k = tau_new + ((t_k - 1.0) / t_new) * (tau_new - tau)
            if np.abs(tau_new - tau).max() < 1e-10:
                tau = tau_new
                break
            tau, t_k = tau_new, t_new

        self.tau = tau
        self.beta_bhp = np.zeros(self.n_prod)
        for j in range(self.n_prod):
            Z, pinv = Ms[j]
            coef = pinv @ (Q[:, j] - R @ tau[:, j])                 # [c_j, beta_j]
            self.q0[j] = coef[0]
            self.beta_bhp[j] = coef[1]
            yhat = R @ tau[:, j] + Z @ coef
            y = Q[:, j]
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
            self.fit_r2[j] = 1.0 - ss_res / ss_tot
        self.n_runs_used = n
        return self

    # ------------------------------------------------------------------
    # Score del prior.
    #
    # LEGACY (hasta 2026-11-17): 0.45*s + 0.40*voidage + 0.15*brk con pesos a
    # mano. Problema: voidage = 1-|1-s| == s cuando s <= 1 (filas de tau <= 1),
    # o sea 0.85*s + 0.15*brk; ademas no mira la economia y en el toy empujaba
    # a sobreinyectar (bo 16.8M > bo_prior 16.2M).
    #
    # CALIB_V1 (actual): mismas magnitudes fisicas del CRM como FEATURES, pero
    # los pesos salen de una regresion ridge del NPV sobre ellas, ajustada UNA
    # vez con el pool de baselines (antes del BO) y congelada. El prior sigue
    # siendo conocimiento de dominio (5 features fisicas, no un 2o surrogate
    # sobre los 12 controles crudos), con pesos defendibles en vez de a mano.
    SCORE_FEATURES = ("soporte", "dispersion", "inyeccion", "inyeccion2", "q_pred")

    def features(self, X: np.ndarray) -> np.ndarray:
        """(n, 5) features fisicas del CRM por candidato.

        soporte    = sum_j (rate @ tau)_j / sum(rate): fraccion de la inyeccion
                     que llega a productores (eficiencia de barrido)
        dispersion = 1 - sum p_j^2 del soporte por productor (breakthrough:
                     soporte concentrado en un pozo rompe agua antes)
        inyeccion  = sum(rate) y su cuadrado (permite un optimo INTERIOR:
                     el agua inyectada cuesta, waterInjectionCost)
        q_pred     = sum_j (rate @ tau_j + beta_j*bhp_j + q0_j): caudal late-time
                     predicho por el CRM con BHP (en datos Flow la curva es WPR,
                     asi que es agua predicha; la regresion aprende su signo)
        """
        X = np.atleast_2d(np.asarray(X, float))
        rate = X[:, : self.n_inj]
        bhp = X[:, self.n_inj: self.n_inj + self.n_prod]
        support = rate @ self.tau
        r_tot = rate.sum(axis=1) + 1e-9
        s = support.sum(axis=1) / r_tot
        p = support / (support.sum(axis=1, keepdims=True) + 1e-9)
        brk = 1.0 - (p ** 2).sum(axis=1)
        inj = rate.sum(axis=1)
        beta = getattr(self, "beta_bhp", np.zeros(self.n_prod))
        q_pred = (support + bhp * beta + self.q0).sum(axis=1)
        return np.column_stack([s, brk, inj, inj ** 2, q_pred])

    @property
    def calibrated(self) -> bool:
        return getattr(self, "score_w", None) is not None

    def calibrate_score(self, X: np.ndarray, y: np.ndarray, ridge: float = 1.0,
                        holdout: float = 0.2, seed: int = 0) -> dict:
        """Ajusta los pesos del score (ridge sobre features estandarizadas).

        Llamar DESPUES de aplicar el well_mask a tau (el score usa la tau final).
        Valida en un holdout (Spearman score-vs-NPV, la cantidad que importa
        porque la adquisicion combina por RANGOS), compara con el score legacy
        y luego reajusta con todo. Devuelve el informe para el metrics.json.
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        if len(y) < 20:
            raise ValueError(f"calibrar el score necesita >= 20 sims; hay {len(y)}")
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        n_ho = max(int(round(holdout * len(y))), 5)
        ho, tr = idx[:n_ho], idx[n_ho:]

        F = self.features(X)
        self._fit_weights(F[tr], y[tr], ridge)
        rho_cal = _spearman(self._score_calibrated(F[ho]), y[ho])
        rho_leg = _spearman(self._score_legacy(X[ho]), y[ho])
        self._fit_weights(F, y, ridge)                  # pesos finales: todo el pool
        return {"score_kind": "calib_v1", "ridge": ridge,
                "n_calib": int(len(y)), "n_holdout": int(n_ho),
                "spearman_holdout_calib": round(rho_cal, 4),
                "spearman_holdout_legacy": round(rho_leg, 4),
                "features": list(self.SCORE_FEATURES),
                "pesos_estandarizados": [round(float(w), 6) for w in self.score_w]}

    def _fit_weights(self, F, y, ridge):
        self.score_mu = F.mean(axis=0)
        self.score_sd = F.std(axis=0) + 1e-12
        Z = (F - self.score_mu) / self.score_sd
        yc = (y - y.mean()) / (y.std() + 1e-12)
        self.score_w = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ yc)

    def _score_calibrated(self, F):
        return ((F - self.score_mu) / self.score_sd) @ self.score_w

    def _score_legacy(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        rate = X[:, : self.n_inj]
        support = rate @ self.tau
        r_tot = rate.sum(axis=1) + 1e-9
        s = support.sum(axis=1) / r_tot
        voidage = 1.0 - np.abs(1.0 - support.sum(axis=1) / r_tot)
        p = support / (support.sum(axis=1, keepdims=True) + 1e-9)
        brk = 1.0 - (p ** 2).sum(axis=1)
        return 0.45 * s + 0.40 * voidage + 0.15 * brk

    def score_many(self, X: np.ndarray) -> np.ndarray:
        """Score por candidato (mayor = mas promisorio). calib_v1 si hay pesos."""
        if self.calibrated:
            return self._score_calibrated(self.features(X))
        return self._score_legacy(X)

    def score(self, controls: np.ndarray) -> float:
        return float(self.score_many(np.asarray(controls, float)[None, :])[0])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


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
#   beta_bhp    (n_prod,)  [opcional]    — sensibilidad del caudal al BHP del productor
#   score_w/mu/sd (5,)     [opcional]    — pesos del score calib_v1 (sin ellos: legacy)
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
        beta_bhp=np.asarray(getattr(crm, "beta_bhp", np.zeros(len(crm.q0))), float),
        **({"score_w": crm.score_w, "score_mu": crm.score_mu,
            "score_sd": crm.score_sd} if crm.calibrated else {}),
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
        # opcional (anadido 2026-11-17, compatible hacia atras con crm_v1 previos)
        crm.beta_bhp = (np.asarray(z["beta_bhp"], float) if "beta_bhp" in z.files
                        else np.zeros(tau.shape[1]))
        if "score_w" in z.files:                       # score calib_v1
            crm.score_w = np.asarray(z["score_w"], float)
            crm.score_mu = np.asarray(z["score_mu"], float)
            crm.score_sd = np.asarray(z["score_sd"], float)
        crm.n_runs_used = int(z["n_runs_used"])
        mask = np.asarray(z["well_mask"], dtype="int8")
        meta = json.loads(str(z["meta_json"]))
    return crm, mask, meta
