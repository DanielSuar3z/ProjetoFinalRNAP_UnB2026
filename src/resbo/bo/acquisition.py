"""Funcion de adquisicao EI + prior CRM con decaimiento (estilo DynMeanBO).

score(x) = rank01[EI](x) + w(n) * rank01[prior](x)
  w(n) = w0 * exp(-n_obs / tau_decay)  ->  el prior pesa al inicio (data-scarce)
  y se apaga cuando el surrogate ya tiene cobertura (DynMeanBO, ICLR 2026).

Combinamos por RANGOS (no valores crudos): el EI de un ensemble MC-dropout
tiene picos enormes que, en z-score de valores, swampearian cualquier
contribucion del prior por grande que sea w. Con rangos las dos senales son
comparables y `w` controla la mezcla de forma interpretable.
"""
from __future__ import annotations

import numpy as np
from math import erf


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def expected_improvement(mu: np.ndarray, sigma: np.ndarray,
                         f_best: float, xi: float = 0.0) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    imp = mu - f_best - xi
    z = imp / sigma
    ei = imp * _norm_cdf(z) + sigma * np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    return np.maximum(ei, 0.0)


def _rank01(v: np.ndarray) -> np.ndarray:
    """Rango normalizado [0,1]."""
    order = np.argsort(np.argsort(v))
    return order / max(len(v) - 1, 1)


def acquisition(mu, sigma, f_best, prior_scores=None, w_prior=0.0, xi=0.0):
    """Combina EI (surrogate) con prior fisico (opcional), mezcla por rangos."""
    ei = expected_improvement(mu, sigma, f_best, xi)
    s = _rank01(ei)
    if prior_scores is not None and w_prior > 0.0:
        s = s + w_prior * _rank01(prior_scores)
    return s


def prior_weight(n_obs: float, w0: float, tau_decay: float) -> float:
    """Decaimiento del prior con el numero de observaciones (DynMeanBO)."""
    return float(w0 * np.exp(-n_obs / max(tau_decay, 1e-9)))
