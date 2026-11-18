"""Metricas de convergencia y tests estadisticos para los 4 brazos."""
from __future__ import annotations

import numpy as np


def best_so_far(npvs: np.ndarray) -> np.ndarray:
    """Curva de maximo corriente (NPV acumulado por nº de sims)."""
    return np.maximum.accumulate(np.asarray(npvs, float))


def auc(history: list[tuple[int, float]], n_total: int | None = None) -> float:
    """Area bajo la curva best-so-far vs nº de sims (mayor = converge antes)."""
    if not history:
        return 0.0
    n = np.array([h[0] for h in history], float)
    v = np.array([h[1] for h in history], float)
    n_total = n_total or n[-1]
    step = np.diff(np.concatenate([[0], n]))
    return float((v * step).sum() / max(n_total, 1))


def sims_to_threshold(history: list[tuple[int, float]], threshold: float) -> float:
    """Primera simulacion en alcanzar `threshold` (o n_total+1 si nunca)."""
    for n, v in history:
        if v >= threshold:
            return float(n)
    return float(history[-1][0] + 1) if history else float("nan")


# ----------------------------------------------------------------------
def mannwhitney(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """U de Mann-Whitney bilateral + p (aprox normal). Sin scipy."""
    from math import erfc, sqrt
    x, y = np.asarray(x, float), np.asarray(y, float)
    allv = np.concatenate([x, y])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # promediar empates
    uniq, inv = np.unique(allv, return_inverse=True)
    if len(uniq) < len(allv):
        for k in range(len(uniq)):
            m = inv == k
            ranks[m] = ranks[m].mean()
    rx = ranks[: len(x)].sum()
    U1 = rx - len(x) * (len(x) + 1) / 2
    U2 = len(x) * len(y) - U1
    U = min(U1, U2)
    mu = len(x) * len(y) / 2
    sigma = sqrt(len(x) * len(y) * (len(x) + len(y) + 1) / 12)
    z = (U - mu) / (sigma + 1e-12)
    p = erfc(abs(z) / sqrt(2))                      # bilateral
    return float(U), float(min(p, 1.0))


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Tamaño de efecto r_rb en [-1,1] (x vs y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    U1, U2 = [], []
    for a in x:
        for b in y:
            if a > b:
                U1.append(1)
            elif a < b:
                U2.append(1)
    # efecto = (wins_x - wins_y) / (nx*ny)
    wins = sum(U1) - sum(U2)
    return wins / (len(x) * len(y))


def equivalence_bootstrap(x: np.ndarray, y: np.ndarray, delta: float,
                          ci: float = 0.90, n_boot: int = 10000,
                          seed: int = 0) -> dict:
    """Prueba de EQUIVALENCIA de medianas (x vs y) con margen absoluto delta.

    Mann-Whitney con p > 0.05 NO demuestra equivalencia (ausencia de evidencia
    no es evidencia de ausencia). Aca: IC percentil bootstrap (remuestreo
    independiente por grupo) de d = mediana(x) - mediana(y); se declara
    equivalencia si el IC al `ci` cae ENTERO dentro de (-delta, +delta).
    Con ci=0.90 equivale a dos pruebas unilaterales al 5 % (TOST, alpha=0.05).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    bx = rng.choice(x, size=(n_boot, x.size), replace=True)
    by = rng.choice(y, size=(n_boot, y.size), replace=True)
    d = np.median(bx, axis=1) - np.median(by, axis=1)
    a = (1.0 - ci) / 2.0
    lo, hi = np.quantile(d, [a, 1.0 - a])
    return {"diff_mediana": float(np.median(x) - np.median(y)),
            "ic_lo": float(lo), "ic_hi": float(hi), "ic_nivel": ci,
            "delta": float(delta),
            "equivalente": bool(-delta < lo and hi < delta)}
