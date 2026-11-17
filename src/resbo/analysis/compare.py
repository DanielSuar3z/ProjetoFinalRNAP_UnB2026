"""Agregado de los 6 brazos (S5): NPVs finales + Mann-Whitney/rank-biserial.

Entra por el MISMO format que deja el harness de BIOPSO en cada brazo:

    <dir>/results.csv          run_id,npv_final,elapsed_seconds,timestamp
    <dir>/run_XX/convergence.csv   iteration,best_npv      (opcional, p/AUC)
    <dir>/run_XX/simlog.csv        contrato simlog_spec_v1 (opcional, p/AUC)

Los brazos de la coleta S1 (mfo/gwo/pso/foa, 5 seeds) y los de S3
(bo_prior/bo_solo, N=20 con contingencia 16->12) conviven aqui: la unica
diferencia es cuantas filas tiene su results.csv. Los runs de RESCATE
(harvest sin simlog, 1 fila via sidecar) tambien valen para el NPV final;
solo quedan fuera del AUC (no tienen convergence/simlog), y asi se reporta.

Politicas honestas (verificado por tests):
  * run_id duplicado en results.csv -> gana la ULTIMA fila (un re-intento
    registra al final; el harness normaliza borrando, pero no confiamos).
  * AUC en unidades de SIMULACIONES, no de iteraciones: la geometria de
    rondas es distinta por brazo (15x6 classico vs 10+5x16 del BO), y solo
    el conteo de sims del simlog hace la comparacion justa. Sin simlog, la
    fila converge en iteraciones y se marca AUC=nan en vez de mentir.
  * Mann-Whitney bilateral + r_rb del proyecto (analysis/metrics.py): la
    MISMA funcion que valido la demo toy, no una reimplementacion.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np

from .metrics import auc, mannwhitney, rank_biserial

# nombres canonicos del experimento (tabla de 6 brazos do projeto)
ARMS_CANON = ("mfo", "gwo", "pso", "foa", "bo_prior", "bo_solo")


class ArmData:
    """Un brazo: NPVs finales por seed + historias best-so-far (sims, npv)."""

    def __init__(self, name: str):
        self.name = name
        self.npvs: list[float] = []          # por run_id (ultima fila gana)
        self.histories: dict[int, list[tuple[int, float]]] = {}
        self.sin_curvas: list[int] = []      # run_ids sin historia (rescate)

    @property
    def y(self) -> np.ndarray:
        return np.asarray(self.npvs, float)

    def aucs(self) -> np.ndarray:
        return np.array([auc(h) for h in self.histories.values()], float)


def _cum_sims_por_iteracion(simlog_path: str) -> dict[int, int]:
    """iteration -> sims ACUMULADAS hasta esa ronda, contado del simlog real.

    Cubre las dos geometrias (clasica sin ronda 0; resbo con ronda 0 DoE)
    sin hardcodear agents/n_init: el conteo de filas ES la moneda."""
    acum: dict[int, int] = {}
    conteo: dict[int, int] = {}
    with open(simlog_path, newline="") as f:
        rdr = csv.reader(f)
        next(rdr, None)                      # cabecera
        for row in rdr:
            if not row:
                continue
            it = int(float(row[0]))
            conteo[it] = conteo.get(it, 0) + 1
    for it in sorted(conteo):
        acum[it] = (acum[it - 1] if (it - 1) in acum else 0) + conteo[it]
    return acum


def _historia(run_dir: str) -> list[tuple[int, float]] | None:
    """[(n_sims, best_npv)] del run, o None si no hay artefactos (rescate)."""
    conv = os.path.join(run_dir, "convergence.csv")
    slog = os.path.join(run_dir, "simlog.csv")
    if not os.path.exists(conv):
        return None
    puntos: list[tuple[int, float]] = []
    if os.path.exists(slog):
        cum = _cum_sims_por_iteracion(slog)
        with open(conv, newline="") as f:
            rdr = csv.reader(f)
            hdr = next(rdr, None) or []
            for row in rdr:
                if not row:
                    continue
                it = int(float(row[0]))
                if it not in cum:            # convergence dice una ronda que
                    return None              # el simlog no registro: no mezclar
                puntos.append((cum[it], float(row[1])))
    else:
        return None                          # sin simlog: AUC en iteraciones
        # seria injusto contra el brazo BO; mejor nan explicito
    return puntos or None


def load_arm(name: str, path: str) -> ArmData:
    """`path` es un directorio de brazo (con results.csv) o un results.csv.

    Si es un directorio, busca los run_XX/ hermanos para convergence/simlog
    (el layout de run_generic.py y de COLECTA_S1/S3)."""
    if os.path.isfile(path):
        results = path
        base = os.path.dirname(path)
    else:
        results = os.path.join(path, "results.csv")
        base = path
    if not os.path.exists(results):
        raise FileNotFoundError(f"brazo '{name}': no existe {results}")
    arm = ArmData(name)
    por_run: dict[int, float] = {}
    with open(results, newline="") as f:
        for row in csv.DictReader(f):
            por_run[int(float(row["run_id"]))] = float(row["npv_final"])
    for rid in sorted(por_run):              # ultima fila gana (re-intento)
        arm.npvs.append(por_run[rid])
        hist = _historia(os.path.join(base, f"run_{rid:02d}"))
        if hist is None:
            arm.sin_curvas.append(rid)
        else:
            arm.histories[rid] = hist
    if not arm.npvs:
        raise ValueError(f"brazo '{name}': results.csv sin filas")
    return arm


def resumen_brazo(arm: ArmData, threshold: float | None = None) -> dict:
    """Fila de la tabla de brazos (los numeros del artigo)."""
    y = arm.y
    fila = {
        "arm": arm.name,
        "n": int(y.size),
        "mean": float(y.mean()),
        "median": float(np.median(y)),
        "std": float(y.std(ddof=1)) if y.size > 1 else float("nan"),
        "min": float(y.min()),
        "max": float(y.max()),
        "auc_mean": float(np.nanmean(arm.aucs())) if len(arm.histories) else float("nan"),
        "runs_sin_curvas": int(len(arm.sin_curvas)),
    }
    if threshold is not None:
        from .metrics import sims_to_threshold
        st = [sims_to_threshold(h, threshold) for h in arm.histories.values()]
        fila["sims_al_umbral_mediana"] = float(np.median(st)) if st else float("nan")
    return fila


def pairwise(x: np.ndarray, y: np.ndarray, nombre_x: str, nombre_y: str) -> dict:
    """Mann-Whitney bilateral + r_rb en la direccion x vs y (x mejor > 0)."""
    U, p = mannwhitney(x, y)
    rb = rank_biserial(x, y)
    return {
        "comparacion": f"{nombre_x} vs {nombre_y}",
        "n_x": int(np.size(x)), "n_y": int(np.size(y)),
        "mean_diff": float(np.mean(x) - np.mean(y)),
        "U": float(U), "p_valor": float(p), "r_biserial": float(rb),
        "significativo_0.05": bool(p < 0.05),
        # con n pequenos la aproximacion normal es generosa: advertirlo en la
        # propia tabla (el artigo reporta el tamano de efecto como principal)
        "aviso_n_bajo": bool(np.size(x) < 8 or np.size(y) < 8),
    }


def tabla_markdown(filas: list[dict], pares: list[dict]) -> str:
    """Las dos tablas del artigo en markdown (CSV es la fuente, esto es lectura)."""
    def _f(v, d=0):
        return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) \
            else f"{v:,.{d}f}"
    lineas = ["| brazo | n | media | mediana | sd | min | max | AUC | s/curvas |",
              "|---|---|---|---|---|---|---|---|---|"]
    for f_ in filas:
        lineas.append(
            f"| {f_['arm']} | {f_['n']} | {_f(f_['mean'])} | {_f(f_['median'])} "
            f"| {_f(f_['std'])} | {_f(f_['min'])} | {_f(f_['max'])} "
            f"| {_f(f_['auc_mean'])} | {f_['runs_sin_curvas']} |")
    lineas += ["", "| comparacion | U | p | r_rb | signif. | aviso |",
               "|---|---|---|---|---|---|"]
    for p_ in pares:
        lineas.append(
            f"| {p_['comparacion']} | {p_['U']:.0f} | {p_['p_valor']:.2e} "
            f"| {p_['r_biserial']:+.3f} | {'si' if p_['significativo_0.05'] else 'no'} "
            f"| {'n bajo' if p_['aviso_n_bajo'] else ''} |")
    return "\n".join(lineas) + "\n"


def run_compare(arms: list[ArmData], out_dir: str,
                ref: str | None = "bo_prior",
                threshold: float | None = None) -> dict:
    """Escribe arms_summary.csv, pairwise.csv, arms_table.md (+ boxplot si
    hay matplotlib) y devuelve el dict resumen. `ref` = brazo objetivo contra
    el que se comparan los demas; si no esta (o None), todos los pares."""
    os.makedirs(out_dir, exist_ok=True)
    if len(arms) < 2:
        raise ValueError("hace falta al menos 2 brazos para comparar")
    filas = [resumen_brazo(a, threshold) for a in arms]

    nombres = [a.name for a in arms]
    if ref and ref in nombres:
        otros = [n for n in nombres if n != ref]
        pares = []
        for n in otros:
            ax = arms[nombres.index(ref)]
            ay = arms[nombres.index(n)]
            pares.append(pairwise(ax.y, ay.y, ref, n))
    else:
        pares = []
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                pares.append(pairwise(arms[i].y, arms[j].y,
                                      arms[i].name, arms[j].name))

    def _csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _csv(os.path.join(out_dir, "arms_summary.csv"), filas)
    _csv(os.path.join(out_dir, "pairwise.csv"), pares)
    md = tabla_markdown(filas, pares)
    with open(os.path.join(out_dir, "arms_table.md"), "w") as f:
        f.write(md)

    png = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot([a.y for a in arms], tick_labels=nombres)
        ax.set_ylabel("NPV final")
        ax.set_title("6 brazos (mismo presupuesto de simulaciones)")
        fig.autofmt_xdate(rotation=30)
        png = os.path.join(out_dir, "arms_boxplot.png")
        fig.tight_layout()
        fig.savefig(png, dpi=140)
        plt.close(fig)
    except ImportError:
        pass                                 # sin matplotlib: tablas igual

    return {"filas": filas, "pares": pares, "out_dir": out_dir,
            "png": png, "markdown": md}
