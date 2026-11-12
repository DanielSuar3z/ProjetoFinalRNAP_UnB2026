"""Tests rapidos del nucleo (sin MC2). Ejecutar: python -m pytest tests -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from resbo.analysis import auc, mannwhitney, rank_biserial  # noqa
from resbo.config import Protocol  # noqa
from resbo.prior import CRMModel  # noqa
from resbo.simulator import MockSimulator  # noqa

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "protocol.yaml")


def _proto():
    p = Protocol.load(CFG)
    p.d["budget"]["metaheuristics"] = {"pop": 5, "iters": 2}
    p.d["n_seeds"] = 1
    return p


def test_bounds_shape():
    p = _proto()
    lb, ub = p.bounds
    assert lb.shape == (12,) and ub.shape == (12,)
    assert np.all(lb < ub)


def test_mock_npv_finite():
    p = _proto()
    sim = MockSimulator(p, seed=1)
    lb, ub = p.bounds
    out = sim.evaluate(0.5 * (lb + ub))       # punto central de la caja
    assert np.isfinite(out["npv"])
    assert out["prod_curve"].shape == (p.n_prod, p.horizon)


def test_crm_recovers_connectivity_trend():
    """Con runs suficientemente variados, tau_fit correlaciona con tau_true."""
    p = _proto()
    sim = MockSimulator(p, seed=7)
    rng = np.random.default_rng(0)
    lb, ub = p.bounds
    X = lb + rng.random((40, 12)) * (ub - lb)
    curves = [sim.evaluate(x)["prod_curve"] for x in X]
    crm = CRMModel(p.n_inj, p.n_prod).fit(X, curves)
    r = np.corrcoef(crm.tau.ravel(), sim.tau_true.ravel())[0, 1]
    assert r > 0.4, f"CRM no recuper6 estructura (r={r:.2f})"


def test_mannwhitney_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 1, 20)
    b = rng.normal(0, 1, 20)
    U, p = mannwhitney(a, b)
    assert p < 0.05
    assert rank_biserial(a, b) > 0.5


def test_pso_matches_repo_semantics():
    """PSO (version BIOPSO): w const, ancla en centro, pop*iters sims, max NPV."""
    from resbo.database import SimDatabase
    from resbo.pso import PSO
    import tempfile
    p = _proto()
    sim = MockSimulator(p, seed=3)
    with tempfile.TemporaryDirectory() as td:
        db = SimDatabase(td)
        pso = PSO(p, sim, db, seed=0)
        # defaults del repo
        assert (pso.w, pso.c1, pso.c2) == (0.7, 1.5, 1.5)
        best = pso.run()
        _, y, recs = db.controls_npv(["pso"])
        assert best == y.max()                       # maximiza NPV directo
        # PSO del repo evalua dentro del lazo: pop*iters sims (sin evaluacion
        # inicial separada como MFO/GWO)
        assert len(y) == p["budget"]["metaheuristics"]["pop"] * \
            p["budget"]["metaheuristics"]["iters"]
        # la particula 0 arranca anclada al centro de la caja
        it0 = [r for r in recs if r.iteration == 0]
        lb, ub = p.bounds
        assert any(np.allclose(r.controls, 0.5 * (lb + ub)) for r in it0)
        db.save()


def test_foa_elitism_and_budget():
    """FOA (version BIOPSO): elitismo puro + pop*iters sims + mosca 0 al centro."""
    from resbo.database import SimDatabase
    from resbo.foa import FOA
    import tempfile
    p = _proto()
    sim = MockSimulator(p, seed=5)
    with tempfile.TemporaryDirectory() as td:
        db = SimDatabase(td)
        foa = FOA(p, sim, db, seed=0)
        assert foa.step == 0.1                       # default del repo
        best = foa.run()
        _, y, recs = db.controls_npv(["foa"])
        assert best == y.max()
        budget = p["budget"]["metaheuristics"]
        assert len(y) == budget["pop"] * budget["iters"]
        # elitismo: el best-so-far por ronda nunca baja (mosca 0 = centro)
        per_it = {}
        for r in recs:
            per_it.setdefault(r.iteration, []).append(r.npv)
        bsf = np.maximum.accumulate([max(per_it[i]) for i in sorted(per_it)])
        assert np.all(np.diff(bsf) >= -1e-9)
        db.save()


def test_auc_monotone():
    h = [(i, float(i)) for i in range(1, 6)]
    assert auc(h) > 0
