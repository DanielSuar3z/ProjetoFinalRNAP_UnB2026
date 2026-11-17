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
        # la particula 0 arranca anclada al control base del deck (79.5/395)
        it0 = [r for r in recs if r.iteration == 0]
        assert any(np.allclose(r.controls, p.base_controls) for r in it0)
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


# ------------------------------------------------- T-R2: loader de artefactos BIOPSO

def test_loader_simlog_roundtrip():
    """Fixture spec-v1 -> SimDatabase: conteo, shapes y round-trip en disco."""
    import tempfile
    from resbo.database import SimDatabase
    from resbo.loader_biopso import load_run, make_synthetic_run
    p = _proto()
    sim = MockSimulator(p, seed=42)
    with tempfile.TemporaryDirectory() as td:
        run = make_synthetic_run(os.path.join(td, "random", "run_01"),
                                 sim, p, seed=1, n_sims=12, agents_per_round=4)
        db = SimDatabase(os.path.join(td, "db"))
        n = load_run(run, "random", seed=7, db=db)
        assert n == 12 == len(db.records)
        r0 = db.records[0]
        assert r0.algo == "random" and r0.seed == 7 and r0.iteration == 1
        assert r0.controls.shape == (12,)
        assert len(r0.prod_curve) == p.n_prod * p.horizon
        assert np.any(r0.prod_curve > 0)
        # la etiqueta del harvest debe salir del padre (<root>/random/run_XX)
        from resbo.loader_biopso import load_harvest
        db2 = SimDatabase(os.path.join(td, "db2"))
        load_harvest(td, db2)
        assert {r.algo for r in db2.records} == {"random"}
        # persistencia del CSV del lado resbo
        db.save()
        db3 = SimDatabase.load(os.path.join(td, "db"))
        assert len(db3.records) == 12
        assert np.allclose(db3.records[5].controls, db.records[5].controls)
        assert db3.records[5].npv == db.records[5].npv


def test_loader_rejects_bad_artifacts():
    """Cabecera corrupta y wpr todo-ceros deben fallar ruidosamente (SpecViolation)."""
    import csv as _csv
    import tempfile
    from resbo.database import SimDatabase
    from resbo.loader_biopso import SPEC_HEADER, SpecViolation, load_run
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, "run_01")
        os.makedirs(bad)
        with open(os.path.join(bad, "simlog.csv"), "w") as f:
            f.write("iter,agente,x1\n1,0,1.0\n")
        try:
            load_run(bad, "x", 0, SimDatabase(os.path.join(td, "db1")))
            raise AssertionError("cabecera corrupta no fue rechazada")
        except SpecViolation:
            pass
        # npz con wpr todo-ceros (SUMMARY sin WWPR) -> rechazo (spec §2.5)
        zp = os.path.join(td, "run_02")
        os.makedirs(zp)
        with open(os.path.join(zp, "simlog.csv"), "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(SPEC_HEADER)
            w.writerow([1, 0] + ["1.0"] * 12 + ["1.0"])
        np.savez_compressed(os.path.join(zp, "sim_curves.npz"),
                            wpr=np.zeros((1, 4, 130)),
                            wir=np.ones((1, 8, 130)),
                            time_days=np.arange(130.0),
                            sim_index=np.arange(1),
                            producer_wells=np.array("p1 p2 p3 p4".split()),
                            injector_wells=np.array(f"i0 i1 i2 i3 i4 i5 i6 i7".split()))
        try:
            load_run(zp, "x", 0, SimDatabase(os.path.join(td, "db2")))
            raise AssertionError("wpr todo-ceros no fue rechazado")
        except SpecViolation:
            pass


def test_loader_rescue_run():
    """Ruta B (rescate T-B4): well_curves.csv + best_controls.json -> 1 fila."""
    import tempfile
    from resbo.database import SimDatabase
    from resbo.loader_biopso import load_run, make_synthetic_run
    p = _proto()
    sim = MockSimulator(p, seed=42)
    with tempfile.TemporaryDirectory() as td:
        run = make_synthetic_run(os.path.join(td, "mfo", "run_02"),
                                 sim, p, seed=2, n_sims=8, rescue=True)
        db = SimDatabase(os.path.join(td, "db"))
        n = load_run(run, "mfo", seed=0, db=db)
        assert n == 1
        r = db.records[0]
        assert r.iteration == -1 and r.algo == "mfo"
        assert len(r.prod_curve) == p.n_prod * p.horizon
        assert np.any(r.prod_curve > 0)
        assert np.isfinite(r.npv)


# ------------------------------------------------------ T-R3: persistencia del BNN

def test_bnn_save_load_roundtrip():
    """save_npz/load_npz (allow_pickle=False) reproducen las predicciones base."""
    import tempfile
    from resbo.models.bnn import BnnEnsemble
    rng = np.random.default_rng(0)
    X = rng.random((40, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + 0.1 * rng.normal(size=40)
    m = BnnEnsemble(n_models=2, hidden=(8, 8), seed=0)
    m.fit_norm_stats(X)
    m.fit(X, y, epochs=30)
    base_a = np.array([mod.predict(m._norm(X)) for mod in m.models])
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bnn.npz")
        m.save_npz(path)
        m2 = BnnEnsemble.load_npz(path)
        base_b = np.array([mod.predict(m2._norm(X)) for mod in m2.models])
    assert np.allclose(base_a, base_b)
    assert (m2.x_sd > 0).all()
    mu, sd = m2.predict(X, mc_samples=8)
    assert mu.shape == (40,) and (sd >= 0).all()


def test_crm_npz_roundtrip_and_mask():
    """crm_v1.npz: roundtrip exacto, well_mask persiste, schema ajeno rechazado."""
    import tempfile
    from resbo.prior import load_crm_npz, save_crm_npz
    crm = CRMModel(8, 4)
    rng = np.random.default_rng(0)
    crm.tau = rng.random((8, 4))
    crm.q0 = rng.random(4)
    crm.fit_r2 = np.array([0.9, 0.8, 0.5, 0.2])
    crm.n_runs_used = 123
    mask = (crm.fit_r2 >= 0.7).astype("int8")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "crm.npz")
        save_crm_npz(path, crm, mask, meta={"fuente": "test", "tau_full": crm.tau})
        crm2, mask2, meta2 = load_crm_npz(path)
        assert np.allclose(crm.tau, crm2.tau)
        assert np.allclose(crm.q0, crm2.q0)
        assert np.array_equal(mask, mask2) and mask2.tolist() == [1, 1, 0, 0]
        assert crm2.n_runs_used == 123 and meta2["fuente"] == "test"
        assert "tau_full" not in meta2          # se guardo como array, no en meta
        # columna de pozo fallido en cero -> el score ignora ese pozo
        crm2.tau[:, mask2 == 0] = 0.0
        x = np.r_[np.full(8, 100.0), np.full(4, 300.0)]
        s2 = crm2.score(x)
        crm3, _, _ = load_crm_npz(path)
        crm3.tau[:, 2:] = 0.0
        assert abs(s2 - crm3.score(x)) < 1e-12
        # schema ajeno
        np.savez(path, schema=np.array("crm_v0"))
        try:
            load_crm_npz(path)
            raise AssertionError("debio rechazar schema crm_v0")
        except ValueError:
            pass
