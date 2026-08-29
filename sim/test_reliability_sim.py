"""Reliability-sim invariants + validation-against-anchors. Run: python3 test_reliability_sim.py"""
import statistics
from reliability_sim import Config, Sim, run_all, run_policy, latent_events
import numpy as np

SEEDS = range(8)


def _mean(policy, **kw):
    return statistics.mean(run_policy(Config(seed=s, **kw), policy)["ettr"] for s in SEEDS)


def test_metric_bounds():
    for s in (0, 1, 2):
        for p, v in run_all(Config(seed=s, spare_nodes=4)).items():
            assert 0.0 < v["ettr"] <= v["availability"] + 1e-9, (p, v)
            assert 0.0 < v["availability"] <= 1.0, (p, v)
            assert v["mttf_h"] > 0 and v["mttr_h"] >= 0, (p, v)


def test_determinism_and_shared_events():
    a = run_all(Config(seed=5, spare_nodes=4))
    b = run_all(Config(seed=5, spare_nodes=4))
    assert a == b
    # all policies face the SAME latent timeline (identical for a given seed)
    e1 = latent_events(Config(seed=5), np.random.default_rng(5))
    e2 = latent_events(Config(seed=5), np.random.default_rng(5))
    assert e1 == e2 and len(e1) > 50


def test_seed_changes_results():
    assert run_policy(Config(seed=1, spare_nodes=4), "auto-spares") != \
           run_policy(Config(seed=2, spare_nodes=4), "auto-spares")


def test_dollars_layer():
    v = run_policy(Config(seed=0, spare_nodes=4, dollars_per_gpu_h=3.0), "auto-spares")
    assert abs(v["waste_dollars"] - round(v["waste_gpu_h"] * 3.0)) < 1.0


def test_meta_like_anchor_reproduces_high_ettr():
    """Validation: a Meta-like config — a right-sized spare pool, overlapped
    checkpointing, fast detection — reproduces the >90% effective-training-time
    regime (within model slack). A naive config must not come close."""
    meta = _mean("auto-spares-ckpt", spare_nodes=16)   # right-sized, not over-provisioned
    assert meta > 0.85, meta                      # approaches Meta's >0.90
    naive = _mean("auto-restart", spare_nodes=0)
    assert naive < 0.35, naive                    # a naive config is far below


def test_mttf_scales_inverse_with_nodes():
    """RSC anchor: MTTF falls ~inversely with cluster size (more nodes = more
    faults). Doubling nodes should roughly halve MTTF."""
    m1 = statistics.mean(run_policy(Config(seed=s, nodes=1024, spare_nodes=4), "elastic")["mttf_h"] for s in SEEDS)
    m2 = statistics.mean(run_policy(Config(seed=s, nodes=2048, spare_nodes=4), "elastic")["mttf_h"] for s in SEEDS)
    assert 1.6 < m1 / m2 < 2.5, (m1, m2)


def test_spares_have_a_sweet_spot():
    """Because a reserved pool is charged to the envelope, spares help up to a
    point and then HURT — over-provisioning is waste, not safety."""
    e0 = _mean("auto-spares", spare_nodes=0)
    e8 = _mean("auto-spares", spare_nodes=8)
    e64 = _mean("auto-spares", spare_nodes=64)
    assert e0 < e8                                # an adequate pool helps a lot
    assert e64 < e8                              # ...but over-provisioning is charged and hurts


def test_checkpoint_has_an_interior_optimum():
    """Sweeping the (blocking) checkpoint interval, total waste is U-shaped:
    too frequent = write overhead, too rare = lost work."""
    def waste(iv):
        return statistics.mean(
            run_policy(Config(seed=s, spare_nodes=16, ckpt_interval_h=iv), "auto-spares")["waste_gpu_h"]
            for s in SEEDS)
    ivs = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    w = [waste(iv) for iv in ivs]
    lo = w.index(min(w))
    assert 0 < lo < len(ivs) - 1, list(zip(ivs, w))   # optimum is interior


def test_correlated_failures_flip_the_winner():
    """The phase finding: with independent failures and an adequate spare pool,
    spares win; under correlated bursts larger than the pool, elastic wins."""
    # independent, a right-sized pool: spares+ckpt beats elastic
    ind = {p: _mean(p, spare_nodes=8) for p in ("auto-spares-ckpt", "elastic")}
    assert ind["auto-spares-ckpt"] > ind["elastic"], ind
    # correlated rack bursts with a small pool: elastic wins
    corr = {p: _mean(p, spare_nodes=2, burst_rate_per_h=0.03, burst_blast=16)
            for p in ("auto-spares-ckpt", "elastic")}
    assert corr["elastic"] > corr["auto-spares-ckpt"], corr


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        globals()[fn]()
        print(f"ok {fn}")
    print("all reliability-sim tests passed")
