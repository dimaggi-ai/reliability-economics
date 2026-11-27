"""Reliability-sim invariants + validation-against-anchors. Run: python3 test_reliability_sim.py"""
import statistics
from decimal import Decimal
from fractions import Fraction
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


def test_elastic_credits_interval_before_retiring_repairs():
    # Reuses the accounting witness from Dmitry's 077eb6f correction.
    c = Config(nodes=8, horizon_h=10, legal_shape_nodes=1, repair_h=2,
               requal_h=0, reshard_h=0, ckpt_overlap_interval_h=1,
               ckpt_overlap_write_h=0)
    for events in ([(1., 1), (5., 1)], [(1., 1), (3., 1)]):
        result = Sim(c, "elastic", events).run()
        # 80 node-hours minus four in repair and two four-node-hour losses.
        assert result["productive_gpu_h"] == 68 * 8, (events, result)


def test_provisioned_configuration_has_high_local_productive_ratio():
    """Internal policy contrast, not reproduction of Meta's different metric."""
    meta = _mean("auto-spares-ckpt", spare_nodes=16)   # right-sized, not over-provisioned
    assert meta > 0.85, meta                      # retained local regression threshold
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
    # Published sweep budget: 0.2 node-failures/h in bursts, below total 0.323584.
    # The former 0.03*16 over-budget fixture is explicitly refused below.
    # Correlated rack bursts with a small pool: elastic wins
    corr = {p: _mean(p, spare_nodes=2, burst_rate_per_h=0.2/16, burst_blast=16)
            for p in ("auto-spares-ckpt", "elastic")}
    assert corr["elastic"] > corr["auto-spares-ckpt"], corr



class _RecordingRng:
    def __init__(self):
        self.scales = []
    def exponential(self, scale):
        self.scales.append(scale)
        return 2.0


def _refuse_before_rng(cfg, field):
    rng = _RecordingRng()
    try:
        latent_events(cfg, rng)
    except ValueError as exc:
        assert field in str(exc), str(exc)
    else:
        raise AssertionError('expected refusal: ' + field)
    assert rng.scales == [], rng.scales


def test_burst_budget_refusal_equality_and_below():
    _refuse_before_rng(Config(nodes=4, single_rate_per_node_h=1, burst_rate_per_h=3, burst_blast=2, horizon_h=1), 'burst_node_rate')
    for burst, expected in ((2, [.5]), (1, [.5, 1.0]), (0, [.25])):
        rng = _RecordingRng()
        assert latent_events(Config(nodes=4, single_rate_per_node_h=1, burst_rate_per_h=burst, burst_blast=2, horizon_h=1), rng) == []
        assert rng.scales == expected, (burst, rng.scales)
    for cfg in (Config(nodes=0), Config(single_rate_per_node_h=0)):
        rng = _RecordingRng()
        assert latent_events(cfg, rng) == []
        assert rng.scales == []


def test_failure_inputs_refused_before_rng_even_after_mutation():
    for name in ('single_rate_per_node_h', 'burst_rate_per_h'):
        for value in (-1, float('nan'), float('inf'), -float('inf'), True, False, np.bool_(False), Decimal('1'), 1+0j, '0', None, 10**1000):
            cfg = Config(); setattr(cfg, name, value)
            _refuse_before_rng(cfg, name)
    for name in ('nodes', 'burst_blast'):
        for value in (-1, 1.5, float('nan'), float('inf'), -float('inf'), True, False, np.bool_(False), Decimal('4'), 4+0j, 4.0, '2', None):
            cfg = Config(); setattr(cfg, name, value)
            _refuse_before_rng(cfg, name)
    _refuse_before_rng(Config(burst_blast=0), 'burst_blast')
    _refuse_before_rng(Config(single_rate_per_node_h=1e308, nodes=100), 'total_node_rate')
    _refuse_before_rng(Config(burst_rate_per_h=1e308, burst_blast=100), 'burst_node_rate')
    cfg=Config();cfg.horizon_h=float('nan');_refuse_before_rng(cfg,'horizon_h')


def test_integral_numpy_types_do_not_wrap_the_budget():
    # 2**62 * 4 would wrap to zero in NumPy int64 multiplication.
    cfg=Config(nodes=np.int64(2**62), single_rate_per_node_h=np.int64(4), horizon_h=1)
    rng=_RecordingRng();assert latent_events(cfg,rng)==[]
    assert rng.scales == [1.0 / (2**64)]
    cfg=Config(nodes=np.int64(4),single_rate_per_node_h=np.float64(1), burst_rate_per_h=np.float64(2),burst_blast=np.int64(2),horizon_h=1)
    rng=_RecordingRng();assert latent_events(cfg,rng)==[];assert rng.scales==[.5]
    for field in ('nodes','burst_blast','single_rate_per_node_h','burst_rate_per_h'):
        cfg=Config();setattr(cfg,field,np.bool_(True));_refuse_before_rng(cfg,field)


def test_public_policy_refusal_and_recovery():
    cfg=Config(spare_nodes=2,burst_rate_per_h=.03,burst_blast=16)
    try:run_policy(cfg,'elastic')
    except ValueError as exc:assert 'burst_node_rate' in str(exc)
    else:raise AssertionError('old over-budget correlated fixture must refuse')
    cfg.burst_rate_per_h=.2/16
    result=run_policy(cfg,'elastic')
    assert 0 < result['ettr'] <= result['availability'] <= 1



# Frozen old generator body from main 1c3ba66d20613c707dc47e64c44ee3f791b92cca.
# Keep independent of the new implementation so draw/order regressions are visible.
def _old_latent_events(cfg, rng):
    evs: list[tuple[float, int]] = []
    # bursts consume part of the node-failure budget; keep total node rate fixed
    total_node_rate = cfg.single_rate_per_node_h * cfg.nodes      # node-failures/h
    burst_node_rate = cfg.burst_rate_per_h * cfg.burst_blast
    single_node_rate = max(0.0, total_node_rate - burst_node_rate)
    # single-node faults (Poisson)
    t = 0.0
    while single_node_rate > 0:
        t += rng.exponential(1.0 / single_node_rate)
        if t >= cfg.horizon_h:
            break
        evs.append((t, 1))
    # correlated bursts (Poisson)
    if cfg.burst_rate_per_h > 0:
        t = 0.0
        while True:
            t += rng.exponential(1.0 / cfg.burst_rate_per_h)
            if t >= cfg.horizon_h:
                break
            evs.append((t, cfg.burst_blast))
    evs.sort()
    return evs


def _assert_old_parity(cfg, seed):
    actual_rng = np.random.default_rng(seed)
    old_rng = np.random.default_rng(seed)
    actual = latent_events(cfg, actual_rng)
    expected = _old_latent_events(cfg, old_rng)
    assert actual == expected, (seed, cfg)
    assert actual_rng.bit_generator.state == old_rng.bit_generator.state
    assert actual_rng.random() == old_rng.random()
    return actual


def test_old_generator_no_burst_timeline_and_rng_parity():
    for seed in (0, 1, 7, 42, 1234):
        events = _assert_old_parity(Config(), seed)
        assert len(events) > 50  # A real timeline, not an empty stub result.


def test_old_generator_exact_horizon_boundary():
    # Make the first real seeded draw land exactly on the horizon.
    # This catches >= becoming > without relying on a chance floating-point tie.
    for seed in (0, 7, 42):
        horizon = np.random.default_rng(seed).exponential(1.0)
        cfg = Config(nodes=1, single_rate_per_node_h=1, horizon_h=horizon)
        assert _assert_old_parity(cfg, seed) == []


def test_old_generator_valid_burst_timeline_and_rng_parity():
    for seed in (0, 1, 7, 42, 1234):
        events = _assert_old_parity(Config(burst_rate_per_h=.2/16, burst_blast=16), seed)
        assert any(n == 16 for _, n in events)
        assert any(n == 1 for _, n in events)


def test_old_generator_supported_scalar_parity():
    for single, burst, nodes, blast in (
        (1, 1, 4, 2),
        (np.float64(1), np.float64(.5), np.int64(4), np.int64(2)),
        (Fraction(1, 3), Fraction(1, 6), 4, 2),
    ):
        for seed in (0, 7, 42):
            _assert_old_parity(Config(single_rate_per_node_h=single, burst_rate_per_h=burst,
                                     nodes=nodes, burst_blast=blast, horizon_h=10), seed)


def test_narrow_numpy_rates_refused_before_rng_and_float64_recovery():
    for kind in (np.float16, np.float32, np.longdouble):
        if np.dtype(kind) == np.dtype('float64'):
            continue
        for field in ('single_rate_per_node_h', 'burst_rate_per_h'):
            cfg = Config(); setattr(cfg, field, kind(0))
            _refuse_before_rng(cfg, field)
            setattr(cfg, field, kind(1))
            _refuse_before_rng(cfg, field)
    for kind, excess in ((np.float16, 4e-4), (np.float32, 5e-8)):
        cfg = Config(nodes=1, single_rate_per_node_h=kind(1),
                     burst_rate_per_h=1+excess, burst_blast=1, horizon_h=1)
        _refuse_before_rng(cfg, 'single_rate_per_node_h')
        cfg.single_rate_per_node_h = np.float64(1)
        _refuse_before_rng(cfg, 'burst_node_rate')
        cfg.burst_rate_per_h = np.float64(.5)
        _assert_old_parity(cfg, 7)


def test_unrepresentable_positive_rates_and_scales_refuse_before_rng():
    for field in ('single_rate_per_node_h', 'burst_rate_per_h'):
        cfg = Config(); setattr(cfg, field, Fraction(1, 10**400))
        _refuse_before_rng(cfg, field)
    # A representable positive rate may still have an infinite reciprocal.
    _refuse_before_rng(Config(nodes=1, single_rate_per_node_h=5e-324), 'single_node_rate')
    _refuse_before_rng(Config(burst_rate_per_h=5e-324), 'burst_rate_per_h')
    # Huge count conversion is deliberately outside the supported range even at zero rate.
    _refuse_before_rng(Config(single_rate_per_node_h=0., nodes=10**400), 'total_node_rate')
    _refuse_before_rng(Config(burst_rate_per_h=0., burst_blast=10**400), 'burst_node_rate')


def test_public_run_all_refusal_and_corrected_recovery():
    cfg = Config(burst_rate_per_h=.03, burst_blast=16)
    try:
        run_all(cfg)
    except ValueError as exc:
        assert 'burst_node_rate' in str(exc)
    else:
        raise AssertionError('public entry point must refuse an over-budget process')
    cfg.burst_rate_per_h = .2/16
    assert run_all(cfg) == run_all(Config(burst_rate_per_h=.2/16, burst_blast=16))


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        globals()[fn]()
        print(f"ok {fn}")
    print("all reliability-sim tests passed")
