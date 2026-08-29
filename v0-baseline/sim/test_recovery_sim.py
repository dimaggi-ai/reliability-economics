"""Recovery-sim invariant tests. Run: python3 test_recovery_sim.py (seconds)."""

import math

from recovery_sim import COMPONENTS, Config, POLICIES, Simulation


def _run(policy, **kw):
    return Simulation(Config(**kw), policy).run()


def test_conservation():
    """envelope == productive + every waste component, all policies."""
    for policy in POLICIES:
        r = _run(policy, seed=3)
        total = r["productive_gpu_h"] + sum(r[f"waste_{k}"] for k in COMPONENTS)
        assert math.isclose(total, r["envelope_gpu_h"], rel_tol=1e-6), (policy, total)


def test_determinism():
    a = _run("auto-spares", seed=9)
    b = _run("auto-spares", seed=9)
    assert a == b
    assert _run("auto-spares", seed=9) != _run("auto-spares", seed=10)


def test_event_rate_matches_calibration():
    """~419/54d at 16,384 GPUs => ~233 events in 30 days (fewer: downtime
    suppresses the clock; more for fast-recovery policies)."""
    r = _run("auto-spares-tuned", seed=1)
    assert 170 <= r["events_total"] <= 300, r["events_total"]


def test_policy_ordering():
    """The structural ordering the repo claims, on identical seeds."""
    rs = {p: _run(p, seed=42) for p in POLICIES}
    # automation beats paging a human
    assert rs["auto-restart"]["realization"] > rs["manual-ops"]["realization"]
    # spares beat waiting out repairs at gang scale
    assert rs["auto-spares"]["realization"] > rs["auto-restart"]["realization"]
    # tuned detection beats default clocks
    assert rs["auto-spares-tuned"]["realization"] > rs["auto-spares"]["realization"]
    # elastic also beats waiting for repair
    assert rs["elastic-shrink"]["realization"] > rs["auto-restart"]["realization"]


def test_component_attribution():
    rs = {p: _run(p, seed=7) for p in POLICIES}
    # only spare policies pay spare_idle
    for p in POLICIES:
        expect = p in ("auto-spares", "auto-spares-tuned")
        assert (rs[p]["waste_spare_idle"] > 0) == expect, p
    # only no-spare, non-elastic policies wait for repairs
    assert rs["auto-restart"]["waste_repair_wait"] > 0
    assert rs["manual-ops"]["waste_repair_wait"] > 0
    assert rs["auto-spares"]["waste_repair_wait"] == 0
    assert rs["elastic-shrink"]["waste_repair_wait"] == 0
    # only elastic shrinks
    for p in POLICIES:
        expect = p == "elastic-shrink"
        assert (rs[p]["waste_shrink_deficit"] > 0) == expect, p
    # manual pages a human on every event; automated policies only escalate
    assert rs["manual-ops"]["waste_operator_wait"] > rs["auto-spares"]["waste_operator_wait"]
    # automated policies burn capped retries on structural faults
    assert rs["auto-spares"]["waste_failed_retries"] > 0
    assert rs["manual-ops"]["waste_failed_retries"] == 0


def test_tuned_clocks_cut_detection_wait():
    a = _run("auto-spares", seed=5)
    b = _run("auto-spares-tuned", seed=5)
    assert b["waste_detection_wait"] < 0.5 * a["waste_detection_wait"]


def test_checkpoint_interval_drives_lost_work():
    a = _run("auto-spares", seed=6, ckpt_interval_min=15.0)
    b = _run("auto-spares", seed=6, ckpt_interval_min=60.0)
    assert b["waste_lost_work"] > 1.5 * a["waste_lost_work"]


def test_reliability_identities():
    """MTBF = MTTF + MTTR, and Availability = MTTF/(MTTF+MTTR) = MTTF/MTBF.

    Exact on the underlying values; mttf_h/mttr_h/mtbf_h are reported rounded
    to 0.01 h, so recomputing from them carries up to ~2e-3 of rounding error —
    the tolerance covers exactly that, nothing more.
    """
    for policy in POLICIES:
        r = _run(policy, seed=8)
        # each of the three is rounded to 0.01 h independently, so their sum
        # can disagree by up to ~0.015; exact on the underlying values
        assert math.isclose(r["mtbf_h"], r["mttf_h"] + r["mttr_h"], abs_tol=2e-2), policy
        a = r["mttf_h"] / (r["mttf_h"] + r["mttr_h"])
        assert math.isclose(a, r["availability"], abs_tol=5e-3), (policy, a, r["availability"])
        assert math.isclose(r["mttf_h"] / r["mtbf_h"], r["availability"], abs_tol=5e-3), policy
        assert 0.0 < r["availability"] <= 1.0


def test_mttf_is_a_hardware_property():
    """MTTF (mean up-time before failure) is set by the fault process, so it is
    ~constant across policies; MTBF is not (it includes the varying MTTR)."""
    # average over the standard seeds so the ~constant claim is tested on the
    # stable means, not one noisy draw
    seeds = (42, 43, 44)
    mttfs = [sum(_run(p, seed=s)["mttf_h"] for s in seeds) / 3 for p in POLICIES]
    mtbfs = [sum(_run(p, seed=s)["mtbf_h"] for s in seeds) / 3 for p in POLICIES]
    assert max(mttfs) - min(mttfs) < 0.2, mttfs          # MTTF ~flat near ~3 h
    assert all(2.7 < m < 3.4 for m in mttfs), mttfs      # matches 1/lambda
    assert max(mtbfs) - min(mtbfs) > 1.0, mtbfs          # MTBF genuinely varies


def test_mttr_is_the_policy_lever():
    """MTTR must fall sharply from manual to spares; availability tracks it."""
    rs = {p: _run(p, seed=8) for p in POLICIES}
    assert rs["manual-ops"]["mttr_h"] > 5 * rs["auto-spares"]["mttr_h"]
    assert rs["auto-spares"]["availability"] > rs["auto-restart"]["availability"]


def test_availability_exceeds_realization():
    """Being 'up' is necessary but not sufficient for productivity: time-based
    availability is always >= GPU-hour realization for every policy (the gap is
    re-done work plus, for spare policies, idle spares in the denominator)."""
    for policy in POLICIES:
        r = _run(policy, seed=8)
        assert r["availability"] >= r["realization"] - 1e-9, policy


def test_all_horizon_time_accounted_manual():
    """Even the worst policy's identity holds when downtime dominates."""
    r = _run("manual-ops", seed=2, horizon_days=10.0)
    total = r["productive_gpu_h"] + sum(r[f"waste_{k}"] for k in COMPONENTS)
    assert math.isclose(total, r["envelope_gpu_h"], rel_tol=1e-6)
    assert 0.0 < r["realization"] < 1.0


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        globals()[fn]()
        print(f"ok {fn}")
    print("all recovery-sim tests passed")
