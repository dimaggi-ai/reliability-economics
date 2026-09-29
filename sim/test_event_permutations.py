import dataclasses
import itertools
import random

import pytest

from reliability_sim import Config, Sim


@pytest.fixture
def cfg():
    return Config(nodes=64, horizon_h=24, spare_nodes=3, detect_h=0,
                  reload_h=1, repair_h=2, requal_h=1, reshard_h=1,
                  single_rate_per_node_h=0)


@pytest.mark.parametrize('policy', Sim.POLICIES)
def test_order_reversal(cfg, policy):
    events = [(1., 1), (8., 1)]
    assert Sim(cfg, policy, events).run() == Sim(cfg, policy, events[::-1]).run()


@pytest.mark.parametrize('policy', Sim.POLICIES)
@pytest.mark.parametrize('seed', [0, 7, 20260929])
def test_seeded_permutations_with_completion_ties(cfg, policy, seed):
    # t=2 ties recovery/reshard completion; t=4 ties repair/requalification
    # and spare returns. Single faults and bursts also share timestamps.
    events = [(1., 1), (1., 3), (2., 2), (4., 1), (4., 2), (8., 1)]
    expected = Sim(cfg, policy, events).run()
    rng = random.Random(seed)
    for _ in range(32):
        shuffled = events.copy()
        rng.shuffle(shuffled)
        assert Sim(cfg, policy, shuffled).run() == expected


@pytest.mark.parametrize('policy', Sim.POLICIES)
def test_equal_time_fault_permutations(cfg, policy):
    events = [(1., 2), (1., 3), (1., 3)]
    expected = Sim(cfg, policy, events + [(2., 1), (4., 3)]).run()
    for permutation in itertools.permutations(events):
        assert Sim(cfg, policy, list(permutation) + [(2., 1), (4., 3)]).run() == expected


def test_spare_return_precedes_fault(cfg):
    c = dataclasses.replace(cfg, spare_nodes=1, repair_h=2, requal_h=0)
    result = Sim(c, 'auto-spares', [(3., 1), (1., 1)]).run()
    assert result['availability'] == round(22 / 24, 4)
    assert result['interruptions'] == 2


def test_recovery_completion_precedes_fault(cfg):
    c = dataclasses.replace(cfg, spare_nodes=2)
    result = Sim(c, 'auto-spares', [(2., 1), (1., 1)]).run()
    assert result['availability'] == round(22 / 24, 4)
    assert result['interruptions'] == 2


def test_elastic_credits_interval_before_retiring_repairs():
    c = Config(nodes=8, horizon_h=10, legal_shape_nodes=1, repair_h=2,
               requal_h=0, reshard_h=0, ckpt_overlap_interval_h=1,
               ckpt_overlap_write_h=0)
    result = Sim(c, 'elastic', [(1., 1), (5., 1)]).run()
    # 80 node-hours minus four node-hours in repair and two 4-node-hour losses.
    assert result['productive_gpu_h'] == 68 * 8


def test_elastic_repair_completion_precedes_fault():
    c = Config(nodes=8, horizon_h=10, legal_shape_nodes=1, repair_h=2,
               requal_h=0, reshard_h=0, ckpt_overlap_interval_h=1,
               ckpt_overlap_write_h=0)
    result = Sim(c, 'elastic', [(3., 1), (1., 1)]).run()
    assert result['productive_gpu_h'] == 68 * 8


@pytest.mark.parametrize('policy', Sim.POLICIES)
def test_admitted_rows_are_snapshot(cfg, policy):
    events = [[1., 1], [8., 1]]
    sim = Sim(cfg, policy, events)
    expected = sim.run()
    events[0][0] = 20.
    assert sim.run() == expected


@pytest.mark.parametrize('events', [
    [(float('nan'), 1)], [(float('inf'), 1)], [(-1., 1)], [(24., 1)],
    [(1., 0)], [(1., -1)], [(1., 1.5)], [(1., True)],
])
def test_invalid_events_refused(cfg, events):
    with pytest.raises(ValueError, match='event'):
        Sim(cfg, 'auto-restart', events)
