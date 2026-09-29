import math
import pytest
from reliability_sim import Config, Sim


@pytest.fixture
def cfg():
    return Config(nodes=64, horizon_h=24.0, seed=20260922,
                  single_rate_per_node_h=0.0, spare_nodes=0)


def test_sorted_original_result(cfg):
    assert Sim(cfg, 'auto-restart', [(1., 1), (8., 1)]).run() == {
        'policy': 'auto-restart', 'seed': 20260922, 'ettr': 0.5965,
        'availability': 0.6431, 'mttf_h': 7.72, 'mttr_h': 4.28,
        'interruptions': 2, 'reserved_spare_gpu_h': 0.0,
        'productive_gpu_h': 7330.1, 'waste_gpu_h': 4957.9,
        'waste_dollars': 12395.0,
    }


@pytest.mark.parametrize('policy', Sim.POLICIES)
@pytest.mark.parametrize('events', [
    [(8., 1), (1., 1)],
    [(0., 1), (8., 1), (1., 1)],
    [(1., 1), (1., 2), (0., 1)],
    [(math.nextafter(24., 0.), 1), (0., 1)],
    [(math.nextafter(1., math.inf), 1), (1., 1)],
])
def test_decreasing_timestamps_normalized(cfg, policy, events):
    assert Sim(cfg, policy, events).run() == Sim(cfg, policy, sorted(events)).run()


@pytest.mark.parametrize('policy', Sim.POLICIES)
@pytest.mark.parametrize('events', [
    [], [(0., 1)], [(math.nextafter(24., 0.), 1)],
    [(0., 1), (math.nextafter(24., 0.), 1)],
    [(1., 2), (1., 1)], [(1., 1), (1., 2)],
    [(0., 2), (0., 1)],
    [(math.nextafter(24., 0.), 2), (math.nextafter(24., 0.), 1)],
    [(1., 1), (math.nextafter(1., math.inf), 1)],
])
def test_equal_times_and_valid_boundaries(cfg, policy, events):
    sim = Sim(cfg, policy, events)
    assert list(sim.events) == sorted(events)
    result = sim.run()
    assert 0 <= result['ettr'] <= 1
    assert 0 <= result['availability'] <= 1
    assert all(math.isfinite(v) and v >= 0 for v in result.values()
               if isinstance(v, (int, float)))


@pytest.mark.parametrize('policy', Sim.POLICIES)
def test_caller_list_mutation_does_not_reorder_admitted_timeline(cfg, policy):
    events = [(1., 1), (8., 1)]
    sim = Sim(cfg, policy, events)
    expected = sim.run()
    events.reverse()
    assert sim.run() == expected
