"""Input/order regressions ported from PR #2, using the standard-library runner.

Run: python3 -m unittest discover -s sim -p test_event_inputs.py -v
"""
import dataclasses
import itertools
import math
import random
import unittest

from reliability_sim import Config, Sim


class EventInputsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(nodes=64, horizon_h=24, spare_nodes=3, detect_h=0,
                          reload_h=1, repair_h=2, requal_h=1, reshard_h=1,
                          single_rate_per_node_h=0)

    def test_sorted_original_result(self):
        cfg = Config(nodes=64, horizon_h=24., seed=20260922,
                     single_rate_per_node_h=0., spare_nodes=0)
        self.assertEqual(Sim(cfg, "auto-restart", [(1., 1), (8., 1)]).run(), {
            "policy": "auto-restart", "seed": 20260922, "ettr": .5965,
            "availability": .6431, "mttf_h": 7.72, "mttr_h": 4.28,
            "interruptions": 2, "reserved_spare_gpu_h": 0.,
            "productive_gpu_h": 7330.1, "waste_gpu_h": 4957.9,
            "waste_dollars": 12395.,
        })

    def test_order_reversal(self):
        events = [(1., 1), (8., 1)]
        for policy in Sim.POLICIES:
            with self.subTest(policy=policy):
                self.assertEqual(Sim(self.cfg, policy, events).run(),
                                 Sim(self.cfg, policy, events[::-1]).run())

    def test_seeded_permutations_with_completion_ties(self):
        # t=2 ties recovery/reshard completion; t=4 ties spare returns.
        events = [(1., 1), (1., 3), (2., 2), (4., 1), (4., 2), (8., 1)]
        for policy, seed in itertools.product(Sim.POLICIES, (0, 7, 20260929)):
            expected = Sim(self.cfg, policy, events).run()
            rng = random.Random(seed)
            for index in range(32):
                shuffled = events.copy()
                rng.shuffle(shuffled)
                with self.subTest(policy=policy, seed=seed, shuffle=index):
                    self.assertEqual(Sim(self.cfg, policy, shuffled).run(), expected)

    def test_equal_time_fault_permutations_preserve_duplicates(self):
        events = [(1., 2), (1., 3), (1., 3)]
        for policy in Sim.POLICIES:
            expected = Sim(self.cfg, policy, events + [(2., 1), (4., 3)]).run()
            for permutation in itertools.permutations(events):
                with self.subTest(policy=policy, events=permutation):
                    sim = Sim(self.cfg, policy, list(permutation) + [(2., 1), (4., 3)])
                    self.assertEqual(tuple(sim.events).count((1., 3)), 2)
                    self.assertEqual(sim.run(), expected)

    def test_valid_boundaries_are_finite_and_bounded(self):
        before_end = math.nextafter(24., 0.)
        cases = [[], [(0., 1)], [(before_end, 1)], [(before_end, 1), (0., 1)],
                 [(1., 2), (1., 1)], [(0., 2), (0., 1)],
                 [(before_end, 2), (before_end, 1)],
                 [(math.nextafter(1., math.inf), 1), (1., 1)]]
        for policy, events in itertools.product(Sim.POLICIES, cases):
            with self.subTest(policy=policy, events=events):
                sim = Sim(self.cfg, policy, events)
                self.assertEqual(list(sim.events), sorted(events))
                result = sim.run()
                for value in result.values():
                    if isinstance(value, (int, float)):
                        self.assertTrue(math.isfinite(value) and value >= 0)
                self.assertLessEqual(result["ettr"], 1)
                self.assertLessEqual(result["availability"], 1)

    def test_caller_list_and_rows_are_snapshotted_without_mutation(self):
        for policy in Sim.POLICIES:
            with self.subTest(policy=policy):
                events = [[8., 1], [1., 1]]
                sim = Sim(self.cfg, policy, events)
                self.assertEqual(events, [[8., 1], [1., 1]])
                expected = sim.run()
                events.reverse()
                events[0][0] = 20.
                events.append([math.nan, 0])
                self.assertEqual(sim.run(), expected)

    def test_invalid_times_refused_at_construction_for_every_policy(self):
        for policy, time in itertools.product(Sim.POLICIES, (
                math.nan, math.inf, -math.inf, -1., 24., 25.,
                None, "1.0", 10**1000)):
            with self.subTest(policy=policy, time=time):
                with self.assertRaisesRegex(ValueError, "event time"):
                    Sim(self.cfg, policy, [(time, 1)])

    def test_invalid_failed_node_counts_refused_for_every_policy(self):
        for policy, count in itertools.product(Sim.POLICIES, (
                0, -1, 1.5, True, False, math.nan, math.inf, -math.inf, None, "1")):
            with self.subTest(policy=policy, count=count):
                with self.assertRaisesRegex(ValueError, "failed-node count"):
                    Sim(self.cfg, policy, [(1., count)])

    def test_spare_return_precedes_fault(self):
        cfg = dataclasses.replace(self.cfg, spare_nodes=1, repair_h=2, requal_h=0)
        result = Sim(cfg, "auto-spares", [(3., 1), (1., 1)]).run()
        self.assertEqual(result["availability"], round(22 / 24, 4))
        self.assertEqual(result["interruptions"], 2)

    def test_recovery_completion_precedes_fault(self):
        cfg = dataclasses.replace(self.cfg, spare_nodes=2)
        result = Sim(cfg, "auto-spares", [(2., 1), (1., 1)]).run()
        self.assertEqual(result["availability"], round(22 / 24, 4))
        self.assertEqual(result["interruptions"], 2)

    def test_elastic_credits_repair_intervals_and_completion_ties(self):
        cfg = Config(nodes=8, horizon_h=10, legal_shape_nodes=1, repair_h=2,
                     requal_h=0, reshard_h=0, ckpt_overlap_interval_h=1,
                     ckpt_overlap_write_h=0)
        for events in ([(5., 1), (1., 1)], [(3., 1), (1., 1)]):
            with self.subTest(events=events):
                # 80 node-hours minus four in repair and two four-node-hour losses.
                self.assertEqual(Sim(cfg, "elastic", events).run()["productive_gpu_h"], 68 * 8)


if __name__ == "__main__":
    unittest.main()
