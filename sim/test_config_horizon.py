"""Bounded horizon refusal and recovery; no operational maximum is implied."""
import math
from pathlib import Path
import subprocess
import sys
import unittest
from fractions import Fraction
from unittest.mock import Mock, patch

import numpy as np
from reliability_sim import Config, Sim, latent_events, run_all, run_policy


class HorizonTests(unittest.TestCase):
    def invalids(self):
        return [float('nan'), float('inf'), -float('inf'), 0, -1, True,
                False, '1', None, 10 ** 1000, np.bool_(True)]

    def test_constructor_refuses(self):
        for value in self.invalids():
            with self.subTest(value=type(value).__name__), self.assertRaisesRegex(
                    ValueError, 'horizon_h must be a finite positive real'):
                Config(horizon_h=value)

    def test_mutated_entry_points_refuse_before_rng(self):
        for value in self.invalids():
            cfg = Config(horizon_h=1)
            cfg.horizon_h = value
            rng = Mock()
            with self.assertRaises(ValueError):
                latent_events(cfg, rng)
            rng.exponential.assert_not_called()
            for entry in (lambda: run_policy(cfg, 'elastic'), lambda: run_all(cfg)):
                with patch('reliability_sim.np.random.default_rng') as factory:
                    with self.assertRaises(ValueError):
                        entry()
                    factory.assert_not_called()
            with self.assertRaises(ValueError):
                Sim(cfg, 'elastic', [])

    def test_mutation_after_sim_creation_refuses_then_recovers(self):
        for value in self.invalids():
            cfg = Config(horizon_h=1)
            sim = Sim(cfg, 'elastic', [])
            cfg.horizon_h = value
            with self.assertRaises(ValueError):
                sim.run()
            cfg.horizon_h = 1
            self.assertEqual(sim.run(), Sim(Config(horizon_h=1), 'elastic', []).run())

    def test_valid_real_domain_without_arbitrary_cap(self):
        for value in (1, 1.5, Fraction(3, 2), np.float64(1.5), np.int64(1),
                      sys.float_info.max, math.nextafter(0.0, 1.0)):
            self.assertIs(Config(horizon_h=value).horizon_h, value)

    def test_zero_failure_controls(self):
        cfg = Config(horizon_h=1, single_rate_per_node_h=0)
        self.assertEqual(latent_events(cfg, np.random.default_rng(0)), [])
        for policy, result in run_all(cfg).items():
            expected = 0.97 if policy in ('elastic', 'auto-spares-ckpt') else 0.96
            self.assertEqual(result['ettr'], expected)

    def test_seeded_entry_points_agree(self):
        cfg = Config(horizon_h=8, seed=7)
        events = latent_events(cfg, np.random.default_rng(cfg.seed))
        for policy, result in run_all(cfg).items():
            self.assertEqual(result, run_policy(cfg, policy))
            self.assertEqual(result, Sim(cfg, policy, events).run())

    def test_nan_nontermination_regression_is_bounded(self):
        # subprocess.run kills and waits its child if the old unbounded loop returns.
        code = """
from reliability_sim import Config, run_policy
cfg = Config()
cfg.horizon_h = float('nan')
try:
    run_policy(cfg, 'elastic')
except ValueError:
    print('refused')
else:
    raise AssertionError('invalid horizon accepted')
"""
        result = subprocess.run([sys.executable, '-B', '-c', code],
                                cwd=Path(__file__).parent, capture_output=True,
                                text=True, timeout=5, check=True)
        self.assertEqual(result.stdout.strip(), 'refused')


if __name__ == '__main__':
    unittest.main()
