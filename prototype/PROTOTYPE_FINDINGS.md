# Phase-diagram prototype — findings

Throwaway prototype (`phase_prototype.py`) built 2026-08-29 to test the five reviewers'
central claim before committing to a simulator rebuild: does "a two-node warm-spare pool
is the biggest lever" survive once failures are **correlated** and the spare pool is
modelled as a **finite queue with replenishment delay**?

Metric is ETTR (productive ÷ envelope). Fixed disruptive-event rate (~1 per 3.1 h);
`b` = blast radius (nodes lost per event); warm-spare pool of `k` helps iff `k ≥ b`;
elastic-shrink runs narrower but pays a reshard stop + legal-shape (whole-replica) tax.

## What it shows

**1. The headline is a regime artifact — confirmed.** ETTR(spares) − ETTR(elastic):

| blast radius b | no-spares | spares k=2 | spares k=8 | elastic | winner |
|---|---|---|---|---|---|
| 1 (independent) | 0.374 | 0.671 | 0.791 | 0.735 | spares (k≥4) |
| 2 | 0.374 | 0.543 | 0.783 | 0.735 | spares (k≥4) |
| 4 | 0.374 | 0.374 | 0.671 | 0.735 | **elastic** |
| 8 (rack) | 0.374 | 0.374 | 0.543 | 0.734 | **elastic** |
| 16 (PDU) | 0.374 | 0.374 | 0.374 | 0.731 | **elastic** |

The policy winner is a 2-D *region*, not a single ordering. Spares win only in the
top-left (independent failures **and** a pool sized to the blast radius); elastic wins
the rest by degrading gracefully instead of exhausting.

**2. Two nodes doesn't even win its own best case.** With a realistic replenishment
queue, spares k=2 = 0.671 at b=1 — *below* elastic's 0.735. You need k≈4 for spares to
beat elastic even under independent failures. The original repo's "two nodes sufficient"
was doubly optimistic: it assumed independence **and** an always-full pool.

**3. Checkpoint cadence has a real optimum (~27 min) once write-cost is modelled** —
versus the original repo's monotonic (and misleading) "checkpoint more = always better"
sweep. Confirms checkpoint cadence must be a policy lever, not a frozen constant.

## What this means for the rebuild

- The phase diagram **is** the contribution. "Which recovery policy wins in which failure
  regime" is the reference artifact; a single number ("two nodes, 83%") is not.
- The rebuild must model: a correlated/blast-radius failure process, spares as a finite
  queue with repair-time replenishment, checkpoint cadence + write-cost as a lever, and a
  realistic (reshard + legal-shape) elastic cost. All four are load-bearing — each changes
  the winner.
- The waste ledger, the two-clocks observation, and the fidelity linter are untouched by
  any of this and remain publishable as-is.

## Prototype limitations (do not over-read)

Deterministic blast radius (real bursts are variable + rarer than single-node faults — a
compound process); point repair/reload/reshard times; the elastic reshard (0.5 h) and
legal-shape granularity (8 nodes) are plausible guesses, not measured; no SDC/straggler.
The *shape* is robust to these; the exact cell values are not. This is a motivating
sketch, not a result to publish.
