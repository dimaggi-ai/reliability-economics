#!/usr/bin/env python3
"""The validation registry: model vs the public record, in one table.

Three kinds of points, honestly separated:

  calibrated  a constant tuned to — or sharing an input with — a
              published figure. Passing proves the constant has not
              drifted, NOT that the model predicts anything.
  emergent    a behavior nothing was tuned to produce: it falls out of
              the mechanism and lands where the public record (or an
              independently derived closed form) says it should. These
              can fail; that is the point.
  sanity      a property of the model's own machinery — true by
              construction, or deterministic arithmetic. Cites no
              external evidence and claims none, so its ref is '-'.

The distinction that matters here: this model's fault RATE is derived
from Meta's 419 interruptions [1]. Any point whose value is carried by
that rate therefore shares an input with its own reference and cannot
be evidence about it, however well it agrees. Those points are labelled
calibrated or sanity, never emergent.

References are to REFERENCES.md. Synthetic data is the seeded
latent-event generator itself; every stochastic point is averaged over
eight seeds so the mechanism, not the seed, carries it.

Anchors this registry deliberately DECLINES to check are listed in
DECLINED below and printed with the table. A registry that reports only
the anchors it passes is a highlight reel, not a validation.

Run: python3 validation.py   (exit 1 if any point fails)
"""
from __future__ import annotations

import dataclasses
import math
import statistics
import sys

import numpy as np

from reliability_sim import Config, latent_events, run_policy

SEEDS = range(8)
LLAMA3_H = 54 * 24.0            # the published 54-day snapshot [1]
LLAMA3_NODES = 2048             # 16,384 H100s / 8 GPUs per server [1]
LLAMA3_COUNT = 419              # unexpected interruptions in that window [1]
LLAMA3_RATE = LLAMA3_COUNT / (LLAMA3_H * LLAMA3_NODES)

# Meta publishes ">90% effective training time" — a LOWER BOUND, not a
# point estimate. The ETTR anchor is therefore one-sided: the model must
# approach the bound without crossing it. The floor is the same 5-point
# width the two-sided band used before, so this is strictly stricter
# than a +/-0.05 tolerance, never looser.
ETTR_BOUND = 0.90
ETTR_FLOOR = 0.85


def ettr_in_band(ettr: float) -> bool:
    return ETTR_FLOOR <= ettr < ETTR_BOUND

# Anchors in the public record that this model does NOT reproduce, and
# why it does not try. Printed with the table.
DECLINED: tuple[tuple[str, str], ...] = (
    ("[2] absolute MTTF (~1.8 h projected at 16,384 GPUs)",
     "this model gives 2.7-3.2 h at that width. [2] measures A100 research "
     "clusters over a different period and job mix than the H100 production "
     "run this model is calibrated to [1]; matching both would mean fitting "
     "one of them. Only the SHAPE of [2]'s scaling is used, never the level."),
    ("[3] ByteRobust's <1% checkpoint overhead",
     "this model's overlapped write is 0.003 h per 0.1 h interval = 3%. It "
     "represents checkpoint cadence as an economic LEVER, not per-step "
     "in-memory checkpointing at that implementation's fidelity, so it would "
     "miss this anchor by 3x."),
)


@dataclasses.dataclass(frozen=True)
class Point:
    name: str
    kind: str        # 'calibrated' | 'emergent' | 'sanity'
    ref: str         # '-' for sanity points, which cite nothing
    expected: float
    tolerance: float
    actual: float
    note: str

    @property
    def ok(self) -> bool:
        return abs(self.actual - self.expected) <= self.tolerance


def _ettr(**overrides) -> float:
    """Mean ETTR of the automation-class policy over the seed set."""
    return statistics.mean(
        run_policy(Config(seed=s, horizon_h=LLAMA3_H, spare_nodes=16,
                          **overrides), "auto-spares-ckpt")["ettr"]
        for s in SEEDS
    )


def _surfaced(policy: str, **overrides) -> float:
    return statistics.mean(
        run_policy(Config(seed=s, horizon_h=LLAMA3_H, **overrides),
                   policy)["interruptions"]
        for s in SEEDS
    )


def _mttf(nodes: int) -> float:
    return statistics.mean(
        run_policy(Config(seed=s, nodes=nodes), "elastic")["mttf_h"]
        for s in SEEDS
    )


def points() -> list[Point]:
    pts: list[Point] = []

    # ------------------------------------------------------------ calibrated
    pts.append(Point(
        "llama3-fault-rate", "calibrated", "[1]",
        expected=LLAMA3_RATE * 1e4, tolerance=0.01,
        actual=Config().single_rate_per_node_h * 1e4,
        note="Default single-node fault rate (x1e4 /node-h) vs the rate "
             "implied by 419 interruptions / 54 days / 2,048 nodes [1]. "
             "Tuned to it; pinned so it cannot drift. Two simplifications "
             "both push this rate DOWN (and so ETTR up), and a skeptic "
             "should push on them: the job is assumed to hold all 16,384 "
             "GPUs for the full 54 days (the paper says 'up to' 16K), and "
             "every unexpected interruption is counted as one node fault "
             "(the paper attributes ~78% to confirmed hardware).",
    ))

    latent = statistics.mean(
        len(latent_events(Config(seed=s, horizon_h=LLAMA3_H),
                          np.random.default_rng(s)))
        for s in SEEDS
    )
    poisson_sd = math.sqrt(LLAMA3_COUNT)
    pts.append(Point(
        "llama3-latent-fault-count", "calibrated", "[1]",
        expected=float(LLAMA3_COUNT), tolerance=25.0, actual=latent,
        note=f"Mean latent faults at the published exposure. Rate x exposure "
             f"plus Poisson noise — a closure check on the generator, not "
             f"evidence: the rate was derived FROM 419, so this point cannot "
             f"disagree with its own reference by more than sampling noise. "
             f"The +/-25 band is 1.2x the Poisson sd of a 419-count "
             f"({poisson_sd:.1f}), not a number chosen to fit.",
    ))

    # -------------------------------------------------------------- emergent
    ettr = _ettr()
    ettr_slow_detect = _ettr(detect_h=Config().detect_h * 3)
    ettr_slow_reload = _ettr(reload_h=Config().reload_h * 2)
    ettr_no_latency = _ettr(detect_h=0.0, reload_h=0.0)
    pts.append(Point(
        "ettr-approaches-published-bound-from-below", "emergent", "[1]",
        expected=1.0, tolerance=0.0,
        actual=float(ettr_in_band(ettr)),
        note=f"ETTR of the automation-class policy at the published 54-day "
             f"exposure, against Meta's >90% effective training time [1] — a "
             f"MEASURED figure from the production run, not one computed at "
             f"an assumed checkpoint interval. The assertion is DIRECTIONAL, "
             f"{ETTR_FLOOR:.2f} <= ETTR < {ETTR_BOUND:.2f}, and the model "
             f"gives {ettr:.4f}. A two-sided band around 0.90 would have been "
             f"worthless here: >90% is a published LOWER BOUND, so a model "
             f"with zero detection and zero reload — recovery in no time at "
             f"all — scores {ettr_no_latency:.4f} and would have passed it, "
             f"landing nearer the reference than the honest model does. "
             f"Requiring the model to stay UNDER the bound is what makes the "
             f"point falsifiable: it fails if recovery costs are ever quietly "
             f"removed. It is tight in both directions and should be. The "
             f"clocks that carry the number are detect_h=0.033 h (a TUNED "
             f"detection posture — the ~30 s NCCL transport timeout rather "
             f"than the 600 s watchdog [7]), reload_h=0.25 h, and the "
             f"overlapped-checkpoint pair (0.1 h interval / 0.003 h write). "
             f"None was taken from [1], but none is swept either, and the "
             f"point is sensitive along exactly those axes: tripling "
             f"detection gives {ettr_slow_detect:.4f} and doubling reload "
             f"gives {ettr_slow_reload:.4f} — both outside the band.",
    ))

    def waste(iv: float) -> float:
        return statistics.mean(
            run_policy(Config(seed=s, spare_nodes=16, ckpt_interval_h=iv),
                       "auto-spares")["waste_gpu_h"]
            for s in SEEDS
        )
    # Fine grid (0.025 h steps): the coarse grid the figures use would put
    # the argmin at a quantization boundary and make the band do the work.
    ivs = tuple(round(0.1 + 0.025 * i, 4) for i in range(37))   # 0.1 .. 1.0
    best = min(ivs, key=waste)
    cfg = Config()
    mtbf = 1.0 / (cfg.single_rate_per_node_h * cfg.nodes)
    young = math.sqrt(2 * cfg.ckpt_write_h * mtbf)
    daly = math.sqrt(2 * cfg.ckpt_write_h * (mtbf + cfg.reload_h)) - cfg.ckpt_write_h
    pts.append(Point(
        "young-daly-optimal-interval", "emergent", "[5]",
        expected=1.0, tolerance=0.10, actual=best / young,
        note=f"Ratio of the swept blocking-checkpoint optimum "
             f"({best:.3f} h) to Young's first-order optimum "
             f"sqrt(2wM) = {young:.3f} h [5] — the model was never told the "
             f"formula. sqrt(2wM) is YOUNG's 1974 approximation; Daly's 2006 "
             f"higher-order form sqrt(2w(M+R))-w gives {daly:.3f} h here, "
             f"1.7% lower. The +/-0.10 band is derived, not fitted: half a "
             f"grid step is {0.0125 / young:.1%} of the target and the "
             f"Young-vs-Daly spread is a further "
             f"{1 - daly / young:.1%}, leaving roughly 2x headroom for "
             f"8-seed Monte Carlo noise.",
    ))

    # ---------------------------------------------------------------- sanity
    surfaced = _surfaced("auto-spares-ckpt", spare_nodes=16)
    naive = _surfaced("auto-restart")
    coalesced = 1.0 - surfaced / latent
    pts.append(Point(
        "automation-surfaces-nearly-every-latent-fault", "sanity", "-",
        expected=1.0, tolerance=0.15, actual=surfaced / latent,
        note=f"Share of latent faults the automation-class policy surfaces "
             f"as its own job interruption: {surfaced / latent:.1%} "
             f"({surfaced:.1f} of {latent:.1f}), i.e. it coalesces "
             f"{coalesced:.1%}. This is deliberately NOT offered as "
             f"agreement with "
             f"Meta's 419 [1]: the fault rate is derived from that same 419, "
             f"so the count cannot land far from it and a model with "
             f"instantaneous recovery would agree even better. Only the "
             f"coalescing fraction is a model output. What IS "
             f"policy-conditioned — and is the repo's core point — is the "
             f"contrast with a naive no-spare posture, which surfaces "
             f"{naive:.1f} of the SAME latent faults, "
             f"{surfaced / naive:.1f}x fewer.",
    ))

    widths = (512, 1024, 2048, 4096, 8192)
    mttfs = [_mttf(n) for n in widths]
    ratios = [a / b for a, b in zip(mttfs, mttfs[1:])]
    steeper = sum(1 for r in ratios if r >= 2.0)
    pts.append(Point(
        "mttf-degradation-steepens-past-linear", "sanity", "-",
        expected=float(len(ratios)), tolerance=0.0, actual=float(steeper),
        note="MTTF ratio at each cluster doubling from 512 to 8,192 nodes: "
             + ", ".join(f"{r:.2f}x" for r in ratios) + ". A ~2x ratio is "
             "STRUCTURAL and cannot fail: latent_events sets "
             "total_node_rate = single_rate x nodes, so rate ~ N is an input "
             "of the generator, not a finding. What this point pins is the "
             "EXCESS above 2x — the recovery layer degrading faster than "
             "linearly as a fixed spare pool and repair queue saturate at "
             "width. Cites nothing: the agreement with Meta's ~1/N [2] and "
             "the LANL study [10] is a consistency check on direction, "
             "discussed in docs/study.md, not evidence produced here.",
    ))

    return pts


def validate() -> tuple[list[Point], bool]:
    pts = points()
    return pts, all(p.ok for p in pts)


def main() -> int:
    pts, ok = validate()
    w = max(len(p.name) for p in pts)
    print(f"{'point':<{w}}  {'kind':<10}  {'ref':<5}  {'expected':>9}  "
          f"{'actual':>9}  {'band':>7}  verdict")
    for p in pts:
        band = (f"{p.tolerance / abs(p.expected):>6.1%}"
                if p.expected else f"{p.tolerance:>6.3f}")
        print(f"{p.name:<{w}}  {p.kind:<10}  {p.ref:<5}  {p.expected:>9.4f}  "
              f"{p.actual:>9.4f}  {band}  {'PASS' if p.ok else 'FAIL'}")
    print()
    print("anchors this registry declines to check:")
    for anchor, why in DECLINED:
        print(f"  - {anchor}\n      {why}")
    print()
    if ok:
        print("all points reproduced — calibrated points prove the model has "
              "not drifted from the citations it was fitted to, emergent "
              "points meet the record on axes nothing was tuned to, sanity "
              "points pin the model's own machinery and claim no evidence")
    else:
        print("VALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
