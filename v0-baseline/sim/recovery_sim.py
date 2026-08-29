#!/usr/bin/env python3
"""Event-driven recovery-policy simulator for one gang-scheduled training job.

The question this answers: for a fixed fault load (calibrated to the Llama 3
405B run), how many GPU-hours does each *recovery policy* waste, and where?

Policies compared (the real design space, not straw men):

  manual-ops        every interruption pages a human before restart; no spare
                    nodes, so a node failure also waits out hardware repair.
  auto-restart      automated detect->reload from last checkpoint; still no
                    spares, so node failures wait for repair (the counterfactual
                    that prices spare capacity).
  auto-spares       + a warm spare-node pool: node failures swap instead of
                    waiting; the pool's idle GPUs are charged as waste.
  auto-spares-tuned + tuned detection clocks: desync/hang detection at 90 s
                    (flight-recorder-style) instead of the 600 s watchdog
                    default, and transport-level fabric detection.
  elastic-shrink    no spares: on node failure the job resizes to the surviving
                    nodes and continues narrower until repair returns capacity.

Fault classes and rates are calibrated to public primary sources (see
REFERENCES.md and the class-mix note below):

  - Interruption rate: 419 unexpected interruptions in a 54-day snapshot on
    16,384 H100s (Llama 3 paper, arXiv:2407.21783 S3.3.4)
    => 4.74e-4 interruptions per GPU-day while the job runs.
  - Class mix (grouped from the same paper's root-cause table, informed by
    Meta's fleet finding that most NCCL watchdog timeouts are CPU-side
    desyncs, not slow collectives): node-hardware 0.55, transient-software
    0.20, hang-desync 0.15, fabric-transient 0.10. The grouping is an
    assumption; the sensitivity is exposed via Config.
  - Checkpoint reload: 15 min default (a 504-GPU B200 site measured a 33 min
    median load, arXiv:2605.09370; larger sites report faster paths — swept).
  - Watchdog default: 600 s (PyTorch NCCL default); IB transport gives up in
    roughly 30 s at NCCL defaults (4.096 us * 2^20 * 7 retries).
  - Node repair: 4 h (same assumption as scheduler-vs-more-gpus).

Accounting identity (enforced by tests): over the horizon T,

  envelope = (G + spare_gpus) * T
  envelope = productive + lost_work + detection_wait + reload + repair_wait
             + operator_wait + failed_retries + shrink_deficit + resize_overhead
             + spare_idle

All GPU-hour units conserve exactly. The run also reports the classical
reliability triple with textbook definitions: MTTF (mean operating time before
a failure, ~constant across policies — a property of the fault process), MTTR
(mean stop per failure, the policy lever), and MTBF = MTTF + MTTR (mean time
between successive failures, which varies across policies because it includes
recovery). Availability = MTTF / (MTTF + MTTR) = MTTF / MTBF, the time-based
uptime share. Availability exceeds capacity realization for every policy:
being *up* is necessary but not sufficient for productive work (the gap is
mostly re-done work since the last checkpoint, plus idle spares in the
realization denominator for the spare policies).

Deterministic per seed. Absolute numbers are site-specific by construction;
the finding is the ordering and the sensitivity curves, not any single cell.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

NODE_GPUS = 8

POLICIES = (
    "manual-ops",
    "auto-restart",
    "auto-spares",
    "auto-spares-tuned",
    "elastic-shrink",
)

# waste components, in the order they appear in reports
COMPONENTS = (
    "lost_work",        # progress since last checkpoint, discarded on restart
    "detection_wait",   # fleet frozen until the failure is even noticed
    "reload",           # checkpoint restore + rendezvous, fleet held idle
    "repair_wait",      # whole gang idle while one node is repaired (no spares)
    "operator_wait",    # human response time before recovery starts
    "failed_retries",   # automated retries burned on structural faults
    "shrink_deficit",   # capacity gap while running narrower than full size
    "resize_overhead",  # elastic resize/restart cost
    "spare_idle",       # warm spare pool held out of production
)

# The subset of components that represent the gang being fully stopped (down),
# as opposed to running-but-degraded (shrink_deficit) or re-done work
# (lost_work) or idle-but-available capacity (spare_idle). Summed, these give
# the total downtime that MTTR and availability are computed from.
DOWNTIME_COMPONENTS = (
    "detection_wait",
    "reload",
    "repair_wait",
    "operator_wait",
    "failed_retries",
    "resize_overhead",
)


@dataclasses.dataclass
class Config:
    gpus: int = 16_384                  # one gang job at Llama 3 scale
    horizon_days: float = 30.0
    seed: int = 42
    # fault load
    interruptions_per_gpu_day: float = 419.0 / (54.0 * 16_384.0)
    class_mix: tuple = (
        ("node-hardware", 0.55),
        ("transient-software", 0.20),
        ("hang-desync", 0.15),
        ("fabric-transient", 0.10),
    )
    structural_fraction: float = 0.25   # of transient-software: restart cannot fix
    structural_retries: int = 2         # auto attempts burned before escalation
    # timers (minutes)
    ckpt_interval_min: float = 30.0
    reload_min: float = 15.0
    crash_detect_min: float = 2.0
    hang_watchdog_min: float = 10.0     # PyTorch NCCL default (600 s)
    fabric_detect_min: float = 1.0
    tuned_hang_detect_min: float = 1.5  # flight-recorder-style desync detection
    tuned_fabric_detect_min: float = 0.5
    tuned_crash_detect_min: float = 1.0
    operator_response_min: float = 60.0
    repair_min: float = 240.0
    resize_overhead_min: float = 5.0
    spare_nodes: int = 2

    @property
    def nodes(self) -> int:
        return self.gpus // NODE_GPUS

    @property
    def horizon_h(self) -> float:
        return self.horizon_days * 24.0


class Simulation:
    """One gang job of cfg.gpus GPUs run for the horizon under one policy."""

    def __init__(self, cfg: Config, policy: str):
        assert policy in POLICIES, policy
        probs = [p for _, p in cfg.class_mix]
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-9), cfg.class_mix
        self.cfg = cfg
        self.policy = policy
        self.rng = np.random.default_rng(cfg.seed)
        self.classes = [c for c, _ in cfg.class_mix]
        self.probs = probs
        self.t = 0.0                       # hours
        self.last_restart_t = 0.0          # hours; bounds work-since-checkpoint
        self.productive = 0.0              # GPU-h
        self.waste = {k: 0.0 for k in COMPONENTS}
        self.events = {c: 0 for c in self.classes}
        self.escalations = 0
        # elastic-shrink state: repair-completion times of down nodes
        self.down_until: list[float] = []
        self.spare_exhausted = 0

    # ---------------------------------------------------------------- timers

    def _detect_min(self, cls: str) -> float:
        c = self.cfg
        tuned = self.policy == "auto-spares-tuned"
        if cls == "hang-desync":
            return c.tuned_hang_detect_min if tuned else c.hang_watchdog_min
        if cls == "fabric-transient":
            return c.tuned_fabric_detect_min if tuned else c.fabric_detect_min
        return c.tuned_crash_detect_min if tuned else c.crash_detect_min

    # ---------------------------------------------------------------- helpers

    def _active_gpus(self) -> int:
        if self.policy != "elastic-shrink":
            return self.cfg.gpus
        self.down_until = [e for e in self.down_until if e > self.t]
        return self.cfg.gpus - NODE_GPUS * len(self.down_until)

    def _advance_running(self, dt: float) -> None:
        """Run the job for dt hours, crediting productive GPU-h.

        Under elastic-shrink the allocation changes as repairs complete, so
        integrate piecewise across the repair-completion times inside dt.
        """
        end = self.t + dt
        if self.policy == "elastic-shrink":
            while True:
                self.down_until = [e for e in self.down_until if e > self.t]
                nxt = min([e for e in self.down_until if e < end], default=end)
                g = self.cfg.gpus - NODE_GPUS * len(self.down_until)
                self.productive += (nxt - self.t) * g
                self.waste["shrink_deficit"] += (nxt - self.t) * (self.cfg.gpus - g)
                self.t = nxt
                if self.t >= end:
                    break
        else:
            self.productive += dt * self.cfg.gpus
            self.t = end

    def _downtime(self, hours: float, component: str) -> None:
        """Whole gang idle for `hours`; charge it to one waste component."""
        self.waste[component] += hours * self.cfg.gpus
        self.t += hours

    # ---------------------------------------------------------------- events

    def _handle(self, cls: str) -> None:
        c = self.cfg
        self.events[cls] += 1
        m = 1.0 / 60.0  # minutes -> hours

        # Work since the last checkpoint is lost on any restart. It can be at
        # most one checkpoint interval, and never more than has actually
        # accrued since this job last (re)started — the cap keeps lost work
        # from exceeding accrued productive work when faults arrive faster
        # than the checkpoint cadence.
        elapsed_since_restart = self.t - self.last_restart_t
        lost_h = min(
            float(self.rng.uniform(0.0, c.ckpt_interval_min * m)),
            elapsed_since_restart,
        )
        g_at_fault = self._active_gpus()
        self.productive -= lost_h * g_at_fault
        self.waste["lost_work"] += lost_h * g_at_fault

        self._downtime(self._detect_min(cls) * m, "detection_wait")

        if cls == "node-hardware":
            self._handle_hardware()
        elif cls == "transient-software":
            self._handle_transient()
        else:  # hang-desync, fabric-transient: restart clears it
            self._maybe_operator()
            self._downtime(c.reload_min * m, "reload")

        # the job resumes here; the checkpoint clock restarts
        self.last_restart_t = self.t

    def _maybe_operator(self) -> None:
        if self.policy == "manual-ops":
            self._downtime(self.cfg.operator_response_min / 60.0, "operator_wait")

    def _handle_hardware(self) -> None:
        c = self.cfg
        m = 1.0 / 60.0
        if self.policy in ("auto-spares", "auto-spares-tuned"):
            # swap the failed node for a warm spare; pool assumed sufficient
            # (checked post-hoc via expected occupancy; see run.py note)
            self._downtime(c.reload_min * m, "reload")
        elif self.policy == "elastic-shrink":
            self.down_until.append(self.t + c.repair_min * m)
            self._downtime(c.resize_overhead_min * m, "resize_overhead")
            self._downtime(c.reload_min * m, "reload")
        elif self.policy == "manual-ops":
            # operator responds while repair is under way; job restarts when
            # the node is back (no spares)
            wait = max(c.repair_min, c.operator_response_min) * m
            self._downtime(wait, "repair_wait")
            self._downtime(c.reload_min * m, "reload")
        else:  # auto-restart, no spares: nothing to do but wait for repair
            self._downtime(c.repair_min * m, "repair_wait")
            self._downtime(c.reload_min * m, "reload")

    def _handle_transient(self) -> None:
        c = self.cfg
        m = 1.0 / 60.0
        structural = bool(self.rng.random() < c.structural_fraction)
        if not structural:
            self._maybe_operator()
            self._downtime(c.reload_min * m, "reload")
            return
        # structural: restarts cannot fix it
        if self.policy == "manual-ops":
            self._downtime(c.operator_response_min * m, "operator_wait")
            self._downtime(c.reload_min * m, "reload")
        else:
            # automation burns capped retries, then escalates to a human
            burned = c.structural_retries * (self._detect_min("transient-software") + c.reload_min)
            self._downtime(burned * m, "failed_retries")
            self._downtime(c.operator_response_min * m, "operator_wait")
            self._downtime(c.reload_min * m, "reload")
        self.escalations += 1

    # ---------------------------------------------------------------- run

    def run(self) -> dict:
        c = self.cfg
        lam_per_h = c.interruptions_per_gpu_day * c.gpus / 24.0
        while self.t < c.horizon_h:
            dt = float(self.rng.exponential(1.0 / lam_per_h))
            if self.t + dt >= c.horizon_h:
                self._advance_running(c.horizon_h - self.t)
                break
            self._advance_running(dt)
            if self.t >= c.horizon_h:
                break
            cls = self.classes[int(self.rng.choice(len(self.classes), p=self.probs))]
            self._handle(cls)
        # clamp: a downtime tail may overrun the horizon; trim proportionally
        overrun = self.t - c.horizon_h
        if overrun > 1e-9:
            # a downtime tail overran the horizon; scale ALL waste components
            # down uniformly so the envelope identity holds exactly. This
            # keeps totals exact at a small cost in per-component attribution
            # (sub-percent; the last event's downtime is spread across all
            # components rather than trimmed from itself).
            total_waste = sum(self.waste.values())
            if total_waste > 0:
                scale = overrun * c.gpus / total_waste
                for k in self.waste:
                    self.waste[k] *= 1.0 - scale
            self.t = c.horizon_h

        spare_gpus = (
            c.spare_nodes * NODE_GPUS
            if self.policy in ("auto-spares", "auto-spares-tuned")
            else 0
        )
        self.waste["spare_idle"] = spare_gpus * c.horizon_h
        envelope = (c.gpus + spare_gpus) * c.horizon_h
        wasted = sum(self.waste.values())
        assert self.productive >= -1e-6, "lost-work cap breached; productive < 0"

        # --- reliability metrics (time-based, at the job's own GPU count) ----
        # Downtime hours = the fully-stopped waste, converted from GPU-h back to
        # wall-clock hours (each stop was charged at cfg.gpus). Uptime is the
        # rest of the horizon (running, including narrower under elastic shrink).
        ev = sum(self.events.values())
        downtime_gpu_h = sum(self.waste[k] for k in DOWNTIME_COMPONENTS)
        downtime_h = downtime_gpu_h / c.gpus
        uptime_h = max(0.0, c.horizon_h - downtime_h)
        # Reliability triple, textbook definitions:
        #   MTTF = mean operating time before a failure (mean up-time). This is
        #     ~1/lambda, a property of the fault process — ~constant across
        #     policies (recovery cannot change how often hardware faults).
        #   MTTR = mean stop per failure (the policy lever).
        #   MTBF = MTTF + MTTR = mean time between successive failures. Because
        #     it includes recovery, MTBF *does* vary across policies.
        #   Availability = MTTF / MTBF = MTTF / (MTTF + MTTR) = uptime share
        #     ( = 1 - MTTR/MTBF, so MTBF and MTTR jointly determine it).
        mttf_h = uptime_h / ev if ev else float("inf")
        mttr_h = downtime_h / ev if ev else 0.0
        mtbf_h = mttf_h + mttr_h
        availability = uptime_h / c.horizon_h

        out = {
            "policy": self.policy,
            "seed": c.seed,
            "envelope_gpu_h": round(envelope, 1),
            "productive_gpu_h": round(self.productive, 1),
            "wasted_gpu_h": round(wasted, 1),
            "realization": round(self.productive / envelope, 4),
            "mttf_h": round(mttf_h, 2),
            "mttr_h": round(mttr_h, 2),
            "mtbf_h": round(mtbf_h, 2),
            "availability": round(availability, 4),
            "events_total": ev,
            "escalations": self.escalations,
        }
        for k in COMPONENTS:
            out[f"waste_{k}"] = round(self.waste[k], 1)
        for cls in self.classes:
            out[f"events_{cls}"] = self.events[cls]
        return out


def run_once(policy: str, **cfg_kwargs) -> dict:
    return Simulation(Config(**cfg_kwargs), policy).run()
