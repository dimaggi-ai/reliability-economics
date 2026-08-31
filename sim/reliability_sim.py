#!/usr/bin/env python3
"""Reliability-economics simulator (v1) — which recovery policy wins where.

Rebuilt from the v0 baseline to answer the five-reviewer critique head-on. The
design choices that matter (see ../BUILD_SPEC.md):

  1. SHARED LATENT EVENTS, not policy-conditioned interruptions. Hardware faults
     occur on one absolute wall-clock timeline generated per seed; every policy
     faces the *same* latent faults and differs only in how it detects and
     recovers. This kills the "419 interruptions are what Meta's automation
     already let through" circularity — faults are the input, interruptions are
     an output.
  2. CORRELATED failures. A base rate of single-node faults PLUS rarer
     common-cause bursts (rack/PDU/switch) that fail a blast radius of nodes at
     once. Poisson single-node is the b=1 special case, not the whole model.
  3. SPARES AS A FINITE QUEUE with three clocks: swap-in-and-reload, repair,
     requalification. A used spare is unavailable until repair+requal complete
     (return-to-pool is their sum, not a fourth clock) — the pool can be
     exhausted. Bringing a warm spare in is folded into reload_h rather than
     charged separately; that is the optimistic reading, and ETTR is sensitive
     to it (doubling reload_h costs ~6 points), so sim/validation.py reports
     that sensitivity next to the ETTR anchor.
  4. CHECKPOINT is a policy lever with a WRITE COST: interval c and per-write
     stall w. Lost work per interruption ~ progress since last checkpoint;
     frequent checkpointing trades lost work for write overhead (Young/Daly).
  5. ELASTIC-SHRINK is not free: a reshard/communicator-rebuild stop per event
     and a legal-shape tax (you shrink to whole data-parallel replicas of G
     nodes), so usable capacity falls to the next legal shape, not by one node.

Metrics are the field's: ETTR / goodput (productive / envelope), MTTF/MTTR/
availability as diagnostics, and a $ layer ($/GPU-hour) so spare idle is a cost.

Deterministic per seed. Honest scope: one gang TRAINING job; deterministic
blast radius per burst; point repair/requal/reshard times; progress linear in
usable width above the legal-shape floor. The FINDING is the phase structure
(which policy wins in which regime), validated against public anchors in
validate_anchors(); absolute cell values are site-specific.
"""
from __future__ import annotations
import dataclasses
import heapq
import math
import numpy as np

NODE_GPUS = 8


@dataclasses.dataclass
class Config:
    # cluster / horizon
    nodes: int = 2048                     # 16,384 GPUs
    horizon_h: float = 720.0              # 30 days
    seed: int = 0
    dollars_per_gpu_h: float = 2.5
    # failure process (absolute wall-clock, shared across policies)
    single_rate_per_node_h: float = 1.58e-4   # ~1 node fault / 3.1 h at 2048 nodes
    burst_rate_per_h: float = 0.0             # common-cause events per hour (0 = independent)
    burst_blast: int = 1                      # nodes lost per burst event
    # recovery timing (hours)
    detect_h: float = 0.033               # ~2 min: a TUNED detection posture
                                          # (~30 s NCCL transport timeout, not
                                          # the 600 s watchdog) [7]. Not swept;
                                          # the manual policy uses operator_h.
    reload_h: float = 0.25                # warm-spare swap-in + checkpoint
                                          # reload + rendezvous, together
    repair_h: float = 4.0                 # physical repair of a failed node
    requal_h: float = 1.0                 # burn-in / requalification before reuse
    operator_h: float = 1.0               # human response (manual policy only)
    reshard_h: float = 0.5                # elastic reshard/communicator rebuild
    # checkpoint policy
    ckpt_interval_h: float = 0.5          # blocking-checkpoint interval (default policies)
    ckpt_write_h: float = 0.02            # effective stall per blocking write
    ckpt_overlap_interval_h: float = 0.1  # overlapped-checkpoint interval (ckpt policy)
    ckpt_overlap_write_h: float = 0.003   # effective stall per overlapped write
    # spares / elastic
    spare_nodes: int = 0
    legal_shape_nodes: int = 8            # elastic shrinks by whole G-node replicas

    @property
    def gpus(self) -> int:
        return self.nodes * NODE_GPUS


def latent_events(cfg: Config, rng) -> list[tuple[float, int]]:
    """One absolute-time timeline of hardware faults, shared by all policies.
    Returns sorted (time, n_nodes_failed). Single-node faults + optional bursts,
    with the single-node RATE held fixed so burstiness is isolated from rate:
    a fraction of the single-node budget is re-expressed as bursts when
    burst_rate>0 (so mean node-failures/hour is invariant to blast radius)."""
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


class Sim:
    """Run one policy against a fixed latent-event timeline."""

    POLICIES = ("manual", "auto-restart", "auto-spares", "auto-spares-ckpt", "elastic")

    def __init__(self, cfg: Config, policy: str, events: list[tuple[float, int]]):
        assert policy in self.POLICIES, policy
        self.cfg = cfg
        self.policy = policy
        self.events = events

    def run(self) -> dict:
        c = self.cfg
        # Checkpoint regime: the ckpt-tuned spare policy AND elastic use
        # overlapped (async) checkpointing, so the spares-vs-elastic comparison
        # is not confounded by checkpoint choice; the rest use blocking.
        if self.policy in ("auto-spares-ckpt", "elastic"):
            ckpt, write = c.ckpt_overlap_interval_h, c.ckpt_overlap_write_h
        else:
            ckpt, write = c.ckpt_interval_h, c.ckpt_write_h
        detect = c.operator_h if self.policy == "manual" else c.detect_h
        write_frac = 1.0 - (write / ckpt)

        reserved_spares = c.spare_nodes if self.policy in ("auto-spares", "auto-spares-ckpt") else 0
        spare_avail = reserved_spares
        spare_returns: list[float] = []               # heap of return-to-pool times

        uptime_h = 0.0
        downtime_h = 0.0
        productive_nodeh = 0.0
        n_interruptions = 0

        if self.policy == "elastic":
            # Stays up (minus a reshard stop per event) but runs narrower while
            # nodes are in repair. Downtime = the reshard stops; uptime = the
            # running time (at reduced width). The two tile the horizon.
            down_until: list[float] = []
            t = 0.0

            def credit_elastic(t0, t1):
                nonlocal productive_nodeh
                tt = t0
                while tt < t1:
                    active = [d for d in down_until if d > tt]
                    nxt = min([d for d in active if d < t1], default=t1)
                    usable = c.nodes - _legal_loss(len(active), c.legal_shape_nodes)
                    productive_nodeh += (nxt - tt) * usable * write_frac
                    tt = nxt

            for (te, nfail) in self.events:
                if t >= c.horizon_h:
                    break
                down_until = [d for d in down_until if d > te]
                base = max(t, te)
                credit_elastic(t, base)
                uptime_h += base - t
                usable_now = c.nodes - _legal_loss(len([d for d in down_until if d > te]),
                                                   c.legal_shape_nodes)
                productive_nodeh = max(0.0, productive_nodeh - _expected_lost(ckpt) * usable_now)
                n_interruptions += 1
                for _ in range(nfail):
                    down_until.append(te + c.repair_h + c.requal_h)
                downtime_h += min(base + c.reshard_h, c.horizon_h) - base   # clamp to horizon
                t = base + c.reshard_h
            if t < c.horizon_h:
                credit_elastic(t, c.horizon_h)
                uptime_h += c.horizon_h - t
        else:
            # Fixed-width policies: the job is up except during outage intervals;
            # downtime is the UNION of those intervals (parallel repair). uptime
            # is credited over the up-stretches. Both tile the horizon exactly.
            busy_until = 0.0            # job is down until this wall-time
            last_up = 0.0              # start of the current up-stretch
            for (te, nfail) in self.events:
                while spare_returns and spare_returns[0] <= te:
                    heapq.heappop(spare_returns)
                    spare_avail += 1
                covered = spare_avail >= nfail
                if covered:
                    spare_avail -= nfail
                    for _ in range(nfail):
                        heapq.heappush(spare_returns, te + c.repair_h + c.requal_h)
                    outage = detect + c.reload_h    # swap-in folded into reload
                else:
                    outage = detect + c.repair_h + c.reload_h       # wait for repair
                if te >= busy_until:
                    # job was up on [last_up, te): credit it, then interrupt
                    up = te - last_up
                    uptime_h += up
                    productive_nodeh += up * c.nodes * write_frac
                    productive_nodeh = max(0.0, productive_nodeh - _expected_lost(ckpt) * c.nodes)
                    n_interruptions += 1
                    busy_until = te + outage
                    downtime_h += min(busy_until, c.horizon_h) - te      # clamp to horizon
                    last_up = busy_until
                else:
                    # concurrent fault while already down: extend the union
                    new_end = te + outage
                    if new_end > busy_until:
                        downtime_h += min(new_end, c.horizon_h) - min(busy_until, c.horizon_h)
                        busy_until = new_end
                        last_up = busy_until
            if last_up < c.horizon_h:
                up = c.horizon_h - last_up
                uptime_h += up
                productive_nodeh += up * c.nodes * write_frac

        productive_nodeh = max(0.0, productive_nodeh)
        # invariant: uptime + downtime == horizon (the accounting must close)
        assert abs(uptime_h + downtime_h - c.horizon_h) < 1e-6, \
            (self.policy, uptime_h, downtime_h, c.horizon_h)

        # Envelope INCLUDES the reserved spares — a pool you hold out is a cost,
        # so a bigger pool must earn its keep to raise ETTR.
        envelope = (c.nodes + reserved_spares) * NODE_GPUS * c.horizon_h
        productive_gpu_h = productive_nodeh * NODE_GPUS
        ev = max(1, n_interruptions)
        waste_gpu_h = envelope - productive_gpu_h
        return {
            "policy": self.policy, "seed": c.seed,
            "ettr": round(productive_gpu_h / envelope, 4),
            "availability": round(uptime_h / c.horizon_h, 4),
            "mttf_h": round(uptime_h / ev, 2), "mttr_h": round(downtime_h / ev, 2),
            "interruptions": n_interruptions,
            "reserved_spare_gpu_h": round(reserved_spares * NODE_GPUS * c.horizon_h, 1),
            "productive_gpu_h": round(productive_gpu_h, 1),
            "waste_gpu_h": round(waste_gpu_h, 1),
            "waste_dollars": round(waste_gpu_h * c.dollars_per_gpu_h, 0),
        }


def _legal_loss(down_nodes: int, g: int) -> int:
    if down_nodes <= 0:
        return 0
    return int(math.ceil(down_nodes / g) * g)


def _expected_lost(ckpt_interval_h: float) -> float:
    """Expected lost time (node-hours per node) since the last checkpoint = half
    the interval (uniform)."""
    return ckpt_interval_h / 2.0


def _last_ckpt(t: float, ckpt: float) -> float:
    return math.floor(t / ckpt) * ckpt


def run_policy(cfg: Config, policy: str) -> dict:
    rng = np.random.default_rng(cfg.seed)
    events = latent_events(cfg, rng)
    return Sim(cfg, policy, events).run()


def run_all(cfg: Config) -> dict[str, dict]:
    """All policies on the SAME latent timeline (shared latent events)."""
    rng = np.random.default_rng(cfg.seed)
    events = latent_events(cfg, rng)
    return {p: Sim(cfg, p, events).run() for p in Sim.POLICIES}
