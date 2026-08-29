#!/usr/bin/env python3
"""PROTOTYPE — reliability-policy phase diagram (throwaway, not the repo).

Purpose: test the reviewers' central claim before committing to a rebuild —
does the "a two-node warm-spare pool is the biggest lever" finding survive
once failures are *correlated* (rack/PDU/switch events fail several nodes at
once), and does checkpoint cadence with a real write-cost behave like a
Young/Daly optimum? If the policy ORDERING flips across regimes, then the
current repo's single-Poisson headline is an artifact and the phase diagram
(which policy wins where) is the real contribution.

Deliberately minimal and honest. Key modelling choices, all documented:
  - Faults arrive in *running* time (the clock only advances while the job is
    up), so heavy downtime self-limits instead of exceeding the horizon.
  - Disruptive-event RATE is held fixed; the correlation axis is b = the
    blast radius (nodes lost per event: b=1 single-node, b=8/16/32 rack/PDU).
    A warm-spare pool of k helps only if k >= b (a gang needs every member);
    a burst larger than the pool degrades spares to the no-spares wait.
  - Fixed-width policies alternate up/down; elastic runs narrower while nodes
    are in repair, but pays a real reshard/restart stop AND a legal-shape tax:
    you cannot drop an arbitrary node count from a 3D-parallel job, so usable
    capacity falls to the next legal shape (whole data-parallel replicas of
    G nodes), and the reshard is not free (RESHARD stop per event).
This is a prototype: deterministic blast radius, point repair/reload times,
one gang job. Enough to show the SHAPE and motivate a rebuild, not to publish.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---- calibration (16,384 GPUs / 8 = 2,048 nodes; Llama-3-ish rate) ----------
N_NODES = 2048
T = 720.0                      # hours (30 days)
BASE_RATE = 0.324              # node-failures per hour at b=1 (~1 per 3.1 h)
R = 4.0                        # repair time, hours
SWAP = 0.25                    # warm-spare swap (reload), hours
RESIZE = 0.083                 # elastic resize stop, hours (~5 min)
CKPT_C = 0.5                   # checkpoint interval, hours (30 min)
CKPT_W = 0.033                 # checkpoint write cost, hours (~2 min)
RESHARD = 0.5                  # elastic reshard/communicator-rebuild stop, hours
G_ELASTIC = 8                  # legal-shape granularity: shrink by whole
                               # data-parallel replicas of G nodes
SEEDS = range(24)


def _event_times(rng, event_rate):
    """Fault times in RUNNING time over the horizon (self-limiting)."""
    ts, t = [], 0.0
    while True:
        t += rng.exponential(1.0 / event_rate)
        if t >= T:
            break
        ts.append(t)
    return ts  # these are in running-time; wall-clock is longer (see below)


def ettr_fixed(policy, b, k, ckpt_c=CKPT_C, ckpt_w=CKPT_W, seeds=SEEDS):
    """no-spares / spares-k: job alternates full-width up and fully-down.
    Event rate is fixed; b is the blast radius (spares help iff k >= b)."""
    event_rate = BASE_RATE
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        t_wall = 0.0
        up = 0.0
        n_events = 0
        inv = k
        returns = []  # wall-times at which used spares come back
        while t_wall < T:
            gap = rng.exponential(1.0 / event_rate)      # up-time to next fault
            if t_wall + gap >= T:
                up += T - t_wall
                break
            up += gap
            t_wall += gap
            n_events += 1
            # spare inventory replenishment
            ready = [r for r in returns if r <= t_wall]
            inv += len(ready)
            returns = [r for r in returns if r > t_wall]
            if policy == "spares" and inv >= b:
                down = SWAP
                inv -= b
                returns += [t_wall + R] * b
            else:  # no-spares, or spares with an exhausted pool
                down = R
            t_wall += down
        lost = n_events * (ckpt_c / 2.0)                 # rework since last ckpt
        ckpt_write = up * (ckpt_w / ckpt_c)              # fraction of up spent writing
        productive = max(0.0, up - lost - ckpt_write)
        out.append(productive / T)
    return float(np.mean(out))


def _legal_shape_loss(down_nodes):
    """Usable-capacity loss for `down_nodes` failed nodes: you shrink to whole
    data-parallel replicas of G, so losing any node in a replica costs the
    whole replica until repair."""
    if down_nodes <= 0:
        return 0
    return int(np.ceil(down_nodes / G_ELASTIC) * G_ELASTIC)


def ettr_elastic(b, ckpt_c=CKPT_C, ckpt_w=CKPT_W, seeds=SEEDS):
    """elastic: keeps running narrower while nodes are in repair, but pays a
    reshard/restart stop per event and a legal-shape capacity tax."""
    event_rate = BASE_RATE
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        t = 0.0
        productive_nodeh = 0.0
        n_events = 0
        down_until = []  # repair-completion times of currently-failed nodes
        while t < T:
            gap = rng.exponential(1.0 / event_rate)
            end = min(t + gap, T)
            while t < end:
                down_until = [d for d in down_until if d > t]
                nxt = min([d for d in down_until if d < end], default=end)
                usable = N_NODES - _legal_shape_loss(len(down_until))
                productive_nodeh += (nxt - t) * usable
                t = nxt
            if t >= T:
                break
            n_events += 1
            down_until += [t + R] * b            # b nodes enter repair
            # reshard/communicator rebuild: a real full-gang stop
            t += RESHARD
        lost_nodeh = n_events * (ckpt_c / 2.0) * N_NODES
        ckpt_write_frac = ckpt_w / ckpt_c
        productive_nodeh *= (1.0 - ckpt_write_frac)
        productive_nodeh = max(0.0, productive_nodeh - lost_nodeh)
        out.append(productive_nodeh / (N_NODES * T))
    return float(np.mean(out))


# ---------------------------------------------------------------- figure 1
def fig_crossover_and_ckpt(outdir):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel A: ETTR vs spare-pool size k, for independent (b=1) vs bursty (b=8)
    ks = [0, 1, 2, 4, 8, 16, 32]
    for b, color in ((1, "#4f81bd"), (8, "#c0504d")):
        sp = [ettr_fixed("spares", b, k) for k in ks]
        el = ettr_elastic(b)
        axA.plot(ks, sp, marker="o", color=color,
                 label=f"warm spares, failures in bursts of {b}")
        axA.axhline(el, color=color, ls=":", lw=1.4,
                    label=f"elastic-shrink, bursts of {b}")
    axA.axvline(2, color="#888888", ls="--", lw=1)
    axA.annotate("repo's current\noperating point (k=2)", xy=(2, 0.5),
                 xytext=(6, 0.42), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#888888"))
    axA.set_xlabel("warm-spare pool size k (nodes)")
    axA.set_ylabel("ETTR (productive / envelope)")
    axA.set_ylim(0, 1)
    axA.set_title("Two spares is 'the biggest lever'\nonly if failures are independent")
    axA.legend(fontsize=7.5, loc="lower right")

    # Panel B: checkpoint interval with write cost -> real optimum
    cs = np.linspace(0.05, 2.0, 40)
    loss = [1.0 - ettr_fixed("spares", 1, 8, ckpt_c=c) for c in cs]
    c_star = np.sqrt(2 * CKPT_W / BASE_RATE)
    axB.plot(cs * 60, loss, color="#4e9a06")
    axB.axvline(c_star * 60, color="#888888", ls="--", lw=1)
    axB.annotate(f"Young/Daly optimum\n~{c_star*60:.0f} min", xy=(c_star*60, min(loss)),
                 xytext=(c_star*60 + 25, min(loss) + 0.03), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#888888"))
    axB.set_xlabel("checkpoint interval (minutes)")
    axB.set_ylabel("capacity lost (1 - ETTR)")
    axB.set_title("Checkpoint cadence has a real optimum\nonce write cost is modelled")
    fig.suptitle("Prototype: the current headline is regime-dependent", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "prototype_crossover.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_phase_map(outdir):
    bs = [1, 2, 3, 4, 6, 8, 12, 16]
    ks = [0, 1, 2, 4, 8, 16, 32, 64]
    Z = np.zeros((len(ks), len(bs)))       # ETTR(spares) - ETTR(elastic)
    for j, b in enumerate(bs):
        el = ettr_elastic(b)
        for i, k in enumerate(ks):
            Z[i, j] = ettr_fixed("spares", b, k) - el
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    lim = float(np.abs(Z).max()) or 1e-6
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    im = ax.imshow(Z, origin="lower", cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(range(len(bs)), bs)
    ax.set_yticks(range(len(ks)), ks)
    ax.set_xlabel("failure burstiness  b  (nodes lost per event; b=1 is independent)")
    ax.set_ylabel("warm-spare pool size  k  (nodes)")
    ax.set_title("Which policy wins?  ETTR(warm spares) − ETTR(elastic-shrink)\n"
                 "blue = spares win, red = elastic wins")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("ETTR advantage of spares")
    # contour of the tie line
    ax.contour(Z, levels=[0.0], colors="black", linewidths=1.5)
    # mark the repo's headline point
    ax.scatter([0], [2], marker="*", s=260, color="gold", edgecolor="black", zorder=5)
    ax.annotate("repo's headline\n(k=2, independent)", xy=(0, 2), xytext=(1.2, 3.6),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
    fig.tight_layout()
    fig.savefig(outdir / "prototype_phase_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary():
    print("=== prototype summary (means over seeds) ===")
    print(f"{'regime':28s} {'no-spares':>10s} {'spares k=2':>11s} "
          f"{'spares k=8':>11s} {'elastic':>9s}")
    for b in (1, 2, 4, 8, 16):
        ns = ettr_fixed("no-spares", b, 0)
        s2 = ettr_fixed("spares", b, 2)
        s8 = ettr_fixed("spares", b, 8)
        el = ettr_elastic(b)
        winner = max([("no-spares", ns), ("spares-k2", s2),
                      ("spares-k8", s8), ("elastic", el)], key=lambda x: x[1])
        print(f"bursts of {b:2d} nodes/event     {ns:10.3f} {s2:11.3f} "
              f"{s8:11.3f} {el:9.3f}   winner: {winner[0]}")
    c_star = np.sqrt(2 * CKPT_W / BASE_RATE)
    print(f"\ncheckpoint optimum (Young/Daly): ~{c_star*60:.0f} min "
          f"(vs the repo's fixed 30 min)")


if __name__ == "__main__":
    from pathlib import Path
    outdir = Path(__file__).resolve().parent
    summary()
    fig_crossover_and_ckpt(outdir)
    fig_phase_map(outdir)
    print(f"\nwrote 2 figures to {outdir}")
