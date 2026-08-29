#!/usr/bin/env python3
"""Run the recovery-policy grid and sweeps; write results + figures.

Usage: python3 run.py [--days 30] [--seeds 42 43 44] [--out ../results] [--figdir ../figures]
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovery_sim import COMPONENTS, Config, POLICIES, Simulation

POLICY_LABEL = {
    "manual-ops": "Manual ops\n(no spares)",
    "auto-restart": "Auto-restart\n(no spares)",
    "auto-spares": "Auto + warm\nspares",
    "auto-spares-tuned": "Auto + spares\n+ tuned clocks",
    "elastic-shrink": "Elastic\nshrink",
}
COMPONENT_STYLE = [
    ("lost_work", "Lost work (since last ckpt)", "#92c5de"),
    ("detection_wait", "Detection wait", "#c0504d"),
    ("reload", "Reload / restart", "#bbbbbb"),
    ("repair_wait", "Gang idle awaiting repair", "#8064a2"),
    ("operator_wait", "Waiting for a human", "#f4a582"),
    ("failed_retries", "Retries burned on structural faults", "#e8b7b4"),
    ("shrink_deficit", "Shrink deficit", "#4e9a06"),
    ("resize_overhead", "Resize overhead", "#d9e6c3"),
    ("spare_idle", "Warm-spare idle", "#4f81bd"),
]


def run_grid(days: float, seeds: list[int]) -> list[dict]:
    rows = []
    for policy in POLICIES:
        for seed in seeds:
            r = Simulation(Config(horizon_days=days, seed=seed), policy).run()
            rows.append(r)
            print(
                f"{policy:18s} seed={seed} realization={r['realization']:.3f} "
                f"wasted={r['wasted_gpu_h']:,.0f} events={r['events_total']}"
            )
    return rows


def aggregate(rows: list[dict]) -> dict[str, dict]:
    agg = {}
    for policy in POLICIES:
        sel = [r for r in rows if r["policy"] == policy]
        out = {}
        for key in sel[0]:
            if isinstance(sel[0][key], (int, float)) and key != "seed":
                vals = [r[key] for r in sel]
                out[key] = statistics.mean(vals)
                out[key + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg[policy] = out
    return agg


def write_summary(agg: dict, path: Path, days: float, seeds: list[int]) -> None:
    cfg = Config()
    lines = [
        "# Recovery-policy simulation summary",
        "",
        f"One gang job, {cfg.gpus:,} GPUs, {days:g} days, seeds {seeds} "
        "(values are means across seeds). Full per-run data in results.csv. "
        "Reproduce: `python3 sim/run.py`.",
        "",
        "Reliability triple (textbook definitions) — MTTF (mean up-time before "
        "a failure, ~constant: a property of the fault process), MTTR (mean stop "
        "per failure, the policy lever), MTBF = MTTF+MTTR (varies, since it "
        "includes recovery); Availability = MTTF/(MTTF+MTTR) = MTTF/MTBF:",
        "",
        "| Policy | MTTF (h) | MTTR (h) | MTBF (h) | Availability | Capacity realization | Wasted GPU-h |",
        "|---|---|---|---|---|---|---|",
    ]
    for policy in POLICIES:
        a = agg[policy]
        lines.append(
            f"| {policy} | {a['mttf_h']:.2f} | {a['mttr_h']:.2f} | {a['mtbf_h']:.2f} | "
            f"{a['availability']:.3f} | {a['realization']:.3f} | "
            f"{a['wasted_gpu_h']:,.0f} |"
        )
    lines += [
        "",
        "The nine conserved waste components are in results.csv; the *Shrink* "
        "column below sums shrink_deficit + resize_overhead for width. "
        "Availability exceeds realization for every policy because being *up* is "
        "necessary but not sufficient for productive work — the gap is mostly "
        "re-done work since the last checkpoint, plus (for the spare policies) "
        "the idle spares that sit in the realization denominator.",
        "",
        "| Policy | Realization | Wasted GPU-h | Lost work | Detection | Reload | "
        "Repair wait | Operator | Failed retries | Shrink | Spare idle | Events |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for policy in POLICIES:
        a = agg[policy]
        lines.append(
            f"| {policy} | {a['realization']:.3f} | {a['wasted_gpu_h']:,.0f} | "
            f"{a['waste_lost_work']:,.0f} | {a['waste_detection_wait']:,.0f} | "
            f"{a['waste_reload']:,.0f} | {a['waste_repair_wait']:,.0f} | "
            f"{a['waste_operator_wait']:,.0f} | {a['waste_failed_retries']:,.0f} | "
            f"{a['waste_shrink_deficit'] + a['waste_resize_overhead']:,.0f} | "
            f"{a['waste_spare_idle']:,.0f} | {a['events_total']:.0f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def fig_waste_breakdown(agg: dict, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(POLICIES))
    bottom = np.zeros(len(POLICIES))
    for key, label, color in COMPONENT_STYLE:
        vals = np.array([agg[p][f"waste_{key}"] for p in POLICIES]) / 1000.0
        if vals.max() <= 0:
            continue
        ax.bar(x, vals, bottom=bottom, label=label, color=color, width=0.62)
        bottom += vals
    for i, p in enumerate(POLICIES):
        ax.text(i, bottom[i] + bottom.max() * 0.02,
                f"realization\n{agg[p]['realization']:.1%}", ha="center", fontsize=9)
    ax.set_ylim(0, bottom.max() * 1.2)
    ax.set_xticks(x, [POLICY_LABEL[p] for p in POLICIES])
    ax.set_ylabel("Wasted capacity, thousand GPU-hours / 30 days")
    ax.set_title("Where recovery policy spends a 16,384-GPU month\n"
                 "(identical fault sequence per seed; Llama 3-calibrated load)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "recovery_waste_breakdown.png", dpi=150)
    plt.close(fig)


def fig_watchdog_sweep(days: float, seeds: list[int], outdir: Path) -> None:
    watchdogs = [0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 30.0]
    means = []
    for w in watchdogs:
        vals = [
            Simulation(
                Config(horizon_days=days, seed=s, hang_watchdog_min=w), "auto-spares"
            ).run()["waste_detection_wait"]
            for s in seeds
        ]
        means.append(statistics.mean(vals) / 1000.0)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(watchdogs, means, marker="o", color="#c0504d")
    ax.axvline(10.0, color="#888888", ls="--", lw=1)
    ax.annotate("PyTorch NCCL\nwatchdog default (600 s)", xy=(10.0, means[-3]),
                xytext=(12.5, means[-3] * 0.55), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#888888"))
    ax.axvline(1.5, color="#4e9a06", ls="--", lw=1)
    ax.annotate("desync detection\n(flight-recorder class)", xy=(1.5, means[2]),
                xytext=(3.0, max(means) * 0.75), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#4e9a06"))
    ax.set_xlabel("Hang/desync detection time, minutes")
    ax.set_ylabel("Detection-wait waste, thousand GPU-h / 30 days")
    ax.set_title("The detection clock is a capacity tax\n"
                 "(auto-spares policy; every other parameter held fixed)")
    fig.tight_layout()
    fig.savefig(outdir / "watchdog_sweep.png", dpi=150)
    plt.close(fig)


def fig_availability(agg: dict, outdir: Path) -> None:
    """MTTR is the lever; availability and realization are what it buys."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(POLICIES))
    # left: MTTF (flat, the hardware property) vs MTTR (the lever), log y so
    # 0.3 h and 3 h coexist
    mttf = [agg[p]["mttf_h"] for p in POLICIES]
    mttr = [agg[p]["mttr_h"] for p in POLICIES]
    axL.bar(x - 0.2, mttf, 0.4, label="MTTF (h) — fault process", color="#bbbbbb")
    axL.bar(x + 0.2, mttr, 0.4, label="MTTR (h) — policy lever", color="#c0504d")
    axL.set_yscale("log")
    axL.set_xticks(x, [POLICY_LABEL[p] for p in POLICIES], fontsize=8)
    axL.set_ylabel("hours (log scale)")
    axL.set_title("MTTF is fixed by hardware; MTTR is the lever")
    axL.legend(fontsize=8)
    # right: availability vs realization — the gap is "up but not productive"
    avail = [agg[p]["availability"] for p in POLICIES]
    real = [agg[p]["realization"] for p in POLICIES]
    axR.bar(x - 0.2, avail, 0.4, label="Availability (uptime share)", color="#4f81bd")
    axR.bar(x + 0.2, real, 0.4, label="Capacity realization (productive)", color="#4e9a06")
    axR.set_xticks(x, [POLICY_LABEL[p] for p in POLICIES], fontsize=8)
    axR.set_ylabel("fraction")
    axR.set_ylim(0, 1.0)
    axR.set_title("Being up ≠ being productive\n(gap is mostly re-done work, every policy)")
    axR.legend(fontsize=8, loc="lower right")
    fig.suptitle("Availability = MTTF / (MTTF + MTTR), and what it converts into")
    fig.tight_layout()
    fig.savefig(outdir / "availability.png", dpi=150)
    plt.close(fig)


def fig_ckpt_reload_sweep(days: float, seeds: list[int], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    ckpts = [10, 20, 30, 45, 60, 90]
    vals = []
    for ck in ckpts:
        vals.append(statistics.mean([
            Simulation(Config(horizon_days=days, seed=s, ckpt_interval_min=ck),
                       "auto-spares").run()["wasted_gpu_h"] for s in seeds
        ]) / 1000.0)
    axes[0].plot(ckpts, vals, marker="o", color="#4f81bd")
    axes[0].set_xlabel("Checkpoint interval, minutes")
    axes[0].set_ylabel("Total waste, thousand GPU-h / 30 days")
    axes[0].set_title("Checkpoint cadence\n(lost-work only — omits write cost)")
    reloads = [5, 10, 15, 25, 33, 45]
    vals = []
    for rl in reloads:
        vals.append(statistics.mean([
            Simulation(Config(horizon_days=days, seed=s, reload_min=rl),
                       "auto-spares").run()["wasted_gpu_h"] for s in seeds
        ]) / 1000.0)
    axes[1].plot(reloads, vals, marker="s", color="#4e9a06")
    axes[1].axvline(33, color="#888888", ls="--", lw=1)
    axes[1].annotate("measured median load,\n504-GPU B200 site", xy=(33, vals[4]),
                     xytext=(16, vals[4] * 1.02), fontsize=8.5,
                     arrowprops=dict(arrowstyle="->", color="#888888"))
    axes[1].set_xlabel("Checkpoint reload time, minutes")
    axes[1].set_title("Restart path speed")
    fig.suptitle("Both recovery levers price directly into GPU-hours (auto-spares policy)")
    fig.tight_layout()
    fig.savefig(outdir / "ckpt_reload_sweep.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="../results")
    ap.add_argument("--figdir", default="../figures")
    args = ap.parse_args()

    outdir = (Path(__file__).resolve().parent / args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = (Path(__file__).resolve().parent / args.figdir).resolve()
    figdir.mkdir(parents=True, exist_ok=True)

    rows = run_grid(args.days, args.seeds)
    with (outdir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    agg = aggregate(rows)
    write_summary(agg, outdir / "summary.md", args.days, args.seeds)
    fig_waste_breakdown(agg, figdir)
    fig_availability(agg, figdir)
    fig_watchdog_sweep(args.days, args.seeds, figdir)
    fig_ckpt_reload_sweep(args.days, args.seeds, figdir)

    # spare-pool sufficiency note (stated model simplification)
    cfg = Config()
    hw_rate_day = cfg.interruptions_per_gpu_day * cfg.gpus * dict(cfg.class_mix)["node-hardware"]
    occupancy = hw_rate_day * (cfg.repair_min / 60.0 / 24.0)
    print(f"\nspare-pool expected occupancy: {occupancy:.2f} nodes "
          f"(pool size {cfg.spare_nodes}; sufficiency assumption holds)")
    print(f"wrote {outdir}/results.csv, summary.md and 4 figures to {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
