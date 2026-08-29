#!/usr/bin/env python3
"""Produce the reliability-economics results: the policy phase diagram, the
crossover, the checkpoint optimum, the validation-against-anchors panel, plus
results.csv / summary.md.

Usage: python3 run.py [--seeds 8] [--out ../results] [--figdir ../figures]
"""
from __future__ import annotations
import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from reliability_sim import Config, Sim, run_policy

TOTAL_NODE_RATE = 1.58e-4 * 2048          # node-failures/h at the default fleet
CORR_FRACTION = 0.5                        # half the failure budget is correlated


def burst_rate_for(blast: int) -> float:
    """Correlation axis: hold total node-failure rate fixed; a fixed fraction of
    it arrives as bursts of `blast` nodes (blast=1 ⇒ effectively independent)."""
    if blast <= 1:
        return 0.0
    return CORR_FRACTION * TOTAL_NODE_RATE / blast


def ettr(policy, seeds, **kw):
    return statistics.mean(run_policy(Config(seed=s, **kw), policy)["ettr"] for s in seeds)


# ----------------------------------------------------------------- phase map
def fig_phase(seeds, figdir):
    blasts = [1, 2, 4, 8, 16, 32]
    ks = [0, 2, 4, 8, 16, 32, 64]
    Z = np.zeros((len(ks), len(blasts)))
    for j, b in enumerate(blasts):
        br = burst_rate_for(b)
        el = ettr("elastic", seeds, burst_rate_per_h=br, burst_blast=b)
        for i, k in enumerate(ks):
            sp = ettr("auto-spares-ckpt", seeds, spare_nodes=k, burst_rate_per_h=br, burst_blast=b)
            Z[i, j] = sp - el
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    lim = float(np.abs(Z).max()) or 1e-6
    im = ax.imshow(Z, origin="lower", cmap="RdBu", aspect="auto",
                   norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    ax.set_xticks(range(len(blasts)), blasts)
    ax.set_yticks(range(len(ks)), ks)
    ax.set_xlabel("failure blast radius  (nodes per correlated event; 1 = independent)")
    ax.set_ylabel("warm-spare pool size  k  (nodes)")
    ax.set_title("Which recovery policy wins?\n"
                 "ETTR(spares + overlapped checkpoint) − ETTR(elastic-shrink)\n"
                 "blue = spares win · red = elastic wins")
    ax.contour(Z, levels=[0.0], colors="black", linewidths=1.5)
    fig.colorbar(im, ax=ax, label="ETTR advantage of spares")
    fig.tight_layout()
    fig.savefig(figdir / "phase_diagram.png", dpi=150)
    plt.close(fig)
    return Z


# ----------------------------------------------------------------- crossover
def fig_crossover(seeds, figdir):
    ks = [0, 2, 4, 8, 16, 32, 64]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for b, color in ((1, "#4f81bd"), (16, "#c0504d")):
        br = burst_rate_for(b)
        sp = [ettr("auto-spares-ckpt", seeds, spare_nodes=k, burst_rate_per_h=br, burst_blast=b) for k in ks]
        el = ettr("elastic", seeds, burst_rate_per_h=br, burst_blast=b)
        lbl = "independent (b=1)" if b == 1 else "correlated bursts (b=16)"
        ax.plot(ks, sp, marker="o", color=color, label=f"spares+ckpt — {lbl}")
        ax.axhline(el, color=color, ls=":", lw=1.4, label=f"elastic — {lbl}")
    ax.set_xlabel("warm-spare pool size k (nodes)")
    ax.set_ylabel("ETTR")
    ax.set_ylim(0, 1)
    ax.set_title("Spares beat elastic only with enough pool for the blast radius")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(figdir / "crossover.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------- ckpt optimum
def fig_ckpt(seeds, figdir):
    ivs = np.array([0.03, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0])
    waste = [statistics.mean(
        run_policy(Config(seed=s, spare_nodes=16, ckpt_interval_h=iv), "auto-spares")["waste_gpu_h"]
        for s in seeds) / 1000.0 for iv in ivs]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(ivs * 60, waste, marker="o", color="#4e9a06")
    lo = int(np.argmin(waste))
    ax.axvline(ivs[lo] * 60, color="#888888", ls="--", lw=1)
    ax.annotate(f"optimum ~{ivs[lo]*60:.0f} min", xy=(ivs[lo]*60, waste[lo]),
                xytext=(ivs[lo]*60 + 20, waste[lo] + (max(waste)-min(waste))*0.15),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#888888"))
    ax.set_xlabel("blocking-checkpoint interval (minutes)")
    ax.set_ylabel("wasted capacity, thousand GPU-h / month")
    ax.set_title("Checkpoint cadence has a real optimum\n(write cost vs lost work)")
    fig.tight_layout()
    fig.savefig(figdir / "checkpoint_optimum.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------- validation panel
def fig_validation(seeds, figdir):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    # left: Meta-like config approaches >90% ETTR; naive does not
    configs = [
        ("manual\n(no spares)", "manual", dict(spare_nodes=0)),
        ("auto-restart\n(no spares)", "auto-restart", dict(spare_nodes=0)),
        ("auto-spares\nk=8", "auto-spares", dict(spare_nodes=8)),
        ("Meta-like\nk=16 + overlap ckpt", "auto-spares-ckpt", dict(spare_nodes=16)),
    ]
    vals = [ettr(p, seeds, **kw) for _, p, kw in configs]
    axL.bar(range(len(configs)), vals, color=["#c0504d", "#e08050", "#4f81bd", "#4e9a06"])
    axL.axhline(0.90, color="#888888", ls="--", lw=1)
    axL.text(0.05, 0.905, "Meta Llama 3: >90% effective training time", fontsize=8, color="#555")
    axL.set_xticks(range(len(configs)), [c[0] for c in configs], fontsize=8)
    axL.set_ylabel("ETTR")
    axL.set_ylim(0, 1)
    axL.set_title("Anchor 1 — a Meta-like config reproduces the >90% regime")
    # right: MTTF ~ 1/N (RSC anchor)
    Ns = [512, 1024, 2048, 4096, 8192]
    mttf = [statistics.mean(run_policy(Config(seed=s, nodes=n, spare_nodes=8), "elastic")["mttf_h"]
                            for s in seeds) for n in Ns]
    gpus = [n * 8 for n in Ns]
    axR.loglog(gpus, mttf, marker="o", color="#4f81bd", label="simulated MTTF")
    ref = [mttf[2] * gpus[2] / g for g in gpus]           # 1/N reference through N=2048
    axR.loglog(gpus, ref, ls="--", color="#888888", label="∝ 1/N reference")
    axR.set_xlabel("GPUs")
    axR.set_ylabel("MTTF (h, between job interruptions)")
    axR.set_title("Anchor 2 — MTTF falls with N, consistent with Meta RSC\n"
                  "(slightly steeper than 1/N at the top end)")
    axR.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "validation.png", dpi=150)
    plt.close(fig)


def write_results(seeds, outdir):
    rows = []
    scenarios = {
        "independent": dict(),
        "rack-bursts-b16": dict(burst_rate_per_h=burst_rate_for(16), burst_blast=16),
    }
    for scen, kw in scenarios.items():
        for policy in Sim.POLICIES:
            for k in (0, 2, 8, 32):
                if policy not in ("auto-spares", "auto-spares-ckpt") and k != 0:
                    continue
                vs = [run_policy(Config(seed=s, spare_nodes=k, **kw), policy) for s in seeds]
                row = {"scenario": scen, "policy": policy, "spare_nodes": k}
                for key in ("ettr", "availability", "mttf_h", "mttr_h", "interruptions",
                            "waste_gpu_h", "waste_dollars"):
                    row[key] = round(statistics.mean(v[key] for v in vs), 3)
                rows.append(row)
    with (outdir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    lines = ["# Reliability-economics results (v1)", "",
             f"16,384-GPU gang training job, 30 days, {len(list(seeds))} seeds. "
             "ETTR = productive / envelope. Reproduce: `python3 sim/run.py`.", "",
             "| Scenario | Policy | spares (nodes) | ETTR | Avail | MTTF h | MTTR h | Interruptions | $ waste/mo |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['policy']} | {r['spare_nodes']} | {r['ettr']:.3f} | "
                     f"{r['availability']:.3f} | {r['mttf_h']:.1f} | {r['mttr_h']:.2f} | "
                     f"{r['interruptions']:.0f} | ${r['waste_dollars']:,.0f} |")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--figdir", default="../figures")
    a = ap.parse_args()
    seeds = range(a.seeds)
    outdir = (Path(__file__).resolve().parent / a.out).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    figdir = (Path(__file__).resolve().parent / a.figdir).resolve(); figdir.mkdir(parents=True, exist_ok=True)
    fig_phase(seeds, figdir)
    fig_crossover(seeds, figdir)
    fig_ckpt(seeds, figdir)
    fig_validation(seeds, figdir)
    write_results(seeds, outdir)
    print(f"wrote results + 4 figures to {outdir} / {figdir}")


if __name__ == "__main__":
    main()
