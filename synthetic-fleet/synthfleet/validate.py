"""Validate a baseline (no-chaos) run against public calibration targets."""
import json, os
import numpy as np
import pandas as pd
from .runner import run
from . import telemetry

def _rate_check(obs_n, exposure, target, tol):
    exp = target * exposure
    ok = abs(obs_n - exp) <= max(tol * exp, 2 * np.sqrt(max(exp, 1)))
    return obs_n / exposure if exposure else np.nan, ok

def validate(cfg_path, out_dir, seed=42):
    o = run(cfg_path, out_dir, seed=seed, chaos=False, write=False)
    W, cfg, cal, ev, fates, attr = o["W"], o["cfg"], o["calib"], o["ev"], o["fates"], o["attr"]
    days = cfg["sim"]["days"]; gens = cfg["generations"]; bom = W["bom"]; jobs = W["jobs"]; ints = W["sim"].interrupts
    R = []
    def add(id_, name, kind, target, value, ok, source, note=""):
        R.append(dict(id=id_, check=name, kind=kind, target=target, simulated=value, passed=bool(ok), source=source, note=note))
    bev = ev[(ev.source == "baseline") & (ev.kind == "down") & (ev.t0 >= 0)]
    unpl = bev[~bev.planned]
    for gen, p, src in [("H100", "h100_interrupt_rate", "Llama 3 (Meta 2024)"), ("A100", "a100_interrupt_rate", "Meta RSC (Kokolis 2024)")]:
        n = (unpl.gen == gen).sum(); dd = bom[bom.gen == gen].devices.sum() * days
        v, ok = _rate_check(n, dd, cal[p], 0.15)
        add(f"rate_{gen}", f"{gen} unexpected interruptions per GPU-day", "implementation", cal[p], v, ok, src)
    # job MTTF
    tj = jobs[(jobs.workload == "training")]
    big = tj[(tj.gen == "H100")].sort_values("n_devices").iloc[-1]
    n = ((ints.job_id == big.job_id) & (~ints.planned)).sum()
    m = days * 24 / max(n, 1)
    add("mttf_16k_h100", f"MTTF of {big.n_devices}-GPU H100 training job (h)", "emergent", cal["target_mttf_16k_h100_h"], m * big.n_devices / 16384,
        abs(m * big.n_devices / 16384 - cal["target_mttf_16k_h100_h"]) / cal["target_mttf_16k_h100_h"] < 0.25, "Llama 3: 419 unexpected in 54 days on 16,384 GPUs",
        "scaled to 16,384 GPUs; counts only interrupts outside scheduler gaps")
    a = tj[tj.gen == "A100"]; n = ints[ints.job_id.isin(a.job_id) & ~ints.planned].shape[0]
    gpu_h = (a.n_devices.sum()) * days * 24
    m1024 = gpu_h / max(n, 1) / 1024
    add("mttf_1024_a100", "MTTF of a 1,024-GPU A100 job (h)", "emergent", cal["target_mttf_1024_a100_h"], m1024,
        abs(m1024 - cal["target_mttf_1024_a100_h"]) / cal["target_mttf_1024_a100_h"] < 0.25, "Meta RSC: 7.9 h at 1,024 GPUs",
        "calibrated rate uses RSC's per-node 47.7-day MTTF, so a ~10% gap vs the 7.9 h job figure is expected")
    # ETTR
    f = fates[fates.workload == "training"].merge(jobs[["job_id", "n_devices"]], on="job_id")
    gpu = f[(~f.gen.astype(str).str.startswith("TPU")) & (f.n_devices >= 4096)]
    ettr = gpu.useful_dh.sum() / (gpu.useful_dh + gpu.degraded_dh + gpu.discarded_dh + gpu.blocked_dh).sum()
    lo, hi = cal["target_ettr_large_gpu_jobs"]
    add("ettr_large_gpu", "ETTR, GPU training jobs >= 4,096 devices", "emergent (restart/checkpoint tuned)", f"{lo}-{hi}", ettr, lo - 0.02 <= ettr <= hi + 0.02,
        "Meta RSC ETTR 0.85-0.90 for large jobs")
    tpu = f[f.gen.isin(["TPUv4", "TPUv5p", "TPUv7"])]
    gp = tpu.useful_dh.sum() / (tpu.useful_dh + tpu.degraded_dh + tpu.discarded_dh + tpu.blocked_dh).sum()
    add("tpu_goodput", "Goodput, TPU OCS-superpod training jobs", "emergent", cal["target_tpu_goodput"], gp, abs(gp - cal["target_tpu_goodput"]) <= 0.02,
        "Gemini: goodput 85% -> 97%")
    # TPU rates (all generations, normalised by rate multiplier)
    tp = bom[bom.family == "tpu"]
    mult = {g: (cal[gens[g]["rate_mult_param"]] if gens[g].get("rate_mult_param") else 1.0) for g in tp.gen}
    mach_exp = sum(r.devices / gens[r.gen]["dev_per_node"] * mult[r.gen] for r in tp.itertuples()) * days
    n = ((bev.cause == "tpu_machine")).sum(); v, ok = _rate_check(n, mach_exp, cal["tpuv4_machine_rate"], 0.15)
    add("tpu_machine_rate", "TPU machine failures per machine-day (v4-normalised)", "implementation", cal["tpuv4_machine_rate"], v, ok, "TPU v4 NSDI'23/24: 0.08%/day")
    cab_exp = sum(r.devices * cal["tpu_ici_cables_per_chip"] * mult[r.gen] for r in tp.itertuples()) * days
    n = (bev.cause == "ici_cable").sum(); v, ok = _rate_check(n, cab_exp, cal["tpuv4_ici_cable_rate"], 0.15)
    add("tpu_ici_rate", "ICI cable failures per cable-day", "implementation", cal["tpuv4_ici_cable_rate"], v, ok, "TPU v4: 0.005%/day")
    ocs_exp = sum(r.devices * cal["tpu_ocs_per_4096_chips"] / 4096 * mult[r.gen] for r in tp.itertuples() if "OCS" in gens[r.gen]["fabric"]) * days
    n = ((ev.source == "baseline") & (ev.cause == "ocs") & (ev.t0 >= 0)).sum(); v, ok = _rate_check(n, ocs_exp, cal["tpuv4_ocs_rate"], 0.15)
    add("tpu_ocs_rate", "OCS failures per OCS-day", "implementation", cal["tpuv4_ocs_rate"], v, ok, "TPU v4: 0.04%/day", f"{n} events; Poisson-tolerant check")
    # Ironwood cubes
    seg = o["seg"].merge(ev[["event_id", "t0", "dur", "kind"]], on="event_id")
    pods = W["pods"]; full = pods[(pods.gen == "TPUv7") & (pods.n_devices == 9216)]
    T = W["sim"].T; dt = W["sim"].dt; fr = []
    for p in full.itertuples():
        s = seg[(seg.pod_id == p.pod_id) & (seg.kind == "down")]
        bad = np.zeros((T, 144), bool)
        for r in s.itertuples():
            if r.t0 + r.dur <= 0 or r.t0 >= T * dt: continue
            i0, i1 = int(max(0, r.t0) // dt), int(min(T * dt, r.t0 + r.dur) // dt) + 1
            c0, c1 = r.dev_offset // 64, (r.dev_offset + r.dev_count - 1) // 64 + 1
            bad[i0:i1, c0:c1] = True
        fr.append(((144 - bad.sum(1)) >= 130).mean())
    v = float(np.mean(fr)) if fr else np.nan
    add("ironwood_cubes", "Share of time >= 130 of 144 Ironwood cubes healthy", "emergent (Ironwood rates are assumptions)", 0.95, v, v >= 0.95,
        "Google Ironwood blog: 130/144 cubes available 95% of the month")
    # StepFun shape
    g = unpl[~unpl.gen.astype(str).str.startswith("TPU") & (unpl.gen != "A100")]
    add("mttr_median", "GPU node fault MTTR median (days, excl. A100 which is rescaled)", "implementation", 0.935, g.dur.median() / 1440, abs(g.dur.median() / 1440 - 0.935) / 0.935 < 0.2,
        "StepFun InfiniteHBD-Trace (derived)", "class-mixture lognormal; truncated at 45 d")
    sh = (g.cause == "gpu").mean()
    add("gpu_share", "Share of node faults classed GPU", "implementation", 0.3244, sh, abs(sh - 0.3244) < 0.03, "StepFun InfiniteHBD-Trace")
    mn = (g.nodes.fillna(1) > 1).mean()
    add("multi_node", "Share of faults spanning >1 node", "implementation", cal["multi_node_fault_prob"], mn, abs(mn - cal["multi_node_fault_prob"]) < 0.03, "ByteDance alert trace (derived)")
    pr = bev.planned.sum() / max(1, (~bev.planned).sum())
    add("planned_ratio", "Planned / unexpected interruptions", "implementation", cal["planned_maintenance_ratio"], pr, abs(pr - cal["planned_maintenance_ratio"]) < 0.02, "Llama 3: 47 / 419")
    # telemetry based
    rng = np.random.default_rng([seed, 4])
    cfg2 = json.loads(json.dumps(cfg)); cfg2["sim"]["device_sample_max"] = 3000; cfg2["sim"]["device_sample_min_per_group"] = 40
    ids = telemetry.sample_devices(cfg2, W["pods"], rng)
    dev = telemetry.device_series(cfg2, cal, W["pods"], W["jobs"], W["sim"], W["events"], ids, rng)
    e = dev.groupby("gen", observed=True).agg(ecc=("ecc_correctable", "sum"), n=("ecc_correctable", "size"))
    ratio = (e.loc["H100", "ecc"] / e.loc["H100", "n"]) / (e.loc["A100", "ecc"] / e.loc["A100", "n"])
    add("ecc_ratio", "H100 / A100 correctable memory-error rate", "implementation", cal["h100_vs_a100_memory_error_ratio"], ratio,
        abs(ratio - 3.2) / 3.2 < 0.25, "NCSA Delta (Cui et al. 2025)", "absolute A100 rate is an assumption")
    idle = dev[(dev.true_fate == "idle") & (dev.util < 0.05)]
    for gen in ("A100", "H100", "H200"):
        v = float(idle[idle.gen == gen].power_w.mean()); t = cal["idle_power_w"][gen]
        add(f"idle_{gen}", f"{gen} idle power (W)", "implementation", t, v, abs(v - t) / t < 0.10, "LLM-PowerTrace")
    fac = bom.facility_mw.sum(); tgt = W["target"]
    add("nameplate", "Facility nameplate (MW)", "reconciliation", tgt, fac, abs(fac - tgt) / tgt < 0.01, "config")
    s = fates[[c for c in fates.columns if c.endswith("_dh")]]
    err = float((s.installed_dh - s[["useful_dh", "degraded_dh", "discarded_dh", "blocked_dh", "idle_dh", "down_dh", "undelivered_dh"]].sum(1)).abs().max())
    add("identity", "Max |installed - sum of fates| per row (device-hours)", "identity", 0, err, err < 1e-2, "definition")
    rep = pd.DataFrame(R)
    os.makedirs(out_dir, exist_ok=True)
    rep.to_csv(os.path.join(out_dir, "validation.csv"), index=False)
    with open(os.path.join(out_dir, "VALIDATION.md"), "w") as fh:
        fh.write(f"# Validation (baseline only, seed {seed}, {days} days)\n\n{int(rep.passed.sum())}/{len(rep)} checks passed.\n\n")
        fh.write("Kinds: *implementation* = the generator reproduces a parameter it was given; *emergent* = an outcome that was not set directly; "
                 "*reconciliation*/*identity* = accounting checks.\n\n| id | check | kind | target | simulated | pass | source | note |\n|---|---|---|---|---|---|---|---|\n")
        for r in rep.itertuples():
            sv = f"{r.simulated:.4g}" if isinstance(r.simulated, (float, np.floating, int)) else str(r.simulated)
            fh.write(f"| {r.id} | {r.check} | {r.kind} | {r.target} | {sv} | {'yes' if r.passed else 'NO'} | {r.source} | {r.note} |\n")
    return rep
