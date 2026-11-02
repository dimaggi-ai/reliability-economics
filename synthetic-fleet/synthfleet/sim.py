"""Simulate ground-truth fates per job per interval (device-hours), power, and site curtailment.

Fate identity per (job, interval):
installed = useful + degraded + discarded + blocked + idle + down + undelivered   (exact)
"""
from collections import defaultdict
import numpy as np
import pandas as pd
from .fleet import idle_w

FATES = ["useful", "degraded", "discarded", "blocked", "idle", "down", "undelivered"]

def cover(t0, dur, dt, T):
    a = max(t0, 0.0); b = min(t0 + dur, T * dt)
    if b <= a:
        return None
    i0 = int(a // dt); i1 = min(T, int(np.ceil(b / dt)))
    edges = np.arange(i0, i1 + 1) * dt
    w = (np.minimum(edges[1:], b) - np.maximum(edges[:-1], a)) / dt
    return i0, i1, w

def addw(arr, c, v=1.0):
    if c is not None:
        arr[c[0]:c[1]] += c[2] * v

class Sim:
    def __init__(self, cfg, calib, pods, jobs, events, rng):
        self.cfg, self.calib, self.pods, self.jobs, self.ev, self.rng = cfg, calib, pods, jobs, events, rng
        sim = cfg["sim"]
        self.dt = sim["interval_min"]; self.T = int(sim["days"] * 1440 // self.dt); self.dth = self.dt / 60
        self.t_min = np.arange(self.T) * self.dt
        self.pod_job = dict(zip(pods.pod_id, pods.job_id))
        self.sites = {s["id"]: s for s in cfg["sites"]}

    def _job_events(self):
        by_job = defaultdict(list); site_ev = defaultdict(list)
        for e in self.ev:
            if e["kind"] in ("powercap", "demand", "pue"):
                site_ev[e["site"]].append(e); continue
            cnt = defaultdict(int)
            for (p, off, c) in e["segs"]:
                cnt[self.pod_job[p]] += c
            for j, c in cnt.items():
                by_job[j].append((e, c))
        return by_job, site_ev

    def run(self):
        cal, rng, T, dt, dth = self.calib, self.rng, self.T, self.dt, self.dth
        by_job, site_ev = self._job_events()
        self.site_ev = site_ev
        gens = self.cfg["generations"]
        res = {}
        attrib = {}
        interrupts = []
        for jb in self.jobs.itertuples():
            g = gens[jb.gen]; N = float(jb.n_devices); s = self.sites[jb.site]
            tpu = g["family"] == "tpu"
            R_med = cal["tpu_restart_median_min"] if tpu else cal["gpu_restart_median_min"]
            ckpt = cal["tpu_checkpoint_interval_min"] if tpu else cal["gpu_checkpoint_interval_min"]
            keys = {f: defaultdict(lambda: np.zeros(T)) for f in FATES}
            und = np.zeros(T); down = np.zeros(T)
            blockf = defaultdict(lambda: np.zeros(T)); rework = defaultdict(lambda: np.zeros(T)); degr = defaultdict(lambda: np.zeros(T))
            evs = by_job.get(jb.job_id, [])
            for e, c in evs:
                cv = cover(e["t0"], e["dur"], dt, T); k = (e["cause"], e["party"])
                if e["kind"] == "undelivered":
                    addw(keys["undelivered"][k], cv, c)
                elif e["kind"] == "down":
                    addw(keys["down"][k], cv, c)
                elif e["kind"] == "degrade":
                    v = e["value"] if (e["job_wide"] or jb.workload == "training") else e["value"] * c / N
                    addw(degr[k], cv, v)
            und = sum(keys["undelivered"].values()) if keys["undelivered"] else np.zeros(T)
            und = np.minimum(und, N)
            D = sum(keys["down"].values()) if keys["down"] else np.zeros(T)
            cap = N - und
            scale = np.where(D > cap, cap / np.maximum(D, 1e-9), 1.0)
            for k in keys["down"]:
                keys["down"][k] *= scale
            D = np.minimum(D, cap)
            H = cap - D  # healthy installed devices
            # training: gaps between runs
            gap = np.zeros(T)
            if jb.workload == "training":
                t = -rng.uniform(0, cal["training_run_median_days"] * 1440)
                while t < T * dt:
                    t += np.exp(np.log(cal["training_run_median_days"] * 1440) + 0.5 * rng.standard_normal())
                    gl = np.exp(np.log(cal["training_gap_median_min"]) + 0.5 * rng.standard_normal())
                    addw(gap, cover(t, gl, dt, T)); t += gl
                gap = np.minimum(gap, 1.0)
            # interrupts -> restart block + rework
            for e, c in evs:
                if not e["interrupt"] or e["kind"] not in ("down", "block", "sdc"):
                    continue
                i = int(min(T - 1, max(0, e["t0"] // dt)))
                if e["t0"] < 0 or e["t0"] >= T * dt or gap[i] > 0.5 or H[i] <= 0 and e["kind"] != "block":
                    continue
                k = (e["cause"], e["party"])
                R = np.exp(np.log(cal["inference_restart_min"] if jb.workload == "inference" else R_med) + cal["restart_sigma"] * rng.standard_normal())
                frac = 1.0 if jb.workload == "training" else min(1.0, c / N)
                if e["kind"] == "block":
                    addw(blockf[k], cover(e["t0"], e["dur"] + R, dt, T), frac * e["value"])
                    tau = rng.uniform(0, ckpt)
                    if jb.workload == "training":
                        addw(rework[k], cover(e["t0"] - tau, tau, dt, T))
                elif e["kind"] == "sdc":
                    if jb.workload == "training":
                        addw(rework[k], cover(e["t0"], e["delay"], dt, T))
                        addw(blockf[k], cover(e["t0"] + e["delay"], R, dt, T))
                    else:
                        addw(rework[k], cover(e["t0"], e["delay"], dt, T), g["dev_per_node"] / N)
                else:
                    addw(blockf[k], cover(e["t0"], R, dt, T), frac)
                    if jb.workload == "training":
                        tau = rng.uniform(0, ckpt)
                        addw(rework[k], cover(e["t0"] - tau, tau, dt, T))
                interrupts.append(dict(job_id=jb.job_id, event_id=e["event_id"], t_min=e["t0"], cause=e["cause"], party=e["party"],
                                       planned=e["planned"], source=e["source"], restart_min=R))
            B = sum(blockf.values()) if blockf else np.zeros(T)
            RW = sum(rework.values()) if rework else np.zeros(T)
            G = sum(degr.values()) if degr else np.zeros(T)
            demand = np.zeros(T); served = np.zeros(T)
            if jb.workload == "training":
                spareN = jb.spare_frac * N
                A = np.clip(np.minimum(H, cap - spareN), 0, None)
                spare_idle = H - A
                bfrac = np.minimum(B, 1 - gap)
                run = A * (1 - gap - bfrac)
                dfrac = np.minimum(RW, 1.0); disc = run * dfrac
                prod = run - disc
                gfrac = np.minimum(G, 0.9); deg = prod * gfrac
                use = prod - deg
                keys["idle"][("reserved_spare", "operator")] += spare_idle
                keys["idle"][("scheduler_gap", "tenant")] += A * gap
                blocked = A * bfrac
            else:
                lt = (self.t_min / 60 + s["utc_offset_h"]) % 24
                load = cal["inference_load_base"] + cal["inference_load_diurnal_amp"] * np.sin(2 * np.pi * (lt - 9) / 24)
                load = load * (1 + 0.05 * _ar1(rng, T))
                mult = np.ones(T)
                for e in site_ev.get(jb.site, []):
                    if e["kind"] == "demand" and (not e["gen"] or e["gen"] == jb.gen):
                        cv = cover(e["t0"], e["dur"], dt, T)
                        if cv: mult[cv[0]:cv[1]] *= 1 + (e["value"] - 1) * cv[2]
                demand = np.clip(load, 0.05, None) * mult * N * dth
                bfrac = np.minimum(B, 1.0)
                C = H * (1 - bfrac)
                blocked = H * bfrac
                gfrac = np.minimum(G, 0.9)
                busy = np.minimum(C, demand / dth / (1 - gfrac))
                served = busy * (1 - gfrac) * dth
                dfrac = np.minimum(RW, 1.0)
                disc = busy * dfrac; prod = busy - disc
                deg = prod * gfrac; use = prod - deg
                keys["idle"][("low_demand", "tenant")] += C - busy
            def split(fate, total_dev, parts, total_frac):
                tf = np.where(total_frac > 0, total_dev / np.maximum(total_frac, 1e-12), 0)
                for k, arr in parts.items():
                    keys[fate][k] += arr * tf
            split("blocked", blocked, blockf, B)
            split("discarded", disc, rework, RW)
            split("degraded", deg, degr, G)
            keys["useful"][("-", "-")] += use
            # to device-hours
            out = {}
            for f in FATES:
                for k in keys[f]:
                    keys[f][k] = keys[f][k] * dth
                out[f] = sum(keys[f].values()) if keys[f] else np.zeros(T)
            out["installed"] = np.full(T, N * dth)
            out["demand"] = demand; out["served"] = served
            res[jb.job_id] = out; attrib[jb.job_id] = keys
        self.res, self.attrib, self.interrupts = res, attrib, pd.DataFrame(interrupts)
        self._power()
        return self

    def _power(self):
        cal, T, dth = self.calib, self.T, self.dth
        gens = self.cfg["generations"]
        a = cal["power_util_exponent"]; net = cal["network_overhead_frac"]; hif = cal["host_idle_frac"]
        site_it = defaultdict(lambda: np.zeros(T)); run_kw = {}
        for jb in self.jobs.itertuples():
            g = gens[jb.gen]; r = self.res[jb.job_id]
            pi = idle_w(jb.gen, g, cal)
            u = cal["train_running_util"] if jb.workload == "training" else cal["inference_busy_util"]
            prun = pi + (g["tdp_w"] - pi) * u ** a + g["host_w"]
            pidle = pi + g["host_w"] * hif
            running = (r["useful"] + r["degraded"] + r["discarded"]) / dth
            parked = (r["idle"] + r["blocked"]) / dth
            kw_run = running * prun * (1 + net) / 1e3; kw_idle = parked * pidle * (1 + net) / 1e3
            r["it_kw"] = kw_run + kw_idle; run_kw[jb.job_id] = (kw_run, running * (prun - pi - g["host_w"] * hif) * (1 + net) / 1e3)
            site_it[jb.site] += r["it_kw"]
        lt_h = self.t_min / 60
        rows = []
        self.site_series = {}
        for sid, s in self.sites.items():
            amb = s["ambient_c"] + cal["ambient_diurnal_amp_c"] * np.sin(2 * np.pi * ((lt_h + s["utc_offset_h"]) % 24 - 9) / 24)
            pue = s["pue"] + cal["pue_ambient_sensitivity"] * (amb - s["ambient_c"])
            capf = np.full(T, np.inf); curt_cause = np.full(T, "", dtype=object)
            for e in self.site_ev.get(sid, []):
                cv = cover(e["t0"], e["dur"], self.cfg["sim"]["interval_min"], T)
                if cv is None: continue
                if e["kind"] == "pue":
                    pue[cv[0]:cv[1]] += e["value"] * cv[2]
                elif e["kind"] == "powercap":
                    capf[cv[0]:cv[1]] = np.minimum(capf[cv[0]:cv[1]], e["value"])
            nameplate = self.nameplate_site_mw[sid]
            fac = site_it[sid] / 1e3 * pue
            cap_mw = capf * nameplate
            over = np.clip(fac - cap_mw, 0, None)
            k_site = np.ones(T)
            if over.max() > 0:
                # reduce dynamic power of running devices to fit under the cap
                dyn = sum(run_kw[j][1] for j in self.jobs.job_id[self.jobs.site == sid]) / 1e3  # MW dynamic
                need = over / pue
                k_site = np.clip(1 - need / np.maximum(dyn, 1e-9), 0.2, 1.0)
                thr = k_site ** cal["throughput_power_exponent"]
                for jb in self.jobs[self.jobs.site == sid].itertuples():
                    r = self.res[jb.job_id]; keys = self.attrib[jb.job_id]
                    moved = r["useful"] * (1 - thr)
                    keys["useful"][("-", "-")] -= moved; r["useful"] = r["useful"] - moved
                    keys["degraded"][("grid_power_cap", "grid")] += moved; r["degraded"] = r["degraded"] + moved
                    cut = run_kw[jb.job_id][1] * (1 - k_site)
                    r["it_kw"] = r["it_kw"] - cut
                    site_it[sid] = site_it[sid] - cut
                fac = site_it[sid] / 1e3 * pue
            self.site_series[sid] = dict(it_mw=site_it[sid] / 1e3, pue=pue, facility_mw=fac, cap_mw=np.where(np.isinf(cap_mw), np.nan, cap_mw),
                                         curtailed_mw=over, power_cap_factor=k_site, ambient_c=amb, nameplate_mw=np.full(T, nameplate))

def _ar1(rng, T, phi=0.98):
    x = np.zeros(T); e = rng.standard_normal(T) * np.sqrt(1 - phi ** 2)
    for i in range(1, T):
        x[i] = phi * x[i - 1] + e[i]
    return x
