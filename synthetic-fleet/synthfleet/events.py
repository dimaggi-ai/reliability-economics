"""Baseline fault process and chaos engine. Output: an event list; each event covers device segments of pods.

Event fields: event_id, kind, source, scenario, cause, party, planned, interrupt, site, gen, t0 (min), dur (min),
value, job_wide, delay (min), segs = [(pod_id, offset, count)].
kind in {down, block, degrade, sdc, powercap, undelivered, demand, pue}
"""
import numpy as np

PARTY = {"gpu": "vendor", "unknown": "unattributed", "cable": "operator", "fan": "vendor", "nic": "vendor", "psu": "vendor",
         "software": "operator", "tenant_task": "tenant", "change": "operator", "motherboard_cpu": "vendor", "other_hw": "vendor",
         "tpu_machine": "vendor", "ici_cable": "operator", "ocs": "vendor", "planned_maintenance": "operator"}

class EventSink:
    def __init__(self):
        self.ev = []
    def add(self, **kw):
        kw.setdefault("value", 1.0); kw.setdefault("job_wide", False); kw.setdefault("delay", 0.0)
        kw.setdefault("planned", False); kw.setdefault("interrupt", False); kw.setdefault("segs", [])
        kw.setdefault("source", "baseline"); kw.setdefault("scenario", "")
        kw["event_id"] = len(self.ev)
        self.ev.append(kw)
        return kw

def _lognorm(rng, median, sigma, size=None):
    return median * np.exp(sigma * rng.standard_normal(size))

def node_seg(pod, g, node_idx, k=1):
    dpn = g["dev_per_node"]
    off = node_idx * dpn
    cnt = min(k * dpn, int(pod["n_devices"]) - off)
    return (int(pod["pod_id"]), int(off), int(cnt))

def gen_rate(gen, g, calib):
    if g["family"] == "tpu":
        mult = calib[g["rate_mult_param"]] if g.get("rate_mult_param") else 1.0
        return calib["tpuv4_machine_rate"] * mult  # per machine-day
    return calib[g["rate_param"]]  # per device-day

def baseline(cfg, calib, pods, rng, sink, horizon_min, rate_mult_fn=None):
    gens = cfg["generations"]
    mix = calib["node_fault_class_mix"]; classes = list(mix); probs = np.array([mix[c] for c in classes]); probs /= probs.sum()
    med = calib["mttr_lognormal_median_days"]; sig = calib["mttr_lognormal_sigma"]; mx = calib["mttr_max_days"]
    W = calib["warmup_days"] * 1440  # pre-roll: faults starting before t=0 that are still under repair
    days = (horizon_min + W) / 1440
    t_lo = -W
    for (site, gen), sub in pods.groupby(["site", "gen"], sort=False):
        g = gens[gen]; recs = sub.to_dict("records")
        nodes_per_pod = np.array([int(np.ceil(r["n_devices"] / g["dev_per_node"])) for r in recs])
        n_nodes = int(nodes_per_pod.sum()); cum = np.cumsum(nodes_per_pod)
        def pick_node():
            i = int(rng.integers(n_nodes)); p = int(np.searchsorted(cum, i, side="right"))
            return recs[p], i - (cum[p] - nodes_per_pod[p])
        if g["family"] == "tpu":
            r_node = gen_rate(gen, g, calib)
            mult = calib[g["rate_mult_param"]] if g.get("rate_mult_param") else 1.0
            n = rng.poisson(r_node * n_nodes * days)
            for _ in range(n):
                pod, ni = pick_node()
                d = min(_lognorm(rng, calib["tpu_mttr_median_days"], calib["tpu_mttr_sigma"]), mx) * 1440
                sink.add(kind="down", cause="tpu_machine", party="vendor", site=site, gen=gen, t0=rng.uniform(t_lo, horizon_min), dur=d,
                         interrupt=True, segs=[node_seg(pod, g, ni)])
            # ICI cable faults: the cable's cube (rack) is unusable until repaired
            chips = int(sub.n_devices.sum()); rack = g["dev_per_node"] * g["nodes_per_rack"]
            n = rng.poisson(calib["tpuv4_ici_cable_rate"] * mult * calib["tpu_ici_cables_per_chip"] * chips * days)
            for _ in range(n):
                pod, ni = pick_node(); r = ni // g["nodes_per_rack"]
                cnt = min(rack, int(pod["n_devices"]) - r * rack)
                d = min(_lognorm(rng, calib["tpu_mttr_median_days"], calib["tpu_mttr_sigma"]), mx) * 1440
                sink.add(kind="down", cause="ici_cable", party="operator", site=site, gen=gen, t0=rng.uniform(t_lo, horizon_min), dur=d,
                         interrupt=True, segs=[(int(pod["pod_id"]), r * rack, cnt)])
            if "OCS" in g["fabric"]:
                n_ocs = calib["tpu_ocs_per_4096_chips"] * chips / 4096
                n = rng.poisson(calib["tpuv4_ocs_rate"] * mult * n_ocs * days)
                for _ in range(n):
                    pod, _ni = pick_node()
                    _ocs_effect(sink, calib, rng, pod, site, gen, rng.uniform(t_lo, horizon_min), _lognorm(rng, 4 * 60, 0.6), "ocs", "vendor", "baseline", "")
        else:
            r_dev = gen_rate(gen, g, calib)
            n = rng.poisson(r_dev * g["dev_per_node"] * n_nodes * days)
            cls = rng.choice(len(classes), size=n, p=probs)
            for c in cls:
                cname = classes[c]; pod, ni = pick_node()
                k = int(rng.choice([2, 3])) if rng.random() < calib["multi_node_fault_prob"] else 1
                ni = min(ni, max(0, int(np.ceil(pod["n_devices"] / g["dev_per_node"])) - k))
                d = min(_lognorm(rng, med[cname], sig), mx) * 1440 * calib["mttr_scale_by_gen"].get(gen, 1.0)
                sink.add(kind="down", cause=cname, party=PARTY[cname], site=site, gen=gen, t0=rng.uniform(t_lo, horizon_min), dur=d,
                         interrupt=True, segs=[node_seg(pod, g, ni, k)], nodes=k)
        # planned maintenance (drains a node; interrupts the job like Llama 3's planned interruptions)
        unplanned = sum(1 for e in sink.ev if e["site"] == site and e["gen"] == gen and e["kind"] == "down" and not e["planned"] and e["source"] == "baseline")
        n = rng.poisson(calib["planned_maintenance_ratio"] * unplanned)
        for _ in range(n):
            pod, ni = pick_node()
            sink.add(kind="down", cause="planned_maintenance", party="operator", planned=True, site=site, gen=gen,
                     t0=rng.uniform(t_lo, horizon_min), dur=_lognorm(rng, calib["planned_maintenance_median_h"] * 60, 0.5),
                     interrupt=True, segs=[node_seg(pod, g, ni)])

def _ocs_effect(sink, calib, rng, pod, site, gen, t0, dur, cause, party, source, scen):
    seg = [(int(pod["pod_id"]), 0, int(pod["n_devices"]))]
    if rng.random() < calib["tpu_ft_routing_optin"]:
        sink.add(kind="degrade", cause=cause, party=party, site=site, gen=gen, t0=t0, dur=dur, job_wide=True,
                 value=1 - calib["tpu_ocs_ft_routing_throughput"], segs=seg, source=source, scenario=scen)
    else:
        sink.add(kind="block", cause=cause, party=party, site=site, gen=gen, t0=t0, dur=dur, value=1.0, segs=seg,
                 interrupt=True, source=source, scenario=scen)

class Chaos:
    def __init__(self, cfg, calib, pods, library, rng, sink, horizon_min):
        self.cfg, self.calib, self.pods, self.lib, self.rng, self.sink, self.H = cfg, calib, pods, library, rng, sink, horizon_min
        self.gens = cfg["generations"]

    def _pods(self, tgt, need_ocs=False):
        p = self.pods
        if "site" in tgt: p = p[p.site == tgt["site"]]
        if "gen" in tgt: p = p[p.gen == tgt["gen"]]
        if "hall" in tgt: p = p[p.hall == tgt["hall"]]
        if need_ocs: p = p[p.gen.map(lambda x: "OCS" in self.gens[x]["fabric"])]
        return p

    def apply(self, name, start_h, target, overrides=None, scen_label=None):
        spec = dict(self.lib[name]); spec.update(overrides or {})
        t0 = start_h * 60; rng = self.rng; S = self.sink
        label = scen_label or name
        base = dict(source="chaos", scenario=label, cause=spec.get("cause", name), party=spec.get("party", "operator"))
        typ = spec["type"]
        P = self._pods(target, need_ocs=(typ == "ocs"))
        if len(P) == 0:
            return
        site = target.get("site", P.site.iloc[0])
        if typ == "node_burst":
            for (gen, sub) in P.groupby("gen"):
                g = self.gens[gen]
                nn = int(sub.n_devices.sum() / g["dev_per_node"])
                k = spec.get("nodes") or max(1, int(spec["node_frac"] * nn))
                for _ in range(k):
                    pod = sub.sample(1, random_state=rng.integers(1 << 31)).iloc[0]
                    ni = int(rng.integers(int(np.ceil(pod.n_devices / g["dev_per_node"]))))
                    S.add(kind="down", site=pod.site, gen=gen, t0=t0 + rng.uniform(0, spec["window_h"] * 60),
                          dur=_lognorm(rng, spec["repair_median_h"] * 60, 0.8), interrupt=True, segs=[node_seg(pod, g, ni)], **base)
                if spec.get("nodes"): break
        elif typ == "rack_down":
            for _ in range(spec["racks"]):
                pod = P.sample(1, random_state=rng.integers(1 << 31)).iloc[0]; g = self.gens[pod.gen]
                rack = g["dev_per_node"] * g["nodes_per_rack"]; nr = int(np.ceil(pod.n_devices / rack)); r = int(rng.integers(nr))
                S.add(kind="down", site=pod.site, gen=pod.gen, t0=t0, dur=spec["duration_h"] * 60, interrupt=True,
                      segs=[(int(pod.pod_id), r * rack, int(min(rack, pod.n_devices - r * rack)))], **base)
        elif typ in ("hall_down", "site_down", "thermal"):
            if typ != "site_down":
                hall = target.get("hall") or rng.choice(sorted(P.hall.unique()))
                P = P[P.hall == hall]
            dur = spec["duration_h"] * 60
            for pod in P.itertuples():
                seg = [(int(pod.pod_id), 0, int(pod.n_devices))]
                if typ == "thermal":
                    S.add(kind="degrade", site=pod.site, gen=pod.gen, t0=t0, dur=dur, job_wide=False, value=spec["throttle"], segs=seg, **base)
                    g = self.gens[pod.gen]; nn = int(np.ceil(pod.n_devices / g["dev_per_node"]))
                    for ni in rng.choice(nn, size=max(1, int(spec["trip_frac"] * nn)), replace=False):
                        S.add(kind="down", site=pod.site, gen=pod.gen, t0=t0 + rng.uniform(0, dur / 2), dur=dur, interrupt=True,
                              segs=[node_seg(pod._asdict(), g, int(ni))], **base)
                else:
                    S.add(kind="down", site=pod.site, gen=pod.gen, t0=t0, dur=dur, interrupt=True, segs=seg, **base)
            if typ == "thermal":
                S.add(kind="pue", site=site, gen="", t0=t0, dur=dur, value=spec["pue_delta"], **base)
        elif typ == "power_cap":
            S.add(kind="powercap", site=site, gen="", t0=t0, dur=spec["duration_h"] * 60, value=spec["cap_frac"], **base)
        elif typ == "ocs":
            for pod in P.sample(min(spec["pods"], len(P)), random_state=rng.integers(1 << 31)).itertuples():
                _ocs_effect(S, self.calib, rng, pod._asdict(), pod.site, pod.gen, t0, spec["duration_h"] * 60, base["cause"], base["party"], "chaos", label)
        elif typ == "cable":
            for _ in range(spec["count"]):
                pod = P.sample(1, random_state=rng.integers(1 << 31)).iloc[0]; g = self.gens[pod.gen]
                tt = t0 + rng.uniform(0, spec["window_h"] * 60); d = _lognorm(rng, spec["repair_median_h"] * 60, 0.8)
                if g["family"] == "tpu":
                    rack = g["dev_per_node"] * g["nodes_per_rack"]; nr = int(np.ceil(pod.n_devices / rack)); r = int(rng.integers(nr))
                    seg = (int(pod.pod_id), r * rack, int(min(rack, pod.n_devices - r * rack))); cause = "ici_cable"
                else:
                    ni = int(rng.integers(int(np.ceil(pod.n_devices / g["dev_per_node"])))); seg = node_seg(pod, g, ni); cause = "cable"
                S.add(kind="down", site=pod.site, gen=pod.gen, t0=tt, dur=d, interrupt=True, segs=[seg], **{**base, "cause": cause})
        elif typ == "partition":
            sub = P.sample(max(1, int(spec["pod_frac"] * len(P))), random_state=rng.integers(1 << 31))
            for pod in sub.itertuples():
                S.add(kind="block", site=pod.site, gen=pod.gen, t0=t0, dur=spec["duration_h"] * 60, value=1.0, interrupt=True,
                      segs=[(int(pod.pod_id), 0, int(pod.n_devices))], **base)
        elif typ == "sdc":
            pod = P.sample(1, random_state=rng.integers(1 << 31)).iloc[0]; g = self.gens[pod.gen]
            ni = int(rng.integers(int(np.ceil(pod.n_devices / g["dev_per_node"]))))
            delay = spec["detect_delay_h"] * 60
            S.add(kind="sdc", site=pod.site, gen=pod.gen, t0=t0, dur=delay, delay=delay, interrupt=True, segs=[(int(pod.pod_id), 0, int(pod.n_devices))], **base)
            S.add(kind="down", site=pod.site, gen=pod.gen, t0=t0 + delay, dur=_lognorm(rng, spec["repair_median_h"] * 60, 0.5),
                  segs=[node_seg(pod, g, ni)], **base)
        elif typ == "straggler":
            pods_with_train = P  # job-wide slowdown applied to the job owning the chosen pod
            for pod in pods_with_train.sample(min(spec["jobs"], len(P)), random_state=rng.integers(1 << 31)).itertuples():
                S.add(kind="degrade", site=pod.site, gen=pod.gen, t0=t0, dur=spec["duration_h"] * 60, job_wide=True, value=spec["slowdown"],
                      segs=[(int(pod.pod_id), 0, int(pod.n_devices))], **base)
        elif typ == "firmware":
            gen = target.get("gen") or P.gen.iloc[0]; Pg = P[P.gen == gen]; g = self.gens[gen]
            rate = gen_rate(gen, g, self.calib) * (g["dev_per_node"] if g["family"] != "tpu" else 1) * (spec["failure_mult"] - 1)
            for w in range(spec["waves"]):
                tw = t0 + w * spec["wave_gap_h"] * 60
                for pod in Pg.itertuples():
                    nn = int(np.ceil(pod.n_devices / g["dev_per_node"]))
                    for ni in rng.choice(nn, size=max(1, int(spec["wave_frac"] * nn)), replace=False):
                        S.add(kind="down", site=pod.site, gen=gen, t0=tw + rng.uniform(0, 30), dur=spec["reboot_min"], interrupt=True,
                              segs=[node_seg(pod._asdict(), g, int(ni))], **{**base, "cause": "firmware_reboot"})
            # elevated failure rate on upgraded nodes until rollback
            frac_upgraded = min(1.0, spec["waves"] * spec["wave_frac"])
            nn_total = int(Pg.n_devices.sum() / g["dev_per_node"]) * frac_upgraded
            n = rng.poisson(rate * nn_total * spec["elevated_h"] / 24)
            for _ in range(n):
                pod = Pg.sample(1, random_state=rng.integers(1 << 31)).iloc[0]
                ni = int(rng.integers(int(np.ceil(pod.n_devices / g["dev_per_node"]))))
                S.add(kind="down", site=pod.site, gen=gen, t0=t0 + rng.uniform(0, spec["elevated_h"] * 60), dur=_lognorm(rng, 6 * 60, 0.7),
                      interrupt=True, segs=[node_seg(pod, g, ni)], **{**base, "cause": "firmware_fault"})
        elif typ == "undelivered":
            for gen, sub in P.groupby("gen"):
                k = max(1, int(round(spec["pod_frac"] * len(sub))))
                for pod in sub.iloc[-k:].itertuples():  # last pods in the build sequence arrive late
                    S.add(kind="undelivered", site=pod.site, gen=gen, t0=t0, dur=spec["duration_h"] * 60, segs=[(int(pod.pod_id), 0, int(pod.n_devices))], **base)
        elif typ == "demand":
            S.add(kind="demand", site=site, gen=target.get("gen", ""), t0=t0, dur=spec["duration_h"] * 60, value=spec["mult"], **base)
        elif typ == "cascade":
            sites = [s["id"] for s in self.cfg["sites"] if s["id"] != site]
            rng.shuffle(sites); used = 0
            for st in spec["steps"]:
                ov = {k: v for k, v in st.items() if k not in ("use", "lag_h", "same_site", "other_sites")}
                if st.get("same_site"):
                    self.apply(st["use"], start_h + st["lag_h"], {"site": site}, ov, scen_label=label)
                else:
                    for s2 in sites[used:used + st["other_sites"]]:
                        self.apply(st["use"], start_h + st["lag_h"], {"site": s2}, ov, scen_label=label)
                    used = (used + st["other_sites"]) % len(sites)
        else:
            raise ValueError(f"unknown scenario type {typ}")

def random_schedule(cfg, library, rng, horizon_h, n_scenarios):
    """Monte Carlo: sample scenarios, start times and targets."""
    names = sorted(library)
    out = []
    for _ in range(n_scenarios):
        nm = names[int(rng.integers(len(names)))]; spec = library[nm]
        if spec["type"] == "ocs":
            sites = [s for s in cfg["sites"] if any("OCS" in cfg["generations"][g]["fabric"] for g in s["mix"])]
        else:
            sites = cfg["sites"]
        s = sites[int(rng.integers(len(sites)))]
        tgt = {"site": s["id"]}
        if spec["type"] in ("node_burst", "firmware", "undelivered", "sdc", "straggler", "cable") and rng.random() < 0.7:
            tgt["gen"] = list(s["mix"])[int(rng.integers(len(s["mix"])))]
        dur = spec.get("duration_h", spec.get("window_h", 12))
        out.append({"use": nm, "start_h": float(rng.uniform(0, max(1, horizon_h - min(dur, horizon_h / 2)))), "target": tgt})
    return out
