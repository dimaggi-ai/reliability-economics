"""Orchestrate a run: fleet -> events (baseline + chaos) -> fates -> telemetry -> rollups -> files."""
import json, os, time
import numpy as np
import pandas as pd
from .config import Calib, load_config, load_library, load_yaml
from .fleet import build_fleet, build_jobs, assets_table
from .events import EventSink, baseline, Chaos, random_schedule
from .sim import Sim, FATES
from . import telemetry

def make_world(cfg, calib, seed, chaos_schedule=None, library=None, chaos=True):
    rng = np.random.default_rng(seed)
    bom, pods, target = build_fleet(cfg, calib, rng)
    jobs, pods = build_jobs(cfg, pods, rng)
    H = cfg["sim"]["days"] * 1440
    sink = EventSink()
    baseline(cfg, calib, pods, np.random.default_rng([seed, 1]), sink, H)
    sched = []
    if chaos and chaos_schedule:
        library = library or load_library()
        ch = Chaos(cfg, calib, pods, library, np.random.default_rng([seed, 2]), sink, H)
        for sc in chaos_schedule:
            ch.apply(sc["use"], sc["start_h"], sc.get("target", {}), sc.get("override"))
            sched.append(sc)
    sim = Sim(cfg, calib, pods, jobs, sink.ev, np.random.default_rng([seed, 3]))
    sim.nameplate_site_mw = bom.groupby("site").facility_mw.sum().to_dict()
    sim.run()
    return dict(rng=rng, bom=bom, pods=pods, jobs=jobs, events=sink.ev, sim=sim, target=target, schedule=sched)

def events_df(events):
    ev = pd.DataFrame([{k: v for k, v in e.items() if k != "segs"} for e in events])
    seg = pd.DataFrame([(e["event_id"], p, o, c) for e in events for (p, o, c) in e["segs"]], columns=["event_id", "pod_id", "dev_offset", "dev_count"])
    if len(ev):
        ev["devices"] = ev.event_id.map(seg.groupby("event_id").dev_count.sum()).fillna(0).astype(int)
    return ev, seg

def fates_df(cfg, W):
    sim = W["sim"]; ts = pd.Timestamp(cfg["sim"]["start"]) + pd.to_timedelta(sim.t_min, unit="min")
    sites = {s["id"]: s for s in cfg["sites"]}
    parts = []
    for jb in W["jobs"].itertuples():
        r = sim.res[jb.job_id]; ss = sim.site_series[jb.site]
        d = dict(ts=ts, job_id=np.int32(jb.job_id), site=jb.site, gen=jb.gen, workload=jb.workload)
        for f in ["installed"] + FATES + ["demand", "served"]:
            d[f + "_dh"] = r[f].astype(np.float32)
        d["it_kw"] = r["it_kw"].astype(np.float32)
        d["facility_kw"] = (r["it_kw"] * ss["pue"]).astype(np.float32)
        parts.append(pd.DataFrame(d))
    df = pd.concat(parts, ignore_index=True)
    for c in ("site", "gen", "workload"): df[c] = df[c].astype("category")
    return df

def attribution_hourly(cfg, W):
    sim = W["sim"]; per_h = int(60 // sim.dt); H = sim.T // per_h
    hours = pd.Timestamp(cfg["sim"]["start"]) + pd.to_timedelta(np.arange(H), unit="h")
    rows = []
    for jb in W["jobs"].itertuples():
        for fate, kd in sim.attrib[jb.job_id].items():
            for (cause, party), arr in kd.items():
                h = arr[:H * per_h].reshape(H, per_h).sum(1)
                if h.sum() <= 1e-9: continue
                rows.append(pd.DataFrame(dict(hour=hours, job_id=np.int32(jb.job_id), site=jb.site, gen=jb.gen, workload=jb.workload,
                                              fate=fate, cause=cause, party=party, device_hours=h.astype(np.float32))))
    df = pd.concat(rows, ignore_index=True)
    df = df[df.device_hours > 0]
    for c in ("site", "gen", "workload", "fate", "cause", "party"): df[c] = df[c].astype("category")
    return df.reset_index(drop=True)

def site_power_df(cfg, W):
    sim = W["sim"]; ts = pd.Timestamp(cfg["sim"]["start"]) + pd.to_timedelta(sim.t_min, unit="min")
    return pd.concat([pd.DataFrame(dict(ts=ts, site=s, **{k: v.astype(np.float32) for k, v in d.items()})) for s, d in sim.site_series.items()], ignore_index=True)

def rollups(cfg, W, fates, attr, ev):
    ob = cfg["obligations"]
    fates = fates.copy(); fates["month"] = fates.ts.dt.strftime("%Y-%m")
    g = fates.groupby(["month", "site", "gen", "workload"], observed=True)
    cols = [c for c in fates.columns if c.endswith("_dh")]
    m = g[cols].sum()
    m["it_mwh"] = g.it_kw.sum() * W["sim"].dth / 1e3; m["facility_mwh"] = g.facility_kw.sum() * W["sim"].dth / 1e3
    m = m.reset_index()
    a = attr.copy(); a["month"] = a.hour.dt.strftime("%Y-%m")
    pdown = a[(a.fate == "down") & (a.cause == "planned_maintenance")].groupby(["month", "site", "gen", "workload"], observed=True).device_hours.sum()
    m = m.merge(pdown.rename("planned_down_dh").reset_index(), how="left", on=["month", "site", "gen", "workload"]).fillna({"planned_down_dh": 0})
    deployed = m.installed_dh - m.undelivered_dh
    m["usable_frac"] = m.useful_dh / m.installed_dh
    m["availability"] = 1 - (m.down_dh - m.planned_down_dh) / deployed
    m["ettr"] = np.where(m.workload == "training", m.useful_dh / (m.useful_dh + m.degraded_dh + m.discarded_dh + m.blocked_dh), np.nan)
    m["served_frac"] = np.where(m.workload == "inference", m.served_dh / np.maximum(m.demand_dh, 1e-9), np.nan)
    # MTBF / MTTR by site x gen from unplanned down events
    e = ev[(ev.kind == "down") & (~ev.planned) & (ev.t0 >= 0)].copy()
    e["month"] = (pd.Timestamp(cfg["sim"]["start"]) + pd.to_timedelta(e.t0, unit="min")).dt.strftime("%Y-%m")
    eg = e.groupby(["month", "site", "gen"]).agg(unplanned_failures=("event_id", "size"), mttr_h=("dur", lambda x: x.mean() / 60), mttr_median_h=("dur", lambda x: x.median() / 60)).reset_index()
    sg = m.groupby(["month", "site", "gen"]).agg(installed_dh=("installed_dh", "sum")).reset_index().merge(eg, how="left", on=["month", "site", "gen"])
    sg["mtbf_device_h"] = sg.installed_dh / sg.unplanned_failures
    m = m.merge(sg[["month", "site", "gen", "unplanned_failures", "mttr_h", "mttr_median_h", "mtbf_device_h"]], on=["month", "site", "gen"], how="left")
    # obligations / shortfall attribution
    sla = []
    for r in m.itertuples():
        sel = a[(a.month == r.month) & (a.site == r.site) & (a.gen == r.gen) & (a.workload == r.workload)]
        def by_party(mask):
            return sel[mask(sel)].groupby("party", observed=True).device_hours.sum()
        checks = [("availability", ob["availability_target"], r.availability, (ob["availability_target"] - r.availability) * (r.installed_dh - r.undelivered_dh),
                   lambda s: (s.fate == "down") & (s.cause != "planned_maintenance"))]
        if r.workload == "training":
            checks.append(("training_useful", ob["training_useful_target"], r.usable_frac, ob["training_useful_target"] * r.installed_dh - r.useful_dh, lambda s: s.fate != "useful"))
        else:
            checks.append(("inference_served", ob["inference_served_target"], r.served_frac, ob["inference_served_target"] * r.demand_dh - r.served_dh,
                           lambda s: s.fate.isin(["down", "blocked", "degraded", "undelivered", "discarded"])))
        for name, tgt, actual, short, mask in checks:
            if short <= 0: continue
            bp = by_party(mask); tot = bp.sum()
            for party, v in bp.items():
                if v <= 0: continue
                sla.append(dict(month=r.month, site=r.site, gen=r.gen, workload=r.workload, obligation=name, target=tgt, actual=actual,
                                shortfall_dh=short, party=party, party_share=v / tot, attributed_shortfall_dh=short * v / tot,
                                credit_usd=short * v / tot * ob["credit_usd_per_shortfall_device_hour"]))
    return m, pd.DataFrame(sla)

def daily_site(fates, attr):
    f = fates.copy(); f["day"] = f.ts.dt.floor("D")
    d = f.groupby(["day", "site"], observed=True)[["installed_dh", "useful_dh", "degraded_dh", "down_dh", "undelivered_dh", "blocked_dh", "discarded_dh", "idle_dh"]].sum().reset_index()
    d["usable_pct"] = 100 * (d.installed_dh - d.down_dh - d.undelivered_dh) / d.installed_dh
    d["delivered_pct"] = 100 * d.useful_dh / d.installed_dh
    return d

def run(cfg_path, out_dir, seed=None, chaos_path=None, overrides=None, chaos=True, telemetry_on=True, write=True, mc_schedule=None):
    t_start = time.time()
    cfg = load_config(cfg_path, overrides); calib = Calib()
    if seed is not None: cfg["sim"]["seed"] = seed
    seed = cfg["sim"]["seed"]
    sched = mc_schedule if mc_schedule is not None else (load_yaml(chaos_path)["scenarios"] if (chaos and chaos_path) else [])
    W = make_world(cfg, calib, seed, sched, chaos=chaos)
    ev, seg = events_df(W["events"])
    fates = fates_df(cfg, W); attr = attribution_hourly(cfg, W)
    roll, sla = rollups(cfg, W, fates, attr, ev)
    out = dict(W=W, cfg=cfg, calib=calib, ev=ev, seg=seg, fates=fates, attr=attr, roll=roll, sla=sla)
    if not write:
        return out
    os.makedirs(out_dir, exist_ok=True)
    pq = lambda df, n: df.to_parquet(os.path.join(out_dir, n + ".parquet"), index=False, compression="zstd")
    W["bom"].to_csv(os.path.join(out_dir, "bom.csv"), index=False)
    pq(W["pods"], "pods"); pq(W["jobs"].assign(pod_ids=W["jobs"].pod_ids.map(json.dumps)), "jobs")
    pq(ev, "events"); pq(seg, "event_segments"); pq(W["sim"].interrupts, "interrupts")
    pq(fates, "fates"); pq(attr, "fate_attribution_hourly"); pq(roll, "rollups_monthly"); pq(sla, "sla_shortfall")
    sp = site_power_df(cfg, W); pq(sp, "site_power")
    ds = daily_site(fates, attr); ds.to_csv(os.path.join(out_dir, "daily_site.csv"), index=False)
    roll.to_csv(os.path.join(out_dir, "rollups_monthly.csv"), index=False); sla.to_csv(os.path.join(out_dir, "sla_shortfall.csv"), index=False)
    if telemetry_on:
        rng = np.random.default_rng([seed, 4])
        assets = assets_table(cfg, W["pods"]); pq(assets, "assets")
        ids = telemetry.sample_devices(cfg, W["pods"], rng)
        dev = telemetry.device_series(cfg, calib, W["pods"], W["jobs"], W["sim"], W["events"], ids, rng); pq(dev, "devices_sample")
        net = telemetry.network(cfg, calib, W["pods"], W["sim"], W["events"], rng); pq(net, "network")
        n = cfg["outputs"]["csv_sample_rows"]
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        for name, df in dict(devices_sample=dev, network=net, fates=fates, fate_attribution_hourly=attr, site_power=sp, events=ev).items():
            df.sample(min(n, len(df)), random_state=0).sort_index().to_csv(os.path.join(out_dir, "samples", name + ".csv"), index=False)
        out.update(dev=dev, net=net, assets=assets)
    meta = dict(seed=seed, config=cfg_path, chaos=chaos_path if chaos else None, schedule=W["schedule"], target_facility_mw=W["target"],
                bom_facility_mw=float(W["bom"].facility_mw.sum()), devices=int(W["bom"].devices.sum()), jobs=len(W["jobs"]), events=len(ev),
                intervals=W["sim"].T, interval_min=W["sim"].dt, runtime_s=round(time.time() - t_start, 1))
    json.dump(meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=2, default=str)
    out["meta"] = meta
    return out
