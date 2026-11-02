"""Observable telemetry: sampled per-device time series, per-pod network counters, site power."""
from collections import defaultdict
import numpy as np
import pandas as pd
from .fleet import idle_w
from .sim import cover

ECC_MULT = {"A100": 1.0, "H100": None, "H200": None}  # None -> calibrated Delta ratio

def sample_devices(cfg, pods, rng):
    sim = cfg["sim"]; out = []
    for (site, gen), sub in pods.groupby(["site", "gen"]):
        n = int(sub.n_devices.sum())
        k = max(sim["device_sample_min_per_group"], int(round(sim["device_sample_frac"] * n)))
        ids = np.concatenate([p.dev_start + np.arange(p.n_devices) for p in sub.itertuples()])
        out.append(rng.choice(ids, size=min(k, n), replace=False))
    ids = np.concatenate(out)
    if len(ids) > sim["device_sample_max"]:
        ids = rng.choice(ids, size=sim["device_sample_max"], replace=False)
    return np.sort(ids)

def _code(vendor, cause, rng):
    if cause in ("gpu", "gpu_hw_batch", "unknown", "firmware_fault", "tpu_machine"):
        if vendor == "NVIDIA": return "XID" + str(rng.choice([48, 79, 94, 95, 119, 63]))
        if vendor == "AMD": return "RAS_UE"
        if vendor == "Google": return "TPU_HW"
        return "NEURON_HW"
    if cause in ("nic", "cable", "ici_cable"): return "LINK_DOWN"
    if cause in ("psu", "rack_pdu", "hall_power", "utility_outage"): return "POWER_LOSS"
    if cause == "fan": return "FAN_FAIL"
    if cause in ("cooling",): return "THERMAL_TRIP"
    if cause == "firmware_reboot": return "FW_REBOOT"
    return "NODE_DOWN"

def device_series(cfg, calib, pods, jobs, sim, events, sample_ids, rng):
    gens = cfg["generations"]; sites = {s["id"]: s for s in cfg["sites"]}
    T, dt = sim.T, sim.dt; ts = pd.Timestamp(cfg["sim"]["start"]) + pd.to_timedelta(sim.t_min, unit="min")
    pod_ev = defaultdict(list)
    for e in events:
        for (p, off, c) in e["segs"]:
            pod_ev[p].append((e, off, c))
    podrow = pods.set_index("pod_id"); jobrow = jobs.set_index("job_id")
    starts = pods.dev_start.values; pids = pods.pod_id.values
    a = calib["power_util_exponent"]
    parts = []
    for d in sample_ids:
        pi_ = int(np.searchsorted(starts, d, side="right") - 1); pid = int(pids[pi_]); P = podrow.loc[pid]; o = int(d - P.dev_start)
        g = gens[P.gen]; s = sites[P.site]; jb = jobrow.loc[int(P.job_id)]; r = sim.res[int(P.job_id)]
        down = np.zeros(T); und = np.zeros(T); hot = np.zeros(T); dgr = np.zeros(T)
        cause = np.full(T, "", dtype=object); party = np.full(T, "", dtype=object); planned = np.zeros(T, bool)
        code = np.full(T, "", dtype=object); uce = np.zeros(T, np.int16)
        for e, off, c in pod_ev[pid]:
            if not (off <= o < off + c): continue
            cv = cover(e["t0"], e["dur"], dt, T)
            if cv is None: continue
            sl = slice(cv[0], cv[1])
            if e["kind"] == "down":
                down[sl] += cv[2]; m = cv[2] >= 0.5; idx = np.arange(cv[0], cv[1])[m]
                cause[idx] = e["cause"]; party[idx] = e["party"]; planned[idx] = e["planned"]
                i0 = cv[0]
                if not e["planned"] and e["t0"] >= 0:
                    code[i0] = _code(g["vendor"], e["cause"], rng)
                    if e["cause"] in ("gpu", "gpu_hw_batch") and rng.random() < 0.3: uce[i0] = 1
            elif e["kind"] == "undelivered":
                und[sl] += cv[2]; cause[sl] = e["cause"]; party[sl] = e["party"]
            elif e["kind"] == "degrade":
                dgr[sl] += e["value"] * cv[2]
                if e["cause"] == "cooling": hot[sl] = 1
        inst = r["installed"] - r["undelivered"] - r["down"]
        with np.errstate(invalid="ignore", divide="ignore"):
            f_run = np.nan_to_num((r["useful"] + r["degraded"] + r["discarded"]) / inst)
            f_blk = np.nan_to_num(r["blocked"] / inst); f_disc = np.nan_to_num(r["discarded"] / np.maximum(r["useful"] + r["degraded"] + r["discarded"], 1e-9))
            f_deg = np.nan_to_num(r["degraded"] / np.maximum(r["useful"] + r["degraded"] + r["discarded"], 1e-9))
        uref = calib["train_running_util"] if jb.workload == "training" else calib["inference_busy_util"]
        util = np.clip(f_run * uref * (1 + 0.03 * rng.standard_normal(T)), 0, 1)
        pidle = idle_w(P.gen, g, calib)
        cap_k = sim.site_series[P.site]["power_cap_factor"]
        power = pidle + (g["tdp_w"] - pidle) * util ** a * cap_k
        power *= 1 + 0.02 * rng.standard_normal(T)
        liquid = "liquid" in s["cooling"]
        inlet = 30.0 if liquid else 22 + 0.3 * (sim.site_series[P.site]["ambient_c"] - 15)
        temp = inlet + calib["device_temp_load_rise_c"] * (0.6 if liquid else 1.0) * power / g["tdp_w"] + 12 * hot + rng.normal(0, 0.8, T)
        isdown = down >= 0.5; isund = und >= 0.5
        util[isdown | isund] = 0; power[isdown | isund] = 0; temp[isdown | isund] = np.nan
        base_ecc = calib["a100_correctable_ecc_per_gpu_day"] * (1.0 if P.gen in ("A100",) or g["family"] in ("tpu", "trn") else calib["h100_vs_a100_memory_error_ratio"])
        ecc = rng.poisson(base_ecc * dt / 1440, T).astype(np.int16); ecc[isdown | isund] = 0
        fate = np.where(f_run > 0.5, np.where(f_disc > 0.5, "discarded", np.where((f_deg > 0.1) | (dgr > 0.1), "degraded", "useful")),
                        np.where(f_blk > 0.25, "blocked", "idle")).astype(object)
        fate[isdown] = "down"; fate[isund] = "undelivered"
        health = np.where(hot > 0, "degraded", "healthy").astype(object)
        health[isdown] = np.where(planned[isdown], "maintenance", "down"); health[isund] = "not_installed"
        cause = np.where(fate == "down", cause, np.where(fate == "undelivered", cause, "")); party = np.where((fate == "down") | (fate == "undelivered"), party, "")
        parts.append(pd.DataFrame(dict(ts=ts, device_id=np.int32(d), site=P.site, gen=P.gen, pod_id=np.int32(pid), job_id=np.int32(P.job_id),
                                       util=util.astype(np.float32), power_w=power.astype(np.float32), temp_c=temp.astype(np.float32),
                                       ecc_correctable=ecc, ecc_uncorrectable=uce, error_code=code, health=health, true_fate=fate,
                                       true_cause=cause, true_party=party)))
    df = pd.concat(parts, ignore_index=True)
    for c in ("site", "gen", "error_code", "health", "true_fate", "true_cause", "true_party"):
        df[c] = df[c].astype("category")
    return df

def network(cfg, calib, pods, sim, events, rng):
    gens = cfg["generations"]; H = int(sim.T * sim.dt // 60)
    ev_by_pod = defaultdict(list)
    for e in events:
        for (p, off, c) in e["segs"]:
            ev_by_pod[p].append(e)
    rows = []
    hours = pd.Timestamp(cfg["sim"]["start"]) + pd.to_timedelta(np.arange(H), unit="h")
    for p in pods.itertuples():
        g = gens[p.gen]
        links = int(p.n_devices * (calib["tpu_ici_cables_per_chip"] if g["family"] == "tpu" else 1))
        flaps = rng.poisson(calib["rdma_link_flap_per_link_hour"] * links, H)
        crc = rng.poisson(calib["crc_errors_per_link_hour"] * links, H)
        cable = np.zeros(H, np.int32); ocs = np.zeros(H, np.int32); part = np.zeros(H, bool)
        for e in ev_by_pod[p.pod_id]:
            h = int(e["t0"] // 60)
            if not 0 <= h < H: continue
            if e["cause"] in ("cable", "ici_cable", "nic"): cable[h] += 1; flaps[h] += 3
            if e["cause"] == "ocs": ocs[h] += 1
            if e["cause"] == "network_partition":
                part[h:min(H, int((e["t0"] + e["dur"]) // 60) + 1)] = True
        rows.append(pd.DataFrame(dict(hour=hours, pod_id=np.int32(p.pod_id), site=p.site, gen=p.gen, fabric=g["fabric"], links=links,
                                      link_flaps=flaps.astype(np.int32), crc_errors=crc.astype(np.int32), cable_faults=cable, ocs_faults=ocs, partitioned=part)))
    df = pd.concat(rows, ignore_index=True)
    for c in ("site", "gen", "fabric"): df[c] = df[c].astype("category")
    return df
