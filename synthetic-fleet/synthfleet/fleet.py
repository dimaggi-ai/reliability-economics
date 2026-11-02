"""Build the fleet bill of materials and hierarchy: site > hall > pod > rack > node > device."""
import math
import numpy as np
import pandas as pd

def per_device_w(g, calib):
    return (g["tdp_w"] + g["host_w"]) * (1 + calib["network_overhead_frac"])

def idle_w(gen, g, calib):
    known = calib["idle_power_w"]
    return known.get(gen, g["tdp_w"] * calib["idle_power_frac_other"])

def build_fleet(cfg, calib, rng):
    fl = cfg["fleet"]
    target = fl["target_facility_mw"] * fl.get("scale", 1.0)
    gens = cfg["generations"]
    bom, pods = [], []
    pod_id = 0
    dev_cursor = 0
    for s in cfg["sites"]:
        fac_mw = target * s["share"]
        it_mw = fac_mw / s["pue"]
        site_pods = []
        for gen, frac in s["mix"].items():
            g = gens[gen]
            pdw = per_device_w(g, calib)
            rack = g["dev_per_node"] * g["nodes_per_rack"]
            n = int(round(it_mw * 1e6 * frac / pdw / rack)) * rack  # rack granularity
            n_pods = math.ceil(n / g["dev_per_pod"])
            remaining = n
            for k in range(n_pods):
                size = min(g["dev_per_pod"], remaining)
                remaining -= size
                site_pods.append(dict(site=s["id"], gen=gen, n_devices=size))
            bom.append(dict(site=s["id"], region=s["region"], gen=gen, vendor=g["vendor"], family=g["family"], fabric=g["fabric"],
                            devices=n, nodes=n // g["dev_per_node"], racks=n // rack, pods=n_pods,
                            tdp_w=g["tdp_w"], host_w=g["host_w"], per_device_allin_w=round(pdw, 1),
                            it_mw=n * pdw / 1e6, pue=s["pue"], facility_mw=n * pdw * s["pue"] / 1e6))
        # assign pods to halls round robin (so a hall holds mixed generations)
        for i, p in enumerate(site_pods):
            p["hall"] = f"{s['id']}-H{(i % s['halls']) + 1}"
            p["pod_id"] = pod_id
            p["dev_start"] = dev_cursor
            dev_cursor += p["n_devices"]
            pod_id += 1
            pods.append(p)
    bom = pd.DataFrame(bom)
    pods = pd.DataFrame(pods)
    pods["pod_name"] = pods.apply(lambda r: f"{r.site}-{r.gen}-P{r.pod_id:04d}", axis=1)
    return bom, pods, target

def build_jobs(cfg, pods, rng):
    """Partition pods into training gang jobs (up to max_train_job devices) and inference pools (one per pod)."""
    gens = cfg["generations"]
    jobs = []
    pods = pods.copy()
    pods["job_id"] = -1
    jid = 0
    for s in cfg["sites"]:
        for gen in s["mix"]:
            g = gens[gen]
            sub = pods[(pods.site == s["id"]) & (pods.gen == gen)]
            total = int(sub.n_devices.sum())
            train_target = total * s["training_share"]
            idx = list(sub.index)
            acc, cur, cur_n = 0, [], 0
            i = 0
            while i < len(idx) and acc < train_target:
                n = int(pods.at[idx[i], "n_devices"])
                if cur and cur_n + n > g["max_train_job"]:
                    jobs.append(dict(job_id=jid, site=s["id"], gen=gen, workload="training", pods=cur, n_devices=cur_n)); jid += 1
                    cur, cur_n = [], 0
                cur.append(idx[i]); cur_n += n; acc += n; i += 1
            if cur:
                jobs.append(dict(job_id=jid, site=s["id"], gen=gen, workload="training", pods=cur, n_devices=cur_n)); jid += 1
            for j in idx[i:]:
                jobs.append(dict(job_id=jid, site=s["id"], gen=gen, workload="inference", pods=[j], n_devices=int(pods.at[j, "n_devices"]))); jid += 1
    for jb in jobs:
        for p in jb["pods"]:
            pods.at[p, "job_id"] = jb["job_id"]
        jb["pod_ids"] = [int(pods.at[p, "pod_id"]) for p in jb["pods"]]
        del jb["pods"]
    jobs = pd.DataFrame(jobs)
    jobs["spare_frac"] = jobs.gen.map(lambda x: gens[x]["spare_frac"]) * (jobs.workload == "training")
    return jobs, pods

def assets_table(cfg, pods):
    """Device-level asset register (one row per device)."""
    gens = cfg["generations"]
    parts = []
    for p in pods.itertuples():
        g = gens[p.gen]
        off = np.arange(p.n_devices)
        node = off // g["dev_per_node"]
        rack = node // g["nodes_per_rack"]
        parts.append(pd.DataFrame(dict(device_id=p.dev_start + off, site=p.site, hall=p.hall, gen=p.gen, pod_id=p.pod_id,
                                       rack=rack.astype(np.int32), node=node.astype(np.int32), dev_in_node=(off % g["dev_per_node"]).astype(np.int16),
                                       power_domain=[f"{p.hall}-pdu-{p.pod_id}-{r}" for r in rack], cooling_loop=p.hall + "-loop", job_id=p.job_id)))
    return pd.concat(parts, ignore_index=True)
