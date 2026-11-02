"""Derive calibration parameters from public datasets (StepFun InfiniteHBD-Trace, ByteDance alert trace).

Usage:
  git clone https://github.com/stepfun-ai/InfiniteHBD-Trace
  python tools/derive_calibration.py --stepfun InfiniteHBD-Trace/fault_trace.json [--bytedance]
Prints the derived values that are recorded in configs/calibration.yaml.
"""
import argparse, json, collections
import numpy as np

CLASS_MAP = {"GPU": "gpu", "Unknown Error": "unknown", "Parameter Plane Cable": "cable", "Fan": "fan", "NIC": "nic",
             "Power Supply": "psu", "Training Task Troubleshooting": "tenant_task", "Motherboard": "motherboard_cpu",
             "CPU": "motherboard_cpu", "Change": "change", "Test": "change", "System Crash": "change"}

def stepfun(path):
    ev = sorted(json.load(open(path)), key=lambda e: e["event_time"])
    open_, rows = {}, []
    for e in ev:
        if e["event_type"] == "fault_start":
            open_.setdefault(e["node_id"], []).append(e)
        elif open_.get(e["node_id"]):
            s = open_[e["node_id"]].pop(0)
            rows.append((s["fault_type"]["Level"], s["fault_type"]["Class"], e["event_time"] - s["event_time"]))
    out = collections.defaultdict(list)
    for lvl, cls, d in rows:
        if cls == "Stress Test Failure":
            continue
        k = "software" if lvl == "Software Failure" else CLASS_MAP.get(cls, "other_hw")
        out[k].append(d)
    n = sum(len(v) for v in out.values())
    allr = np.array([d for _, _, d in rows]); allr = allr[allr > 0]
    print("faults (excl. stress test):", n, "of", len(rows))
    print("mix:", {k: round(len(v) / n, 4) for k, v in sorted(out.items(), key=lambda x: -len(x[1]))})
    print("median TTR days:", {k: round(float(np.median(v)), 2) for k, v in out.items()})
    print("overall median %.3f mean %.3f log-sd %.3f" % (np.median(allr), allr.mean(), np.log(allr).std()))
    print("node faults per server-day (400 servers, 348 d): %.5f" % (len(rows) / 400 / 348))

def bytedance():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    j = pd.read_parquet(hf_hub_download("bytedance-research/llm-training-alert-trace", "data/jobs.parquet", repo_type="dataset"))
    e = pd.read_parquet(hf_hub_download("bytedance-research/llm-training-alert-trace", "data/events.parquet", repo_type="dataset"))
    st = j.ip_status.apply(json.loads)
    bad = st.apply(lambda d: sum(v != "normal" for v in d.values()))
    dur = (pd.to_datetime(j.end_time) - pd.to_datetime(j.start_time)).dt.total_seconds() / 3600
    nodes = st.apply(len)
    hop = j.gpu_generation == "Hopper"
    flaps = e[e.event == "rdma_link_flap_inc"].merge(j[["job_id", "gpu_generation"]], on="job_id")
    nh = float((nodes[hop] * dur[hop]).sum())
    print("share of jobs with >=2 faulty nodes: %.3f" % (bad >= 2).mean())
    print("hopper node-hours %.0f, link flap alerts %d, per link-hour (8 links/node) %.2e" % (nh, (flaps.gpu_generation == "Hopper").sum(), (flaps.gpu_generation == "Hopper").sum() / nh / 8))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stepfun"); ap.add_argument("--bytedance", action="store_true")
    a = ap.parse_args()
    if a.stepfun: stepfun(a.stepfun)
    if a.bytedance: bytedance()
