"""CLI: python -m synthfleet {bom,run,montecarlo,validate,calibration-doc}"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from .config import Calib, load_config, load_library, HERE
from .fleet import build_fleet, build_jobs

DEF_CFG = os.path.join(HERE, "configs", "fleet_500mw.yaml")
DEF_CHAOS = os.path.join(HERE, "configs", "chaos_curated.yaml")

def _ov(a):
    ov = {}
    if a.scale: ov["fleet.scale"] = a.scale
    if a.days: ov["sim.days"] = a.days
    if a.interval: ov["sim.interval_min"] = a.interval
    for kv in a.set or []:
        k, v = kv.split("="); ov[k] = json.loads(v) if v[:1] in "0123456789[{-tf" else v
    return ov

def cmd_bom(a):
    cfg = load_config(a.config, _ov(a)); cal = Calib()
    bom, pods, tgt = build_fleet(cfg, cal, None); jobs, _ = build_jobs(cfg, pods, None)
    pd.set_option("display.width", 200)
    print(bom[["site", "region", "gen", "vendor", "devices", "racks", "pods", "per_device_allin_w", "it_mw", "pue", "facility_mw"]].round(2).to_string(index=False))
    print(f"\nTarget facility {tgt:.1f} MW | BOM facility {bom.facility_mw.sum():.2f} MW | IT {bom.it_mw.sum():.2f} MW | "
          f"fleet PUE {bom.facility_mw.sum()/bom.it_mw.sum():.3f} | devices {bom.devices.sum():,} | pods {len(pods)} | jobs {len(jobs)}")
    if a.out: bom.to_csv(a.out, index=False)

def cmd_run(a):
    from .runner import run
    o = run(a.config, a.out, seed=a.seed, chaos_path=None if a.no_chaos else a.chaos, overrides=_ov(a), chaos=not a.no_chaos, telemetry_on=not a.no_telemetry)
    print(json.dumps({k: v for k, v in o["meta"].items() if k != "schedule"}, indent=1))

def cmd_mc(a):
    from .runner import run
    from .events import random_schedule
    lib = load_library(); rows = []; os.makedirs(a.out, exist_ok=True)
    for s in range(a.seed, a.seed + a.n):
        cfg = load_config(a.config, _ov(a)); rng = np.random.default_rng([s, 99])
        sched = random_schedule(cfg, lib, rng, cfg["sim"]["days"] * 24, a.scenarios)
        o = run(a.config, os.path.join(a.out, f"seed_{s}"), seed=s, overrides=_ov(a), mc_schedule=sched, telemetry_on=False, write=False)
        r = o["roll"]; r.insert(0, "seed", s); r.to_parquet(os.path.join(a.out, f"rollups_seed_{s}.parquet"), index=False)
        sla = o["sla"]; sla.insert(0, "seed", s); sla.to_parquet(os.path.join(a.out, f"sla_seed_{s}.parquet"), index=False)
        f = o["fates"]; tot = f[[c for c in f.columns if c.endswith("_dh")]].sum()
        by_party = o["attr"][o["attr"].fate != "useful"].groupby("party", observed=True).device_hours.sum() / tot.installed_dh
        rows.append(dict(seed=s, scenarios=";".join(f"{x['use']}@{x['target']['site']}" for x in sched), usable_frac=tot.useful_dh / tot.installed_dh,
                         down_frac=tot.down_dh / tot.installed_dh, served_frac=tot.served_dh / tot.demand_dh,
                         shortfall_dh=float(sla.attributed_shortfall_dh.sum()) if len(sla) else 0.0,
                         credits_usd=float(sla.credit_usd.sum()) if len(sla) else 0.0,
                         **{f"loss_{k}": v for k, v in by_party.items()}))
        print(f"seed {s}: usable {rows[-1]['usable_frac']:.4f} shortfall {rows[-1]['shortfall_dh']:.0f} dh  [{rows[-1]['scenarios']}]", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "montecarlo_summary.csv"), index=False)
    print(df.describe().T[["mean", "std", "min", "max"]].to_string())

def cmd_validate(a):
    from .validate import validate
    rep = validate(a.config, a.out, seed=a.seed or 42)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
    print(rep[["id", "kind", "target", "simulated", "passed"]].to_string(index=False))
    print(f"\n{int(rep.passed.sum())}/{len(rep)} passed")

def cmd_caldoc(a):
    cal = Calib(); rows = list(cal.rows())
    with open(a.out, "w") as f:
        f.write("# Calibration table\n\nGenerated from `configs/calibration.yaml`. Status: **calibrated** = taken from a public measurement; "
                "**derived** = computed by `tools/derive_calibration.py` from a public dataset; **assumption** = our choice, no public measurement.\n\n")
        for st in ("calibrated", "derived", "assumption"):
            f.write(f"\n## {st.capitalize()} ({sum(r['status']==st for r in rows)})\n\n| param | value | unit | source | note |\n|---|---|---|---|---|\n")
            for r in rows:
                if r["status"] != st: continue
                src = f"[{r['source']}]({r['url']})" if r["url"] else r["source"]
                f.write(f"| {r['param']} | {json.dumps(r['value'])} | {r['unit']} | {src} | {r['note']} |\n")
        f.write("\n## Fleet-config assumptions\n\nTPU, Trainium, MI355X, MI300X host and TDP watts in `configs/fleet_500mw.yaml` are assumptions (no vendor system-power datasheet used); "
                "GB200 per-GPU overhead is derived from the ~120 kW NVL72 rack figure. Site PUEs are chosen within the public 1.10 (hyperscale) to 1.56 (survey average) range. "
                "All chaos-scenario magnitudes in `configs/scenarios/library.yaml` are assumptions.\n")
    print("wrote", a.out)

def main(argv=None):
    p = argparse.ArgumentParser(prog="synthfleet")
    sub = p.add_subparsers(dest="cmd", required=True)
    def common(sp, out):
        sp.add_argument("--config", default=DEF_CFG); sp.add_argument("--out", default=out); sp.add_argument("--seed", type=int)
        sp.add_argument("--scale", type=float); sp.add_argument("--days", type=float); sp.add_argument("--interval", type=float)
        sp.add_argument("--set", action="append", help="override e.g. sim.device_sample_max=300")
    sp = sub.add_parser("bom"); common(sp, None); sp.set_defaults(f=cmd_bom)
    sp = sub.add_parser("run"); common(sp, "runs/default"); sp.add_argument("--chaos", default=DEF_CHAOS); sp.add_argument("--no-chaos", action="store_true")
    sp.add_argument("--no-telemetry", action="store_true"); sp.set_defaults(f=cmd_run)
    sp = sub.add_parser("montecarlo"); common(sp, "runs/montecarlo"); sp.add_argument("--n", type=int, default=5); sp.add_argument("--scenarios", type=int, default=8)
    sp.set_defaults(f=cmd_mc)
    sp = sub.add_parser("validate"); common(sp, "runs/validation"); sp.set_defaults(f=cmd_validate)
    sp = sub.add_parser("calibration-doc"); sp.add_argument("--out", default=os.path.join(HERE, "docs", "CALIBRATION.md")); sp.set_defaults(f=cmd_caldoc)
    a = p.parse_args(argv)
    if a.cmd == "montecarlo" and a.seed is None: a.seed = 1000
    a.f(a)

if __name__ == "__main__":
    main()
