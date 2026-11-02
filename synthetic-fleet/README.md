# synthetic-fleet

A seeded, config-driven generator of synthetic AI-accelerator fleet telemetry, with a chaos engine and **ground-truth labels** for every interval: what each device-hour really was (useful, degraded, discarded, blocked, idle, down, undelivered), why, and who is responsible (operator, vendor, grid, tenant).

Default fleet: **500 MW facility** across six campuses (ERCOT, PJM, MISO, Nordics, Korea, WECC), mixing NVIDIA A100/H100/H200/B200/GB200 NVL72, Google TPU v4/v5e/v5p/Trillium/Ironwood, AMD MI300X/MI355X and AWS Trainium2. Device counts follow from TDP + host + network overhead and site PUE, and reconcile to the target (500.04 MW, 428,776 accelerators).

It is built to test the DIMAGGI ledger / obligation tooling: run your detector or accounting on the observable tables, compare with the ground truth.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate && pip install numpy pandas pyarrow pyyaml
python -m synthfleet bom                      # bill of materials and MW reconciliation
python -m synthfleet validate                 # baseline run vs public calibration targets -> runs/validation/VALIDATION.md
python -m synthfleet run --out runs/default   # 500 MW, 30 days, 5 min, baseline + curated chaos (~20 s, ~155 MB)
python -m synthfleet montecarlo --n 5 --scenarios 8 --out runs/montecarlo   # random chaos per seed, rollups only
python -m synthfleet run --scale 2 --days 7 --out runs/scale_1gw          # 1 GW
python -m synthfleet calibration-doc          # regenerate docs/CALIBRATION.md
```

Overrides: `--seed`, `--scale`, `--days`, `--interval`, `--no-chaos`, `--no-telemetry`, `--chaos path.yaml`, `--set key.path=value` (e.g. `--set sim.device_sample_max=300`). Same seed + config = identical output.

## Layout

```
configs/fleet_500mw.yaml        sites, generations (TDP, host W, topology, fabric), sim settings, obligations
configs/calibration.yaml        every rate/parameter with value, unit, status (calibrated/derived/assumption), source
configs/scenarios/library.yaml  16 declarative chaos scenarios
configs/chaos_curated.yaml      the schedule used by the default run (14 scenarios over 30 days)
synthfleet/                     fleet.py (BOM, hierarchy), events.py (baseline faults + chaos engine),
                                sim.py (fates, power, curtailment), telemetry.py (device, network), runner.py, validate.py, cli.py
tools/derive_calibration.py     recomputes StepFun / ByteDance-derived parameters from the public traces
docs/                           SCHEMA.md, CALIBRATION.md, VALIDATION.md, BOM.txt
```

## Model in one paragraph

Hierarchy is site > hall > pod (HGX scalable unit, NVL72 8-rack pod, TPU superpod or slice) > rack > node > device. Training gang jobs span pods (up to 16,384 H100s, 18,432 GB200s, a full 9,216-chip Ironwood superpod); every other pod is an inference pool with a diurnal load. Baseline faults are Poisson per node at generation-specific rates (Llama 3, Meta RSC), with StepFun's class mix and lognormal repair times, ByteDance-derived multi-node faults and Llama 3's planned-maintenance ratio. TPU superpods use the TPU v4 machine, ICI-cable and OCS rates, with OCS fault-tolerant routing (92.2% throughput, 95% opt-in). Every interruption costs a restart window plus rework back to the last checkpoint. Power is idle + (TDP - idle) x util^0.9 plus host, PUE moves with ambient temperature, and a grid cap scales running power and moves the lost throughput into *degraded / grid*. Chaos scenarios add events with explicit blast radius (device ranges in pods), time and party.

## Chaos scenarios (configs/scenarios/library.yaml)

device_failure_burst, node_failure, rack_pdu_loss, cooling_thermal_event, hall_outage, site_outage, grid_curtailment, ocs_fault, cable_fault_burst, network_partition, silent_data_corruption, straggler, firmware_rollout_bad, cascading_failure (multi-site), supply_delay, demand_shock. Add one by writing a YAML entry with a `type` the engine knows; Monte Carlo samples from all entries.

## Caveats

* Synthetic. Rates for B200, GB200, H200, MI300X/MI355X, Trainium2, TPU v5e/v5p/Trillium/Ironwood, all TPU/AMD/Trainium host power, restart and checkpoint times, and every chaos magnitude are **assumptions**, labelled in `docs/CALIBRATION.md`.
* Restart/checkpoint times were tuned so that emergent large-job ETTR (0.89) and TPU goodput (0.97) land on Meta's and Gemini's published figures; those two checks are therefore not independent.
* Inference demand is a stylised diurnal curve, not a trace. Credits use an illustrative $2 per shortfall device-hour.
* MTTR has StepFun's heavy tail (mean ~5.7 d); repairs longer than the run are truncated at the end.

## Where it would live

Fits `dimaggi-ai/reliability-economics` (as `synthetic-fleet/`) or alongside `ai-cluster-chaos-fidelity`; the fate/party labels match the four-fate accounting in `usable-compute`. Not pushed anywhere.

## What the tarball contains

`synthetic-fleet.tar.gz` ships code, configs, docs, the validation run, the Monte Carlo summaries and the small tables of the default run. The two large tables of the default run (`fates.parquet` ~90 MB, `devices_sample.parquet` ~62 MB) and the 1 GW run are excluded; regenerate them bit-for-bit with `python -m synthfleet run --out runs/default --set sim.device_sample_max=600` (seed 42).
