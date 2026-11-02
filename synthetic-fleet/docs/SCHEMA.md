# Output schema

All times are UTC. `dh` = device-hours. One run directory holds:

| file | grain | key columns |
|---|---|---|
| `bom.csv` | site x generation | devices, nodes, racks, pods, tdp_w, host_w, per_device_allin_w (incl. 5% network), it_mw, pue, facility_mw |
| `pods.parquet` | pod / superpod / slice | pod_id, site, hall, gen, n_devices, dev_start (first device_id), job_id |
| `assets.parquet` | device | device_id, site, hall, gen, pod_id, rack, node, dev_in_node, power_domain (PDU), cooling_loop, job_id |
| `jobs.parquet` | job | job_id, site, gen, workload (training gang job or inference pool), n_devices, pod_ids (JSON), spare_frac |
| `events.parquet` | event (ground truth) | event_id, kind (down, block, degrade, sdc, powercap, undelivered, demand, pue), cause, party, source (baseline or chaos), scenario, planned, interrupt, site, gen, t0 / dur (minutes from start), value (degrade fraction, cap fraction, demand multiplier, PUE delta), devices |
| `event_segments.parquet` | event x pod | event_id, pod_id, dev_offset, dev_count: the blast radius as device ranges |
| `interrupts.parquet` | job interruption | job_id, event_id, t_min, cause, party, planned, source, restart_min |
| `fates.parquet` | job x interval (**exact, all devices**) | installed_dh = useful_dh + degraded_dh + discarded_dh + blocked_dh + idle_dh + down_dh + undelivered_dh; demand_dh / served_dh (inference); it_kw; facility_kw |
| `fate_attribution_hourly.parquet` | job x hour x fate x cause x party | device_hours. Summing it over cause/party gives `fates` exactly |
| `devices_sample.parquet` | sampled device x interval | observable: util, power_w, temp_c, ecc_correctable, ecc_uncorrectable, error_code (XIDnn / RAS_UE / TPU_HW / LINK_DOWN / POWER_LOSS ...), health; ground truth: true_fate, true_cause, true_party |
| `network.parquet` | pod x hour | links, link_flaps, crc_errors, cable_faults, ocs_faults, partitioned |
| `site_power.parquet` | site x interval | it_mw, pue, facility_mw, cap_mw (NaN = uncapped), curtailed_mw, power_cap_factor, ambient_c, nameplate_mw |
| `rollups_monthly.*` | month x site x gen x workload | fate sums, it_mwh, facility_mwh, usable_frac, availability (excl. planned), ettr (training), served_frac (inference), unplanned_failures, mtbf_device_h, mttr_h, mttr_median_h |
| `sla_shortfall.*` | month x site x gen x workload x obligation x party | target, actual, shortfall_dh, party_share, attributed_shortfall_dh, credit_usd (illustrative $/dh) |
| `daily_site.csv` | day x site | usable_pct = (installed - down - undelivered)/installed, delivered_pct = useful/installed |
| `samples/*.csv` | random rows | small CSV samples of the big tables |
| `run_meta.json` | run | seed, config, chaos schedule, MW, counts, runtime |

## Fates

| fate | meaning | typical party |
|---|---|---|
| useful | step work kept | - |
| degraded | running slower than nameplate (thermal throttle, straggler, OCS reroute, grid power cap); the lost share | operator / vendor / grid |
| discarded | work done then thrown away (rollback to last checkpoint; SDC detection delay) | cause's party |
| blocked | healthy but cannot run (restart window, network partition, OCS opt-out jobs) | cause's party |
| idle | healthy, nothing to run (scheduler gap, low inference demand, reserved TPU spare cubes) | tenant / operator |
| down | device unavailable (fault, maintenance, outage) | cause's party |
| undelivered | contracted but not energized/accepted (supply delay) | vendor |

Parties: `vendor` (hardware, firmware, SDC, supply), `operator` (cables, software, PDU, cooling, hall power, network, maintenance), `grid` (utility outage, curtailment), `tenant` (job errors, demand), `unattributed` (StepFun "Unknown" class).

## Sampling

Fleet aggregates (`fates`, attribution, rollups, SLA, site power) cover every device exactly; only `devices_sample` is sampled (`sim.device_sample_frac`, min per site x gen, capped by `sim.device_sample_max`). Device-level fate is the majority state of the interval; job-level fate is fractional.
