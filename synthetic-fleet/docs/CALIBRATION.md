# Calibration table

Generated from `configs/calibration.yaml`. Status: **calibrated** = taken from a public measurement; **derived** = computed by `tools/derive_calibration.py` from a public dataset; **assumption** = our choice, no public measurement.


## Calibrated (21)

| param | value | unit | source | note |
|---|---|---|---|---|
| h100_interrupt_rate | 0.0004737 | per GPU-day | [The Llama 3 Herd of Models (Table 5, sec 3.3.4)](https://arxiv.org/abs/2407.21783) | 419 unexpected interruptions / 54 days / 16,384 H100 |
| a100_interrupt_rate | 0.002621 | per GPU-day | [Revisiting Reliability in Large-Scale ML Research Clusters (Meta RSC)](https://arxiv.org/abs/2410.21680) | MTTF 47.7 days for 8-GPU jobs => 1/(47.7*8); research cluster, broader failure definition than Llama 3 |
| tpuv4_machine_rate | 0.0008 | per TPU machine-day (4 chips) | [Resiliency at Scale: Managing Google's TPUv4 ML Supercomputer (NSDI'24)](https://www.usenix.org/conference/nsdi24/presentation/zu) | 0.08% of TPU machines fail per day |
| tpuv4_ici_cable_rate | 5e-05 | per ICI cable-day | [Resiliency at Scale: Managing Google's TPUv4 ML Supercomputer (NSDI'24)](https://www.usenix.org/conference/nsdi24/presentation/zu) | 0.005% of ICI cables per day |
| tpuv4_ocs_rate | 0.0004 | per OCS-day | [Resiliency at Scale: Managing Google's TPUv4 ML Supercomputer (NSDI'24)](https://www.usenix.org/conference/nsdi24/presentation/zu) | 0.04% of OCS per day |
| tpu_ici_cables_per_chip | 1.5 | cables per chip | [Resiliency at Scale: Managing Google's TPUv4 ML Supercomputer (NSDI'24)](https://www.usenix.org/conference/nsdi24/presentation/zu) | 6,144 optical ICI links per 4,096-chip pod |
| tpu_ocs_per_4096_chips | 48 | OCS per 4 | [Resiliency at Scale: Managing Google's TPUv4 ML Supercomputer (NSDI'24)](https://www.usenix.org/conference/nsdi24/presentation/zu) |  |
| tpu_ocs_ft_routing_throughput | 0.922 | fraction | [Resiliency at Scale: Managing Google's TPUv4 ML Supercomputer (NSDI'24)](https://www.usenix.org/conference/nsdi24/presentation/zu) | Throughput under one OCS outage with fault-tolerant routing (4x4x4) |
| tpu_ft_routing_optin | 0.95 | fraction of jobs | [Resiliency at Scale: Managing Google's TPUv4 ML Supercomputer (NSDI'24)](https://www.usenix.org/conference/nsdi24/presentation/zu) |  |
| planned_maintenance_ratio | 0.112 | planned / unexpected interruptions | [The Llama 3 Herd of Models (Table 5, sec 3.3.4)](https://arxiv.org/abs/2407.21783) | 47 planned vs 419 unexpected |
| target_ettr_large_gpu_jobs | [0.85, 0.9] | ETTR | [Revisiting Reliability in Large-Scale ML Research Clusters (Meta RSC)](https://arxiv.org/abs/2410.21680) | Validation target, large jobs |
| target_tpu_goodput | 0.97 | goodput | [Gemini: A Family of Highly Capable Multimodal Models (goodput 85% -> 97%)](https://arxiv.org/abs/2312.11805) | Validation target |
| target_mttf_1024_a100_h | 7.9 | hours | [Revisiting Reliability in Large-Scale ML Research Clusters (Meta RSC)](https://arxiv.org/abs/2410.21680) |  |
| target_mttf_16k_h100_h | 3.09 | hours | [The Llama 3 Herd of Models (Table 5, sec 3.3.4)](https://arxiv.org/abs/2407.21783) | 54*24/419 |
| ironwood_cube_availability | {"cubes_needed": 130, "of": 144, "time_share": 0.95} | fraction of month | [Cluster reliability for trillion parameter models on TPUs (Google Cloud, May 2026)](https://cloud.google.com/blog/products/compute/cluster-reliability-for-trillion-parameter-models-on-tpus) | Google's modelled target; used as a validation target |
| h100_vs_a100_memory_error_ratio | 3.2 | x | [Story of Two GPUs: A100 vs H100 resilience on NCSA Delta (SC25)](https://arxiv.org/abs/2503.11901) | H100 per-GPU memory MTBE 3.2x lower than A100 |
| idle_power_w | {"A100": 72.9, "H100": 126.3, "H200": 120.9} | W loaded idle | [LLM-PowerTrace (Table 13 loaded idle power per GPU)](https://arxiv.org/abs/2610.38315) |  |
| host_overhead_w | {"A100": 412.5, "H100": 575.0, "H200": 575.0, "B200": 787.5} | W per GPU (system max / 8 - GPU TDP) | [NVIDIA DGX system datasheets (max system power: DGX A100 6.5 kW, DGX H100/H200 10.2 kW, DGX B200 14.3 kW)](https://www.nvidia.com/en-us/data-center/dgx-platform/) |  |
| pue_survey_avg | 1.56 | PUE | [dimaggi-ai/cooling-pue-ladder (survey PUE 1.56, hyperscale ~1.1, NVL72-class ~120 kW rack)](https://github.com/dimaggi-ai/cooling-pue-ladder) |  |
| pue_hyperscale | 1.1 | PUE | [dimaggi-ai/cooling-pue-ladder (survey PUE 1.56, hyperscale ~1.1, NVL72-class ~120 kW rack)](https://github.com/dimaggi-ai/cooling-pue-ladder) |  |
| nvl72_rack_kw | 120 | kW per NVL72 rack | [dimaggi-ai/cooling-pue-ladder (survey PUE 1.56, hyperscale ~1.1, NVL72-class ~120 kW rack)](https://github.com/dimaggi-ai/cooling-pue-ladder) |  |

## Derived (5)

| param | value | unit | source | note |
|---|---|---|---|---|
| node_fault_class_mix | {"gpu": 0.3244, "unknown": 0.2957, "cable": 0.0821, "fan": 0.0678, "nic": 0.0616, "psu": 0.0534, "software": 0.0493, "tenant_task": 0.0287, "change": 0.0144, "motherboard_cpu": 0.0144, "other_hw": 0.0082} | share of node faults (stress-test failures excluded) | [StepFun InfiniteHBD-Trace (fault_trace.json, 400 servers, 348 days)](https://github.com/stepfun-ai/InfiniteHBD-Trace) | tools/derive_calibration.py: 584 fault starts, 97 burn-in stress-test failures removed, renormalised |
| mttr_lognormal_median_days | {"gpu": 0.7, "unknown": 0.65, "cable": 1.97, "fan": 3.06, "nic": 0.49, "psu": 14.63, "software": 0.8, "tenant_task": 0.69, "motherboard_cpu": 1.05, "change": 0.17, "other_hw": 0.99} | days | [StepFun InfiniteHBD-Trace (fault_trace.json, 400 servers, 348 days)](https://github.com/stepfun-ai/InfiniteHBD-Trace) | Median fault_start->fault_end per class |
| mttr_lognormal_sigma | 2.525 | log-sd | [StepFun InfiniteHBD-Trace (fault_trace.json, 400 servers, 348 days)](https://github.com/stepfun-ai/InfiniteHBD-Trace) | Std of log repair time across all 584 faults (paired faults excl. zero-length: median 0.94 d, mean 5.7 d) |
| multi_node_fault_prob | 0.276 | probability a fault takes 2-3 nodes of a rack | [ByteDance llm-training-alert-trace (98 jobs, 344k alerts, CC-BY-4.0)](https://huggingface.co/datasets/bytedance-research/llm-training-alert-trace) | Share of jobs with >=2 faulty nodes (tools/derive_calibration.py), applied per fault |
| rdma_link_flap_per_link_hour | 0.00027 | per link-hour | [ByteDance llm-training-alert-trace (98 jobs, 344k alerts, CC-BY-4.0)](https://huggingface.co/datasets/bytedance-research/llm-training-alert-trace) | 1,807 rdma_link_flap_inc alerts / 828,601 Hopper node-hours / 8 links; upper bound (sample of failing jobs) |

## Assumption (39)

| param | value | unit | source | note |
|---|---|---|---|---|
| h200_interrupt_rate | 0.0004737 | per GPU-day | DIMAGGI assumption (no public data) | Same Hopper platform as H100 |
| b200_interrupt_rate | 0.0006158 | per GPU-day | DIMAGGI assumption (no public data) | H100 x 1.3: new platform, early-life failures; no public Blackwell fleet data |
| gb200_interrupt_rate | 0.0007106 | per GPU-day | DIMAGGI assumption (no public data) | H100 x 1.5: NVL72 rack-scale NVLink domain, liquid cooling, early life; no public data |
| mi300x_interrupt_rate | 0.0005684 | per GPU-day | DIMAGGI assumption (no public data) | H100 x 1.2; no public AMD fleet reliability data |
| mi355x_interrupt_rate | 0.0006632 | per GPU-day | DIMAGGI assumption (no public data) | H100 x 1.4; no public data |
| trn2_interrupt_rate | 0.0005684 | per chip-day | DIMAGGI assumption (no public data) | H100 x 1.2; no public Trainium data |
| tpu_v5e_rate_mult | 1.0 | x v4 machine rate | DIMAGGI assumption (no public data) |  |
| tpu_v5p_rate_mult | 1.0 | x v4 machine rate | DIMAGGI assumption (no public data) |  |
| tpu_v6e_rate_mult | 1.1 | x v4 machine rate | DIMAGGI assumption (no public data) |  |
| tpu_v7_rate_mult | 1.2 | x v4 machine rate | DIMAGGI assumption (no public data) | No public Ironwood failure rates |
| mttr_max_days | 45 | days | DIMAGGI assumption (no public data) | Truncate heavy tail (StepFun p90 13.5 d) |
| tpu_mttr_median_days | 0.5 | days | DIMAGGI assumption (no public data) | Automated TPU repair workflows (NSDI'24) but no published repair time |
| tpu_mttr_sigma | 1.5 | log-sd | DIMAGGI assumption (no public data) |  |
| planned_maintenance_median_h | 2.0 | hours | DIMAGGI assumption (no public data) |  |
| gpu_restart_median_min | 13 | minutes | DIMAGGI assumption (no public data) | Detect + reschedule + reload; tuned so large-job ETTR lands in Meta's 0.85-0.90 |
| gpu_checkpoint_interval_min | 15 | minutes | DIMAGGI assumption (no public data) | Rework ~ U(0, interval) |
| tpu_restart_median_min | 15 | minutes | DIMAGGI assumption (no public data) | Tuned so OCS-superpod training goodput lands near Gemini's 97%; the Gemini report describes the mechanism, not the minutes |
| tpu_checkpoint_interval_min | 3 | minutes | DIMAGGI assumption (no public data) | In-memory redundant state => small rework |
| inference_restart_min | 6 | minutes | DIMAGGI assumption (no public data) |  |
| a100_correctable_ecc_per_gpu_day | 0.02 | events per GPU-day | DIMAGGI assumption (no public data) |  |
| idle_power_frac_other | 0.17 | fraction of TDP | DIMAGGI assumption (no public data) | Hopper loaded idle is ~0.17-0.18 of TDP (PowerTrace); applied to generations without data |
| network_overhead_frac | 0.05 | fraction of IT | DIMAGGI assumption (no public data) | Switches/optics |
| power_util_exponent | 0.9 | P = Pidle + (TDP-Pidle)*util^a | DIMAGGI assumption (no public data) |  |
| throughput_power_exponent | 0.7 | throughput ~ power^b under caps | DIMAGGI assumption (no public data) |  |
| inference_load_base | 0.55 | fraction | DIMAGGI assumption (no public data) |  |
| inference_load_diurnal_amp | 0.3 | fraction | DIMAGGI assumption (no public data) |  |
| training_run_median_days | 3.0 | days | DIMAGGI assumption (no public data) |  |
| training_gap_median_min | 45 | minutes | DIMAGGI assumption (no public data) | Scheduler gap between training runs |
| crc_errors_per_link_hour | 0.05 | per link-hour | DIMAGGI assumption (no public data) |  |
| train_running_util | 0.9 | GPU/TPU busy fraction while a training step runs | DIMAGGI assumption (no public data) |  |
| inference_busy_util | 0.75 | busy fraction of a serving device under load | DIMAGGI assumption (no public data) |  |
| host_idle_frac | 0.6 | host power when accelerators idle / host max | DIMAGGI assumption (no public data) |  |
| restart_sigma | 0.4 | log-sd of restart time | DIMAGGI assumption (no public data) |  |
| pue_ambient_sensitivity | 0.004 | PUE per degC above site mean | DIMAGGI assumption (no public data) |  |
| ambient_diurnal_amp_c | 5.0 | degC | DIMAGGI assumption (no public data) |  |
| device_temp_idle_c | 38 | degC above which load adds | DIMAGGI assumption (no public data) |  |
| device_temp_load_rise_c | 32 | degC rise at full power (liquid cooled x0.6) | DIMAGGI assumption (no public data) |  |
| mttr_scale_by_gen | {"A100": 0.18} | multiplier on node repair time | DIMAGGI assumption (no public data) | Meta RSC's A100 rate counts every job-failing node event (many cleared by reboot/health check); scaling StepFun repair times by rate_H100/rate_A100 keeps A100 steady-state downtime near H100's instead of ~15% |
| warmup_days | 45 | days of pre-roll so repairs already in progress at t=0 exist (equals mttr_max) | DIMAGGI assumption (no public data) |  |

## Fleet-config assumptions

TPU, Trainium, MI355X, MI300X host and TDP watts in `configs/fleet_500mw.yaml` are assumptions (no vendor system-power datasheet used); GB200 per-GPU overhead is derived from the ~120 kW NVL72 rack figure. Site PUEs are chosen within the public 1.10 (hyperscale) to 1.56 (survey average) range. All chaos-scenario magnitudes in `configs/scenarios/library.yaml` are assumptions.
