# Validation (baseline only, seed 42, 30 days)

20/20 checks passed.

Kinds: *implementation* = the generator reproduces a parameter it was given; *emergent* = an outcome that was not set directly; *reconciliation*/*identity* = accounting checks.

| id | check | kind | target | simulated | pass | source | note |
|---|---|---|---|---|---|---|---|
| rate_H100 | H100 unexpected interruptions per GPU-day | implementation | 0.0004737 | 0.0004678 | yes | Llama 3 (Meta 2024) |  |
| rate_A100 | A100 unexpected interruptions per GPU-day | implementation | 0.002621 | 0.002629 | yes | Meta RSC (Kokolis 2024) |  |
| mttf_16k_h100 | MTTF of 16384-GPU H100 training job (h) | emergent | 3.09 | 3.243 | yes | Llama 3: 419 unexpected in 54 days on 16,384 GPUs | scaled to 16,384 GPUs; counts only interrupts outside scheduler gaps |
| mttf_1024_a100 | MTTF of a 1,024-GPU A100 job (h) | emergent | 7.9 | 9.056 | yes | Meta RSC: 7.9 h at 1,024 GPUs | calibrated rate uses RSC's per-node 47.7-day MTTF, so a ~10% gap vs the 7.9 h job figure is expected |
| ettr_large_gpu | ETTR, GPU training jobs >= 4,096 devices | emergent (restart/checkpoint tuned) | 0.85-0.9 | 0.8927 | yes | Meta RSC ETTR 0.85-0.90 for large jobs |  |
| tpu_goodput | Goodput, TPU OCS-superpod training jobs | emergent | 0.97 | 0.9683 | yes | Gemini: goodput 85% -> 97% |  |
| tpu_machine_rate | TPU machine failures per machine-day (v4-normalised) | implementation | 0.0008 | 0.0007931 | yes | TPU v4 NSDI'23/24: 0.08%/day |  |
| tpu_ici_rate | ICI cable failures per cable-day | implementation | 5e-05 | 4.906e-05 | yes | TPU v4: 0.005%/day |  |
| tpu_ocs_rate | OCS failures per OCS-day | implementation | 0.0004 | 0.0003465 | yes | TPU v4: 0.04%/day | 15 events; Poisson-tolerant check |
| ironwood_cubes | Share of time >= 130 of 144 Ironwood cubes healthy | emergent (Ironwood rates are assumptions) | 0.95 | 1 | yes | Google Ironwood blog: 130/144 cubes available 95% of the month |  |
| mttr_median | GPU node fault MTTR median (days, excl. A100 which is rescaled) | implementation | 0.935 | 0.9353 | yes | StepFun InfiniteHBD-Trace (derived) | class-mixture lognormal; truncated at 45 d |
| gpu_share | Share of node faults classed GPU | implementation | 0.3244 | 0.3307 | yes | StepFun InfiniteHBD-Trace |  |
| multi_node | Share of faults spanning >1 node | implementation | 0.276 | 0.2771 | yes | ByteDance alert trace (derived) |  |
| planned_ratio | Planned / unexpected interruptions | implementation | 0.112 | 0.1182 | yes | Llama 3: 47 / 419 |  |
| ecc_ratio | H100 / A100 correctable memory-error rate | implementation | 3.2 | 3.394 | yes | NCSA Delta (Cui et al. 2025) | absolute A100 rate is an assumption |
| idle_A100 | A100 idle power (W) | implementation | 72.9 | 73.06 | yes | LLM-PowerTrace |  |
| idle_H100 | H100 idle power (W) | implementation | 126.3 | 126.6 | yes | LLM-PowerTrace |  |
| idle_H200 | H200 idle power (W) | implementation | 120.9 | 121.1 | yes | LLM-PowerTrace |  |
| nameplate | Facility nameplate (MW) | reconciliation | 500.0 | 500 | yes | config |  |
| identity | Max |installed - sum of fates| per row (device-hours) | identity | 0 | 0.0001221 | yes | definition |  |
