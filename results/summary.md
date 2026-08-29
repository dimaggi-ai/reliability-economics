# Reliability-economics results (v1)

16,384-GPU gang training job, 30 days, 8 seeds. ETTR = productive / envelope. Reproduce: `python3 sim/run.py`.

| Scenario | Policy | spares (nodes) | ETTR | Avail | MTTF h | MTTR h | Interruptions | $ waste/mo |
|---|---|---|---|---|---|---|---|---|
| independent | manual | 0 | 0.159 | 0.181 | 3.1 | 13.91 | 43 | $24,816,775 |
| independent | auto-restart | 0 | 0.216 | 0.247 | 3.0 | 9.18 | 59 | $23,111,910 |
| independent | auto-spares | 0 | 0.216 | 0.247 | 3.0 | 9.18 | 59 | $23,111,910 |
| independent | auto-spares | 2 | 0.588 | 0.668 | 3.1 | 1.55 | 154 | $12,168,303 |
| independent | auto-spares | 8 | 0.803 | 0.915 | 3.2 | 0.29 | 208 | $5,837,552 |
| independent | auto-spares | 32 | 0.794 | 0.915 | 3.2 | 0.29 | 208 | $6,183,152 |
| independent | auto-spares-ckpt | 0 | 0.235 | 0.247 | 3.0 | 9.18 | 59 | $22,554,452 |
| independent | auto-spares-ckpt | 2 | 0.637 | 0.668 | 3.1 | 1.55 | 154 | $10,713,711 |
| independent | auto-spares-ckpt | 8 | 0.870 | 0.915 | 3.2 | 0.29 | 208 | $3,862,800 |
| independent | auto-spares-ckpt | 32 | 0.859 | 0.915 | 3.2 | 0.29 | 208 | $4,208,400 |
| independent | elastic | 0 | 0.801 | 0.843 | 2.7 | 0.50 | 226 | $5,875,774 |
| rack-bursts-b16 | manual | 0 | 0.387 | 0.421 | 6.1 | 8.31 | 50 | $18,079,432 |
| rack-bursts-b16 | auto-restart | 0 | 0.454 | 0.495 | 6.0 | 6.11 | 60 | $16,097,856 |
| rack-bursts-b16 | auto-spares | 0 | 0.454 | 0.495 | 6.0 | 6.11 | 60 | $16,097,856 |
| rack-bursts-b16 | auto-spares | 2 | 0.790 | 0.860 | 6.1 | 1.00 | 101 | $6,210,247 |
| rack-bursts-b16 | auto-spares | 8 | 0.844 | 0.922 | 6.2 | 0.52 | 108 | $4,611,554 |
| rack-bursts-b16 | auto-spares | 32 | 0.864 | 0.955 | 6.2 | 0.29 | 111 | $4,062,117 |
| rack-bursts-b16 | auto-spares-ckpt | 0 | 0.476 | 0.495 | 6.0 | 6.11 | 60 | $15,462,496 |
| rack-bursts-b16 | auto-spares-ckpt | 2 | 0.826 | 0.860 | 6.1 | 1.00 | 101 | $5,127,197 |
| rack-bursts-b16 | auto-spares-ckpt | 8 | 0.883 | 0.922 | 6.2 | 0.52 | 108 | $3,458,047 |
| rack-bursts-b16 | auto-spares-ckpt | 32 | 0.904 | 0.955 | 6.2 | 0.29 | 111 | $2,870,242 |
| rack-bursts-b16 | elastic | 0 | 0.883 | 0.919 | 5.7 | 0.50 | 116 | $3,456,755 |
