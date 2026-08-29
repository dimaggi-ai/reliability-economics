# Reliability Economics — build spec (flagship v1)

One-page plan for the rebuild, encoding what the five expert reviews and the
[prototype](prototype/PROTOTYPE_FINDINGS.md) established. Status: **design locked,
build not started.** The v0 baseline (`sim/`, and `v0-baseline/`) is kept for reference and has been superseded by the v1 in `sim/` + `docs/study.md`.

## Thesis (revised)

Not "faster recovery saves GPU-hours" (obvious) but: **which recovery
intervention — spares, elastic degradation, checkpoint cadence, or repair —
wins depends on the failure regime, and the boundaries can be mapped.** The
deliverable is that map. The prototype already showed the winner flips from
warm-spares to elastic-shrink as failures go from independent to correlated, and
that a two-node pool doesn't win even its own best case once the spare queue is
modelled — so the single-number headline was an artifact, and the phase map is
the contribution.

## The core artifact

A **policy phase diagram**: which policy maximizes ETTR across
`spare-fraction × checkpoint-interval × cluster-width × failure-correlation`,
with a correlated-burst overlay — validated against published anchors, priced in
$ and GPU-hours, and shipped with a trace schema so others can plug in their data.
Everything else exists to make that diagram trustworthy.

## Model components (each is load-bearing — the prototype showed each flips a result)

1. **Causal structure, not policy-conditioned faults.** Generate *latent*
   component events shared across policies (common random numbers); each policy
   has its own detection and remediation. Fixes the "419 interruptions are what
   Meta's automation already let through" critique — do not reuse a fixed list of
   job interruptions as the arrival process.
2. **Correlated / topology-aware failure process.** A base rate of single-node
   faults plus rarer common-cause bursts (rack/PDU/switch blast radius), not one
   Poisson. Report policy ordering across the burstiness axis.
3. **Spares as a finite queue.** Distinct clocks: failover, physical repair,
   requalification, return-to-pool. A spare stays consumed through repair. Size
   `k` is a swept lever, and the "add a third node" result is *shown*, not asserted.
4. **Checkpoint cadence + tier as a policy**, with a write-cost term (Young/Daly
   optimum; cf. CheckFreq, Gemini, JIT). The prototype confirms the naive
   monotonic sweep was wrong.
5. **Realistic elastic-shrink.** Reshard/communicator-rebuild cost + a legal-shape
   (whole-replica) constraint + non-linear progress in width — not `progress ∝ n`.
6. **Metrics in the field's language.** Headline = **ETTR / goodput**; MTTF/MTTR
   /availability and MFU as diagnostics/bounds; map the nine waste components onto
   Google badput buckets; add a **$ layer** (configurable $/GPU-hour) so spare
   idle is a cost, not just a waste name.

## Validation anchors (the model must reproduce these within stated uncertainty)

- Llama 3: 419 unexpected interruptions / 54 days / 16,384 GPUs at **>90% ETTR**
  under a Meta-like recovery config (so manual-ops is not a straw man).
- Meta RSC: MTTF ∝ 1/N (≈1.8 h at 16,384 GPUs); failure rate 2.5–17.5 /1,000
  node-days (time-varying).
- ByteRobust: ~97% ETTR at 9,600 GPUs with every-step checkpointing <0.9% overhead
  (the checkpoint-dominated regime).

## Deliverables

`sim/` (rebuilt engine + invariant tests + CI), `traces/` (a trace schema + a
Llama-3-calibrated seed family + one second calibration), `figures/` (the phase
diagram + validation-against-anchors plots), `docs/` (the study, with a prominent
"what a skeptic should attack" section and the assumption-sensitivity chapter as a
required output of every run), `REFERENCES.md`. Reuses the [prototype](prototype/)
as the engine skeleton and the standard repo's waste-ledger discipline.

## Explicitly out of scope (stated, not hidden)

Inference-serving economics (training only, unless a separate replica-goodput
model is added); per-GPU electrical/thermal simulation; a live-hardware harness.
SDC and straggler *chaos specs* live in the companion standard repo; their
*economic* effect (SDC can invalidate the last checkpoint; stragglers tax every
step) is modelled here as regimes, not devices.

## Sequencing

Build order matches dependency: (1) causal engine + correlated process + spare
queue → reproduce the prototype crossover at higher fidelity; (2) checkpoint-as-
policy + realistic elastic; (3) ETTR/$ metrics + badput mapping; (4) validation
against the three anchors; (5) the phase diagram + sensitivity chapter; (6)
adversarial review, then publish. Each stage is a checkpoint to review before the
next — this is deliberate, multi-session work, not a patch.
