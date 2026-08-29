# v0 baseline (superseded — kept for contrast)

This is the first-generation recovery-policy simulator: single-Poisson faults,
a policy-independent interruption stream, an always-sufficient spare pool, and a
frozen checkpoint regime. It reported a single headline ("a two-node warm-spare
pool is the biggest lever, ~83%").

Five independent expert reviews and the [prototype](../prototype/) showed that
headline was an artifact of those assumptions. The current model
([../sim/](../sim/)) replaces the fault process (shared latent events, correlated
bursts), models spares as a finite queue, makes checkpoint cadence a lever with a
write cost, and reports a phase diagram instead of a number — see
[../docs/study.md](../docs/study.md). This directory is retained so the
correction is auditable, not hidden.
