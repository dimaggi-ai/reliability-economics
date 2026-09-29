# Simulator event ordering

`Sim` accepts hardware faults as `(time_h, failed_nodes)` pairs. It snapshots
and sorts them by time, then by failed-node count. Reversing or permuting the
same event multiset therefore produces the same result for a fixed configuration
and policy. The caller's list and its rows can change without altering a
constructed simulator.

Times must be finite and lie in `[0, horizon_h)`. Failed-node counts must be
positive integers; booleans are refused. Invalid values raise `ValueError`.

At the same timestamp:

- Repairs and requalification finish, and spares return, before new faults.
- A fixed-width job's recovery completes before a new fault interrupts it.
- Smaller faults precede larger faults when competing for spares. Identical
  pairs remain separate events; they are not merged into a larger burst.
- Elastic reshard stops remain serialized, including for simultaneous faults.

The fault tie rule is a deterministic modeling convention, not a claim about
physical priority. It matches the tuple ordering used by `latent_events`.
Elastic accounting credits each preceding interval at its applicable width
before discarding completed repairs.

## Scope

This timeline covers hardware faults, recovery, and spare availability for one
training job. Checkpoint losses remain an expected half-interval cost; the
simulator does not schedule individual checkpoint writes. Elastic reshard stops
are serialized, rather than modeled as overlapping operations.

Thermal and network state changes are not inputs to this API. The joined
receiver experiment uses scalar thermal/network parameters and multiplies
retained runtime by a constant healthy-step compute share. Sorting reliability
events cannot make that composition time-dependent. A dynamic integration needs
timestamped thermal/network state transitions, a shared clock and tie rules,
and interval accounting that splits useful work and communication cost only
while recovery permits progress. That requires changes to the receiver and
its model interfaces as well as this simulator; this change does not establish
dynamic cross-domain coupling.
