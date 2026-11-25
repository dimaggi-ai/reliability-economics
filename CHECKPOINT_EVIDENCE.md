# Synthetic checkpoint phase reference

This bounded product-domain reference makes two missing meanings executable:
checkpoint staging, durable commit and restore outcome are separate evidence;
overlapping blocked intervals must be attributed once. Every input and result
is synthetic. Timings are arithmetic examples, not measurements or operator
requirements. No existing recovery algorithm is changed.

Run from this repository root with Python 3.10+:

```sh
python3 sim/checkpoint_evidence.py
PYTHONPATH=sim python3 -m pytest sim/test_checkpoint_evidence.py -q
```

The first command emits deterministic JSON with input fixtures, evaluated
phases, the blocked-interval ledger and limitations. It writes no files and
uses no dependencies outside the Python standard library. Tests use pytest.

## Checkpoint evidence

`sim/checkpoint_evidence.py:evaluate_checkpoint` accepts exactly one synthetic
checkpoint/save attempt, its expected shard IDs, staging records, an optional
commit record and an optional restore attempt. The emitted JSON supplies the
complete example wire shape. Records use `checkpoint-phases/v1`.

- `workload_id`, `checkpoint_id` and save `attempt_id` must match across every
  supplied phase. Expected shard IDs must be nonempty and unique; unexpected
  or duplicate staging records are invalid. Each expected shard represents one
  declared durable-state shard, not an inferred GPU, rank or file.
- Restore uses a separate `restore_attempt_id` and explicitly retains the
  source save `attempt_id`. This reference handles at most one save and one
  restore attempt; retries and distributed identity ownership are outside it.
- `window_s` is a positive evidence-window duration. All phase times are
  seconds since its common origin, within `[0, window_s]`. Known staging
  completion cannot precede its start; commit cannot precede known staging
  phases; restore cannot precede known staging or commit; restore completion
  cannot precede its start. Equality at phase boundaries is allowed.
- Missing shard records, null phase times, absent/null commit and
  absent/unknown restore outcomes remain explicit missing evidence. A reported
  commit can remain visible while missing staging evidence makes evaluation
  incomplete. Unknown evidence is never silently turned into a success.
- A terminal restore result requires its completion time. A failed restore is
  a complete negative result, not a successful checkpoint use. `restore_verified`
  is true/false only when all phase evidence is complete, otherwise null. It
  refers only to the supplied synthetic example, never actual restorability.
- A commit record **asserts** synthetic durable completion. The function does
  not inspect storage, verify an acknowledgment, authenticate a receipt, or
  establish that saved state contains everything needed for training recovery.

Malformed inputs raise `ValueError`; incomplete evidence returns structured
incompleteness. Exact field sets, finite nonnegative times, explicit types,
nonempty IDs, schema versions and evidence classes are checked. Booleans are
never numbers. Only `synthetic` evidence is accepted.

## Independent blocked-interval ledger

`blocked_ledger` accepts `checkpoint-blocked-intervals/v1`: a fixed full-gang
GPU count, one workload ID, a window, explicit interval coverage and uniquely
identified half-open intervals `[start_s, end_s)`. Every interval blocks the
entire declared gang. A partial-width or varying-width model is not provided.

It computes disjoint GPU-second buckets. Recovery takes attribution priority
over checkpoint blocking wherever intervals overlap; this is an accounting
convention, not a causal explanation. Duplicate same-kind coverage is unioned.
The remainder is called `healthy_gpu_s`, meaning only that no supplied blocking
interval covers it. It does not imply useful compute or retained training work.

The exposed acceptance example is a synthetic 100-second window and two GPUs:

| Declared interval | Time |
|---|---|
| Checkpoint blocked | `[20, 30)` seconds |
| Recovery blocked | `[25, 45)` seconds |

Recovery owns 20 seconds × 2 GPUs = **40 GPU-s**; checkpoint alone owns 5 × 2
= **10 GPU-s**; the remaining 75 × 2 = **150 GPU-s** is unblocked. The total is
**200 GPU-s**. The overlap is counted once.

Interval differences, GPU-second multiplication and conservation use exact
internal arithmetic on the supplied integer and binary-float values. No positive
interval is rounded away and conservation has no acceptance epsilon. JSON emits
integers for integral totals/buckets and rounded finite floats otherwise, so
displayed fractional buckets may differ in their last bit when added. Derived
overflow or nonzero-to-zero underflow raises `ValueError`; no complete ledger
is emitted for those cases. Any decimal precision policy belongs upstream.

Unknown interval coverage or any unknown endpoint returns null buckets and an
incomplete evaluation. The known nominal window total remains available. An
empty interval list earns unblocked time only with explicit complete coverage.
Phase durations do not automatically become blocked intervals: asynchronous
checkpoint work can overlap training.

**Do not add or subtract this ledger into `Sim`'s retained-progress ledger or
the joined scenario's GPU-hour buckets.** The existing model already accounts
for checkpoint/restart/discard effects. This reference does not model lost
work, checkpoint bytes, host memory, CPU demand, bandwidth, training throughput
or restore duration from inputs belonging to another model.

## Fixtures and limits

`synthetic_cases()` provides a development success example and exposed
acceptance cases for staging without commit, failed restore, unknown restore
and blocked-interval overlap. `sim/test_checkpoint_evidence.py` adds negative
cases for identity mismatch, phase ordering, duplicate/missing shards,
invalid/non-finite inputs, interval boundaries and incomplete coverage.

The literal fixture generator has a stable version, no randomness and a null
seed. IDs and relationships are preserved within each case; counterfactual
cases reuse logical IDs under distinct case IDs. Both development and exposed
acceptance fixtures are visible. None is held out, representative of a failure
distribution, or evidence of operator/GPU performance.

Operator evidence would need a named workload/runtime/storage backend, complete
state and shard manifest, attributable phase timestamps, backend-specific
durability semantics, and a compatible successful restore test. Those remain
unknown. This module has no executor, admission permission, reservation,
credentials or production action.
