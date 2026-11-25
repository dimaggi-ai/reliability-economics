"""Synthetic checkpoint phase and wall-clock accounting reference; no executor.

This module does not modify, invoke or adjust reliability_sim's retained-work
ledger. Its interval ledger counts only declared wall-clock blocking, with a
fixed full-gang width. No stochastic process or measured timings are supplied.
"""
from __future__ import annotations

import copy
import json
import math
from fractions import Fraction

IDENTITY = {"workload_id", "checkpoint_id", "attempt_id"}
COMMON = {"schema_version", "evidence_class", "case_id", "split", "workload_id", "window_s"}
SPLITS = {"development", "exposed-acceptance"}


def _object(value, fields, label):
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} must contain exactly {sorted(fields)}")


def _text(value, label):
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be nonempty text without outer whitespace")


def _number(value, label, *, positive=False):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number, excluding bool")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _derived_number(value, label):
    """Render exact arithmetic; a nonzero duration must not become zero."""
    try:
        number = value.numerator if value.denominator == 1 else float(value)
        if not math.isfinite(number) or (value and number == 0):
            raise OverflowError
    except OverflowError as exc:
        raise ValueError(f"{label} arithmetic overflow or underflow") from exc
    return number


def _common(record, version):
    if record["schema_version"] != version or record["evidence_class"] != "synthetic":
        raise ValueError("unsupported schema or evidence class; this reference accepts synthetic only")
    _text(record["split"], "split")
    if record["split"] not in SPLITS:
        raise ValueError("split must be development or exposed-acceptance")
    for key in ("workload_id", "case_id"):
        _text(record[key], key)
    _number(record["window_s"], "window_s", positive=True)


def _time(value, label, window):
    if value is not None:
        _number(value, label)
        if value > window:
            raise ValueError(f"{label} exceeds the evidence window")
    return value


def _binding(event, record):
    for key in IDENTITY:
        _text(event[key], key)
        if event[key] != record[key]:
            raise ValueError(f"mismatched {key}")


def evaluate_checkpoint(record):
    """Validate one synthetic save attempt and at most one restore attempt.

    Missing evidence gives an incomplete evaluation. Contradictory known times,
    malformed records and mismatched identities raise ValueError. A supplied
    commit asserts synthetic durability; this function cannot verify storage.
    """
    _object(record, COMMON | IDENTITY | {"expected_shard_ids", "staging", "commit", "restore"}, "checkpoint")
    _common(record, "checkpoint-phases/v1")
    for key in IDENTITY:
        _text(record[key], key)
    expected = record["expected_shard_ids"]
    if type(expected) is not list or not expected:
        raise ValueError("expected_shard_ids must be a nonempty list")
    for shard in expected:
        _text(shard, "shard_id")
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate expected shard ID")
    if type(record["staging"]) is not list:
        raise ValueError("staging must be a list")
    window = record["window_s"]
    seen, missing, completed = set(), [], []
    starts = []
    for stage in record["staging"]:
        _object(stage, IDENTITY | {"shard_id", "started_s", "completed_s"}, "stage")
        _binding(stage, record)
        shard = stage["shard_id"]
        _text(shard, "shard_id")
        if shard not in expected or shard in seen:
            raise ValueError("unexpected or duplicate staged shard ID")
        seen.add(shard)
        start = _time(stage["started_s"], "stage.started_s", window)
        end = _time(stage["completed_s"], "stage.completed_s", window)
        if start is None:
            missing.append(f"staging start:{shard}")
        else:
            starts.append(start)
        if end is None:
            missing.append(f"staging completion:{shard}")
        elif start is not None and end < start:
            raise ValueError("staging completion precedes start")
        else:
            completed.append(end)
    missing.extend(f"staging record:{shard}" for shard in expected if shard not in seen)
    staging_complete = not missing

    commit_time = None
    commit_status = "missing"
    if record["commit"] is not None:
        commit = record["commit"]
        _object(commit, IDENTITY | {"committed_s"}, "commit")
        _binding(commit, record)
        commit_time = _time(commit["committed_s"], "commit.committed_s", window)
        commit_status = "unknown" if commit_time is None else "committed"
        if commit_time is not None and any(t > commit_time for t in starts + completed):
            raise ValueError("commit precedes a known staging phase")
    if commit_time is None:
        missing.append("durable commit")

    restore_status = "not_reported"
    restore_attempt_id = None
    if record["restore"] is not None:
        restore = record["restore"]
        _object(restore, IDENTITY | {"restore_attempt_id", "started_s", "completed_s", "outcome"}, "restore")
        _binding(restore, record)
        restore_attempt_id = restore["restore_attempt_id"]
        _text(restore_attempt_id, "restore_attempt_id")
        if restore_attempt_id == record["attempt_id"]:
            raise ValueError("restore and save attempt IDs must differ")
        restore_status = restore["outcome"]
        if restore_status not in ("succeeded", "failed", "unknown"):
            raise ValueError("unknown restore outcome")
        start = _time(restore["started_s"], "restore.started_s", window)
        end = _time(restore["completed_s"], "restore.completed_s", window)
        if start is None:
            missing.append("restore start")
        if end is not None and start is not None and end < start:
            raise ValueError("restore completion precedes start")
        if (end is None) != (restore_status == "unknown"):
            raise ValueError("terminal restore outcome needs completion; unknown needs null completion")
        restore_times = [t for t in (start, end) if t is not None]
        if commit_time is not None and any(t < commit_time for t in restore_times):
            raise ValueError("restore precedes commit")
        if any(t > r for t in starts + completed for r in restore_times):
            raise ValueError("restore precedes a known staging phase")
    if restore_status in ("unknown", "not_reported"):
        missing.append("restore outcome")
    evaluation = "incomplete" if missing else "complete"
    return {
        "schema_version": "checkpoint-phase-result/v1", "evidence_class": "synthetic",
        "case_id": record["case_id"], "split": record["split"],
        **{key: record[key] for key in sorted(IDENTITY)},
        "expected_shard_ids": list(expected), "restore_attempt_id": restore_attempt_id,
        "staging_status": "complete" if staging_complete else "incomplete",
        "commit_status": commit_status, "restore_status": restore_status,
        "evaluation": evaluation, "missing_evidence": missing,
        "restore_verified": (restore_status == "succeeded") if not missing else None,
        "execution_authorized": False, "mutation_request": None,
    }


def blocked_ledger(record):
    """Disjoint full-gang GPU-second buckets; recovery wins overlapping time.

    Intervals are half-open [start, end), within [0, window_s]. Unknown coverage
    or endpoints produces null buckets rather than fabricated healthy time.
    """
    _object(record, COMMON | {"gpus", "interval_coverage", "intervals"}, "ledger")
    _common(record, "checkpoint-blocked-intervals/v1")
    gpus = record["gpus"]
    if type(gpus) is not int:
        raise ValueError("gpus must be a positive integer")
    _number(gpus, "gpus", positive=True)
    window = record["window_s"]
    total_exact = Fraction(window) * gpus
    total = _derived_number(total_exact, "total_gpu_s")
    if record["interval_coverage"] not in ("complete", "unknown"):
        raise ValueError("unknown interval coverage class")
    if type(record["intervals"]) is not list:
        raise ValueError("intervals must be a list")
    missing = ["interval coverage"] if record["interval_coverage"] == "unknown" else []
    intervals, seen = [], set()
    for interval in record["intervals"]:
        _object(interval, {"interval_id", "workload_id", "kind", "start_s", "end_s"}, "interval")
        _text(interval["interval_id"], "interval_id")
        _text(interval["workload_id"], "workload_id")
        if interval["interval_id"] in seen or interval["workload_id"] != record["workload_id"]:
            raise ValueError("duplicate interval ID or mismatched workload ID")
        seen.add(interval["interval_id"])
        if interval["kind"] not in ("checkpoint", "recovery"):
            raise ValueError("unknown interval kind")
        start = _time(interval["start_s"], "interval.start_s", window)
        end = _time(interval["end_s"], "interval.end_s", window)
        if start is None or end is None:
            missing.append(f"interval endpoints:{interval['interval_id']}")
        elif end <= start:
            raise ValueError("interval must have positive duration")
        else:
            intervals.append((Fraction(start), Fraction(end), interval["kind"]))
    buckets = None
    if not missing:
        points = sorted({Fraction(0), Fraction(window)} | {t for a, b, _ in intervals for t in (a, b)})
        durations = {"recovery_gpu_s": Fraction(0), "checkpoint_only_gpu_s": Fraction(0), "healthy_gpu_s": Fraction(0)}
        for a, b in zip(points, points[1:]):
            kinds = {kind for start, end, kind in intervals if start <= a and b <= end}
            key = "recovery_gpu_s" if "recovery" in kinds else "checkpoint_only_gpu_s" if kinds else "healthy_gpu_s"
            durations[key] += (b - a) * gpus
        if sum(durations.values()) != total_exact:
            raise ValueError("wall-clock ledger failed conservation")
        buckets = {key: _derived_number(value, key) for key, value in durations.items()}
    return {
        "schema_version": "checkpoint-blocked-ledger/v1", "evidence_class": "synthetic",
        "case_id": record["case_id"], "split": record["split"],
        "workload_id": record["workload_id"], "window_s": window, "gpus": gpus,
        "interval_ids": [i["interval_id"] for i in record["intervals"]],
        "evaluation": "incomplete" if missing else "complete", "missing_evidence": missing,
        "total_gpu_s": total, "buckets": buckets, "overlap_priority": "recovery",
        "accounting_basis": "wall-clock blocking, not retained progress",
        "execution_authorized": False, "mutation_request": None,
    }


def synthetic_cases():
    """Literal deterministic fixtures; exposed cases are not held-out evidence."""
    binding = {"workload_id": "synthetic-workload-0", "checkpoint_id": "synthetic-checkpoint-0", "attempt_id": "synthetic-save-0"}
    valid = {
        "schema_version": "checkpoint-phases/v1", "evidence_class": "synthetic",
        "case_id": "complete-restore", "split": "development", "window_s": 100,
        **binding, "expected_shard_ids": ["shard-0", "shard-1"],
        "staging": [{**binding, "shard_id": shard, "started_s": 10, "completed_s": 20}
                    for shard in ("shard-0", "shard-1")],
        "commit": {**binding, "committed_s": 30},
        "restore": {**binding, "restore_attempt_id": "synthetic-restore-0", "started_s": 40,
                    "completed_s": 50, "outcome": "succeeded"},
    }
    missing_commit = copy.deepcopy(valid)
    missing_commit.update(case_id="staged-without-commit", split="exposed-acceptance", commit=None, restore=None)
    failed = copy.deepcopy(valid)
    failed.update(case_id="committed-restore-failed", split="exposed-acceptance")
    failed["restore"]["outcome"] = "failed"
    unknown = copy.deepcopy(valid)
    unknown.update(case_id="restore-outcome-unknown", split="exposed-acceptance")
    unknown["restore"].update(completed_s=None, outcome="unknown")
    intervals = {
        "schema_version": "checkpoint-blocked-intervals/v1", "evidence_class": "synthetic",
        "case_id": "overlap-conservation", "split": "exposed-acceptance",
        "workload_id": binding["workload_id"], "window_s": 100, "gpus": 2,
        "interval_coverage": "complete",
        "intervals": [{"interval_id": ident, "workload_id": binding["workload_id"],
                       "kind": kind, "start_s": a, "end_s": b}
                      for ident, kind, a, b in (("synthetic-block-0", "checkpoint", 20, 30),
                                               ("synthetic-block-1", "recovery", 25, 45))],
    }
    return {"phases": [valid, missing_commit, failed, unknown], "intervals": [intervals]}


def reference_report():
    cases = synthetic_cases()
    return {
        "schema_version": "checkpoint-reference/v1", "evidence_class": "synthetic",
        "generator": {"id": "literal-checkpoint-cases/v1", "seed": None,
                      "randomness": "none", "held_out": False},
        "inputs": cases,
        "phase_results": [evaluate_checkpoint(case) for case in cases["phases"]],
        "interval_results": [blocked_ledger(case) for case in cases["intervals"]],
        "limitations": ["Synthetic semantics only; no operator, storage or GPU performance evidence.",
                        "Commit is a declared fixture fact, not verified durability or an execution receipt.",
                        "No phase duration implies a blocked interval; blocked intervals are supplied separately.",
                        "Do not add or subtract this wall-clock ledger into retained-progress accounting.",
                        "No checkpoint size, CPU memory, storage bandwidth or stochastic process is inferred.",
                        "No executor, admission permission, reservation or authority is implemented."],
    }


if __name__ == "__main__":
    print(json.dumps(reference_report(), indent=2, sort_keys=True, allow_nan=False))
