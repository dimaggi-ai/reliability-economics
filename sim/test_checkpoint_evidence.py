"""Synthetic reference acceptance and negative cases; no held-out claims."""
import copy
import json

import pytest

from checkpoint_evidence import blocked_ledger, evaluate_checkpoint, reference_report, synthetic_cases


def phase():
    return synthetic_cases()["phases"][0]


def ledger():
    return synthetic_cases()["intervals"][0]


def test_phase_outcomes_remain_separate():
    results = {r["case_id"]: r for r in reference_report()["phase_results"]}
    assert results["complete-restore"]["restore_verified"] is True
    staged = results["staged-without-commit"]
    assert (staged["staging_status"], staged["commit_status"], staged["restore_verified"]) == ("complete", "missing", None)
    failed = results["committed-restore-failed"]
    assert (failed["evaluation"], failed["commit_status"], failed["restore_verified"]) == ("complete", "committed", False)
    assert results["restore-outcome-unknown"]["restore_verified"] is None
    assert all(r["execution_authorized"] is False for r in results.values())
    assert all(r["mutation_request"] is None for r in results.values())


def test_missing_shard_does_not_erase_reported_commit_or_establish_restore():
    record = phase()
    record["staging"].pop()
    result = evaluate_checkpoint(record)
    assert result["commit_status"] == "committed"
    assert result["staging_status"] == result["evaluation"] == "incomplete"
    assert result["restore_verified"] is None
    assert "staging record:shard-1" in result["missing_evidence"]


def test_unknown_commit_and_stage_times_remain_unknown():
    record = phase()
    record["staging"][0]["completed_s"] = None
    record["commit"]["committed_s"] = None
    result = evaluate_checkpoint(record)
    assert result["evaluation"] == "incomplete"
    assert result["commit_status"] == "unknown"
    assert result["restore_verified"] is None


@pytest.mark.parametrize("event", ["staging", "restore"])
def test_unknown_start_with_known_completion_remains_incomplete(event):
    record = phase()
    target = record[event][0] if event == "staging" else record[event]
    target["started_s"] = None
    result = evaluate_checkpoint(record)
    assert result["evaluation"] == "incomplete"
    assert result["restore_verified"] is None


def test_known_restore_completion_cannot_precede_commit_even_with_unknown_start():
    record = phase()
    record["restore"].update(started_s=None, completed_s=29)
    with pytest.raises(ValueError, match="precedes commit"):
        evaluate_checkpoint(record)


@pytest.mark.parametrize("key", ["workload_id", "checkpoint_id", "attempt_id"])
@pytest.mark.parametrize("event", ["staging", "commit", "restore"])
def test_identity_mismatches_refused(key, event):
    record = phase()
    target = record[event][0] if event == "staging" else record[event]
    target[key] = "another-id"
    with pytest.raises(ValueError, match="mismatched"):
        evaluate_checkpoint(record)


@pytest.mark.parametrize("mutation", [
    lambda r: r["staging"][0].update(completed_s=9),
    lambda r: r["commit"].update(committed_s=19),
    lambda r: r["restore"].update(started_s=29),
    lambda r: r["restore"].update(completed_s=39),
    lambda r: r["restore"].update(outcome="succeeded", completed_s=None),
    lambda r: r["restore"].update(outcome="unknown", completed_s=50),
    lambda r: r["restore"].update(restore_attempt_id="synthetic-save-0"),
    lambda r: r["staging"].append(copy.deepcopy(r["staging"][0])),
    lambda r: r["staging"][0].update(shard_id="unexpected"),
    lambda r: r.update(expected_shard_ids=["shard-0", "shard-0"]),
    lambda r: r.update(expected_shard_ids=[]),
    lambda r: r.update(extra="unsupported"),
])
def test_known_contradictions_and_malformed_relations_refused(mutation):
    record = phase()
    mutation(record)
    with pytest.raises(ValueError):
        evaluate_checkpoint(record)


@pytest.mark.parametrize("value", [True, False, "20", -1, float("nan"), float("inf"), 101, 10**1000])
def test_invalid_phase_times_refused(value):
    record = phase()
    record["staging"][0]["completed_s"] = value
    with pytest.raises(ValueError):
        evaluate_checkpoint(record)


@pytest.mark.parametrize("key,value", [
    ("schema_version", "future/v2"), ("evidence_class", "observed"),
    ("evidence_class", "benchmark"), ("split", "held-out"), ("split", []),
    ("window_s", 0), ("window_s", True), ("workload_id", " "),
])
@pytest.mark.parametrize("kind", ["phase", "ledger"])
def test_unknown_classes_and_invalid_common_fields_refused(key, value, kind):
    record = phase() if kind == "phase" else ledger()
    record[key] = value
    with pytest.raises(ValueError):
        (evaluate_checkpoint if kind == "phase" else blocked_ledger)(record)


def test_recovery_priority_conserves_exact_wallclock():
    result = blocked_ledger(ledger())
    assert result["total_gpu_s"] == 200
    assert result["buckets"] == {"recovery_gpu_s": 40, "checkpoint_only_gpu_s": 10, "healthy_gpu_s": 150}
    assert sum(result["buckets"].values()) == result["total_gpu_s"]
    assert result["accounting_basis"] == "wall-clock blocking, not retained progress"
    assert result["execution_authorized"] is False
    assert result["mutation_request"] is None


def test_interval_union_not_sum_and_input_order_irrelevant():
    record = ledger()
    record["intervals"].append({**record["intervals"][0], "interval_id": "nested", "start_s": 22, "end_s": 28})
    result = blocked_ledger(record)
    record["intervals"].reverse()
    assert blocked_ledger(record)["buckets"] == result["buckets"] == blocked_ledger(ledger())["buckets"]


def test_touching_intervals_and_full_window():
    record = ledger()
    record["intervals"][0].update(start_s=0, end_s=50)
    record["intervals"][1].update(start_s=50, end_s=100)
    assert blocked_ledger(record)["buckets"] == {"recovery_gpu_s": 100, "checkpoint_only_gpu_s": 100, "healthy_gpu_s": 0}


def test_empty_intervals_only_mean_healthy_with_complete_coverage():
    record = ledger()
    record["intervals"] = []
    assert blocked_ledger(record)["buckets"]["healthy_gpu_s"] == 200
    record["interval_coverage"] = "unknown"
    assert blocked_ledger(record)["buckets"] is None


@pytest.mark.parametrize("key", ["start_s", "end_s"])
def test_unknown_interval_endpoints_do_not_create_healthy_credit(key):
    record = ledger()
    record["intervals"][0][key] = None
    result = blocked_ledger(record)
    assert result["evaluation"] == "incomplete"
    assert result["buckets"] is None
    assert result["total_gpu_s"] == 200


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(gpus=True), lambda r: r.update(gpus=2.0), lambda r: r.update(gpus=0),
    lambda r: r.update(window_s=1e308, gpus=10),
    lambda r: r["intervals"][0].update(kind="upload"),
    lambda r: r["intervals"][0].update(workload_id="wrong"),
    lambda r: r["intervals"][0].update(start_s=True),
    lambda r: r["intervals"][0].update(end_s=float("nan")),
    lambda r: r["intervals"][0].update(start_s=30, end_s=20),
    lambda r: r["intervals"][0].update(start_s=20, end_s=20),
    lambda r: r["intervals"][0].update(end_s=101),
    lambda r: r["intervals"].append(copy.deepcopy(r["intervals"][0])),
    lambda r: r.update(interval_coverage="partial"),
])
def test_invalid_intervals_and_numeric_bounds_refused(mutation):
    record = ledger()
    mutation(record)
    with pytest.raises(ValueError):
        blocked_ledger(record)


def test_deterministic_synthetic_artifact_and_inputs_not_mutated():
    cases = synthetic_cases()
    before = copy.deepcopy(cases)
    for case in cases["phases"]:
        evaluate_checkpoint(case)
    for case in cases["intervals"]:
        blocked_ledger(case)
    assert cases == before
    report = reference_report()
    assert report == reference_report()
    assert report["evidence_class"] == "synthetic"
    assert report["generator"] == {"id": "literal-checkpoint-cases/v1", "seed": None, "randomness": "none", "held_out": False}
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_positive_mixed_integer_float_interval_cannot_disappear():
    record = ledger()
    record.update(window_s=2**53 + 1, gpus=1)
    record["intervals"] = [{**record["intervals"][0], "kind": "recovery",
                            "start_s": float(2**53), "end_s": 2**53 + 1}]
    result = blocked_ledger(record)
    assert result["evaluation"] == "complete"
    assert result["buckets"] == {"recovery_gpu_s": 1, "checkpoint_only_gpu_s": 0,
                                  "healthy_gpu_s": 2**53}
    assert sum(result["buckets"].values()) == result["total_gpu_s"] == 2**53 + 1
    json.dumps(result, allow_nan=False)


def test_large_integral_gpu_seconds_are_not_rounded_by_float_summation():
    record = ledger()
    record.update(window_s=1, gpus=2**53 + 1, intervals=[])
    result = blocked_ledger(record)
    assert result["buckets"]["healthy_gpu_s"] == result["total_gpu_s"] == 2**53 + 1


def test_smallest_positive_interval_keeps_nonzero_blocked_time():
    record = ledger()
    smallest = float.fromhex("0x0.0000000000001p-1022")
    record.update(window_s=smallest, gpus=1)
    record["intervals"] = [{**record["intervals"][0], "kind": "recovery", "start_s": 0, "end_s": smallest}]
    result = blocked_ledger(record)
    assert result["buckets"]["recovery_gpu_s"] == result["total_gpu_s"] == smallest
    assert result["buckets"]["healthy_gpu_s"] == 0


def test_finite_integer_factors_cannot_overflow_ledger_total():
    record = ledger()
    record.update(window_s=10**308, gpus=2, intervals=[])
    with pytest.raises(ValueError, match="arithmetic overflow"):
        blocked_ledger(record)
