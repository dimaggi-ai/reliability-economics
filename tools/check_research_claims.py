"""Reproduce committed tables and check the economic claims editors can reach.

Default is a check. --write refreshes marked prose only after the committed CSV
and generated summary reproduce; it never rewrites the simulator or result data.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from decimal import Decimal
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
from run import write_results  # noqa: E402
from reliability_sim import Config, NODE_GPUS  # noqa: E402
from recovery_sensitivity import experiment  # noqa: E402

START = "<!-- economics-headline:start -->"
END = "<!-- economics-headline:end -->"
DOCUMENTS = ("README.md", "docs/study.md", "paper/paper.md")
# Changes to these assumptions require editorial review of the full study,
# not just mechanically refreshing its output numbers.
PUBLISHED_CONFIG = {
    "nodes": 2048, "horizon_h": 720.0, "seed": 0,
    "dollars_per_gpu_h": 2.5, "single_rate_per_node_h": 1.58e-4,
    "burst_rate_per_h": 0.0, "burst_blast": 1,
    "detect_h": 0.033, "reload_h": 0.25, "repair_h": 4.0,
    "requal_h": 1.0, "operator_h": 1.0, "reshard_h": 0.5,
    "ckpt_interval_h": 0.5, "ckpt_write_h": 0.02,
    "ckpt_overlap_interval_h": 0.1, "ckpt_overlap_write_h": 0.003,
    "spare_nodes": 0, "legal_shape_nodes": 8,
}


def headline_amounts(csv_text):
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    def amount(policy, spares):
        matches = [row for row in rows if row["scenario"] == "independent"
                   and row["policy"] == policy and row["spare_nodes"] == str(spares)]
        if len(matches) != 1:
            raise ValueError(f"expected one independent {policy}/{spares} row")
        return Decimal(matches[0]["waste_dollars"])
    return (amount("manual", 0), amount("elastic", 0),
            amount("auto-spares-ckpt", 8))


def claim_block(csv_text):
    manual, elastic, spares = headline_amounts(csv_text)
    return "\n".join((
        START,
        "**Committed independent-failure scenario:** 16,384 active GPUs, 30 days "
        "(720 hours), seeds 0–7, valued at an assumed $2.50/GPU-hour.",
        "",
        f"Mean modeled nonproductive-capacity value: manual/no spares "
        f"**${manual:,.3f}**; elastic/no spares **${elastic:,.3f}**; "
        f"automated recovery with eight spare nodes and overlapped checkpointing "
        f"**${spares:,.3f}**.",
        "",
        f"Subtracting the committed `waste_dollars` values gives "
        f"**${manual - spares:,.3f}** versus manual (about $21M), and "
        f"**${elastic - spares:,.3f}** versus elastic "
        f"(**${elastic - spares:,.0f}** rounded to whole dollars). "
        "These are modeled capacity valuations, not realized cash savings.",
        END,
    ))


def update_block(text, block):
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("expected exactly one marked economics block")
    before, rest = text.split(START)
    old, after = rest.split(END)
    return before + block + after


def check(root, generated_csv, generated_summary, generated_sensitivity, write=False):
    root = Path(root)
    errors = []
    if asdict(Config()) != PUBLISHED_CONFIG or NODE_GPUS != 8:
        return ["published scenario assumptions changed: review prose and baseline contract before refreshing claims"]
    for name, expected in (("results/results.csv", generated_csv),
                           ("results/summary.md", generated_summary),
                           ("results/recovery-sensitivity.json", generated_sensitivity)):
        if (root / name).read_text() != expected:
            errors.append(f"{name}: committed output differs from deterministic regeneration")
    if errors:
        return errors  # Do not propagate a stale result into prose.
    manual, elastic, spares = headline_amounts(generated_csv)
    expected_claims = {
        "manual-to-eight-spare difference": manual - spares,
        "elastic-to-eight-spare difference": elastic - spares,
        "eight-spare nonproductive capacity value": spares,
    }
    manifest = json.loads((root / "evidence.json").read_text())
    for name, expected in expected_claims.items():
        matches = [row for row in manifest.get("results", []) if row.get("name") == name]
        if (len(matches) != 1 or type(matches[0].get("value")) not in (int, float)
                or Decimal(str(matches[0]["value"])) != expected):
            errors.append(f"evidence.json: {name} differs from reproduced CSV arithmetic")
    if errors:
        return errors  # Hash resealing must not hide a stale numerical claim.
    block = claim_block(generated_csv)
    for name in DOCUMENTS:
        path = root / name
        text = path.read_text()
        try:
            updated = update_block(text, block)
        except ValueError as error:
            errors.append(f"{name}: {error}")
            continue
        if text != updated:
            if write:
                path.write_text(updated)
            else:
                errors.append(f"{name}: economic claims differ from reproduced results")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    scratch = ROOT / ".pytest_cache" / "research-claims"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        generated = Path(temporary)
        write_results(range(8), generated)
        sensitivity = json.dumps(experiment(), indent=2, allow_nan=False) + "\n"
        errors = check(ROOT, (generated / "results.csv").read_text(),
                       (generated / "summary.md").read_text(), sensitivity, args.write)
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print("22 result rows (176 seeded policy runs), 324 sensitivity scenarios (972 policy runs), "
              "summary, three manifest values and three economic claim blocks agree")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
