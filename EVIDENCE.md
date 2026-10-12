# Evidence and provenance

Run from this repository root:

```sh
python tools/validate_evidence.py evidence.json
```

The manifest covers selected consequential inputs, headline results and contract
limitations, not incidental constants. Classifications apply to results separately
from inputs. A simulated result stays simulated even with calibrated inputs.

Local sources and artifacts are SHA-256 bound. A source change must trigger a
review of the dependent claims and regeneration before updating its hash. Do not
refresh hashes just to silence CI. Referenced source ledgers retain citation and
reuse qualifications; structural validation is not independent source verification.

The validator is a byte-identical vendored copy of research/evidence/validate.py,
so CI works without fetching an unpinned second repository. The research portfolio
check tests copy parity. Synchronize copies when changing the convention.

Qualitative contracts use a statement rather than manufacturing numerical results.
Numeric valid_bounds are basic sanity limits, not empirically supported ranges.

The economic publication check is `make research-claims`. It recomputes every
committed CSV row (22 rows, seeds 0–7), verifies the generated summary and all
324 sensitivity scenarios (972 policy runs), compares the exact manual/elastic
gaps and eight-spare value in the manifest, and binds
marked dollar comparisons in README, study and paper to those results. It also
refuses changed scenario defaults until the full assumption prose is reviewed.
`make figures` regenerates figures, tables and the matched-fault sensitivity
report together, then refreshes the marked comparisons. CI checks the existing
artifacts before any regeneration. PNGs are SHA-bound rather than byte-compared
across rendering-library versions. Provenance hashes are reviewed and resealed
separately after inspecting the regenerated outputs; no command silently reseals
them. The prior CI only ran the generator successfully and did not reject stale
committed tables or hand-maintained prose.

The 12 October 2026 review found the original public dollar arithmetic reproducible,
but re-confirmed the elastic repair-credit defect already corrected by Dmitry in
`077eb6f`. This branch reuses only that interval-credit fix and its arithmetic
witness; it does not accept or combine the unrelated pending event-order work.
See the study's economics section for prior and corrected snapshot values.
