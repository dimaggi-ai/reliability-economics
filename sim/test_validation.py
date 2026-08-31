"""Validation-registry tests. Run: python3 test_validation.py (seconds —
the event-driven sim makes the registry cheap).

These tests exist to stop the registry from flattering the model. Three
of them check the registry's own shape rather than its results, because
a registry whose only test re-runs its own assertions would still pass
if someone widened a tolerance to 1e9, moved an `expected` onto the
model's output, or quietly deleted a failing point.
"""

from validation import (
    DECLINED, ETTR_BOUND, ETTR_FLOOR, Point, ettr_in_band, points, validate,
    _ettr,
)

_POINTS = points()
_EXPECTED_POINTS = 6


def test_every_point_reproduces():
    for p in _POINTS:
        assert p.ok, (
            f"{p.name}: expected {p.expected} +/- {p.tolerance} {p.ref}, "
            f"model gives {p.actual:.4f}"
        )


def test_all_three_kinds_present():
    kinds = {p.kind for p in _POINTS}
    assert kinds == {"calibrated", "emergent", "sanity"}, kinds


def test_evidence_points_cite_sources_and_sanity_points_do_not():
    for p in _POINTS:
        if p.kind in ("calibrated", "emergent"):
            assert p.ref.startswith("["), f"{p.name} claims evidence, cites {p.ref!r}"
        else:
            assert p.ref == "-", f"{p.name} is sanity but cites {p.ref!r}"


def test_registry_has_not_silently_shrunk():
    # Deleting an inconvenient point is the cheapest way to make a
    # registry green; the README quotes this count.
    assert len(_POINTS) == _EXPECTED_POINTS, len(_POINTS)


def test_no_tolerance_is_wide_enough_to_be_meaningless():
    for p in _POINTS:
        # Every point is expressed against a non-zero reference so that
        # its band is readable as a percentage; a point whose expected
        # value is 0 hides how wide its tolerance really is.
        assert p.expected, f"{p.name}: express the point as a ratio, not vs 0"
        assert abs(p.tolerance / p.expected) <= 0.20, (
            f"{p.name}: +/-{p.tolerance} is "
            f"{p.tolerance / abs(p.expected):.0%} of the expected value"
        )


def test_negative_control_an_impossible_model_fails_the_ettr_anchor():
    """The one test that can catch an OPTIMISTIC model.

    Every other point fails only when the model is too pessimistic. A
    config with zero detection and zero reload — recovery in no time at
    all — scores HIGHER than the shipped model, so it would sail through
    any two-sided band drawn around Meta's published lower bound. It
    must fail, or the anchor is not evidence of anything.
    """
    impossible = _ettr(detect_h=0.0, reload_h=0.0)
    assert impossible >= ETTR_BOUND, (
        f"expected an instantaneous-recovery model to beat the published "
        f"bound; it scored {impossible:.4f}"
    )
    assert not ettr_in_band(impossible), (
        f"a zero-latency-recovery model scores {impossible:.4f} and still "
        f"lands inside [{ETTR_FLOOR}, {ETTR_BOUND}) — the anchor cannot "
        f"discriminate"
    )


def test_negative_control_a_sluggish_model_also_fails_the_ettr_anchor():
    """...and the band must still catch a model that is too pessimistic."""
    assert not ettr_in_band(_ettr(reload_h=0.5)), "band is too wide below"


def test_declined_anchors_are_disclosed():
    assert DECLINED, "the registry must name the anchors it does not check"
    for anchor, why in DECLINED:
        assert anchor.startswith("["), anchor
        assert len(why) > 40, anchor


def test_validate_reports_all_ok():
    pts, ok = validate()
    assert ok
    assert len(pts) == _EXPECTED_POINTS
    assert all(isinstance(p, Point) for p in pts)


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        globals()[fn]()
        print(f"ok {fn}")
    print("all validation tests passed")
